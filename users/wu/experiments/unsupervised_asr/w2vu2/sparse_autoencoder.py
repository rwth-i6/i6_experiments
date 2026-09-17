import json
import os

from sisyphus import Job, Task


def normalize_frames(x):
    import torch.nn.functional as F

    return F.normalize(x.float(), p=2, dim=-1)


def build_model(input_dim=1024, num_features=8192, k=50):
    import torch
    from torch import nn

    class BatchTopKSAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(num_features, input_dim, bias=False)
            with torch.no_grad():
                weight = self.decoder.weight
                weight.div_(weight.norm(dim=0, keepdim=True) + torch.finfo(weight.dtype).eps)
            self.encoder = nn.Linear(input_dim, num_features)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()
            self.b_dec = nn.Parameter(torch.zeros(input_dim))

        def encode(self, x):
            activations = torch.relu(self.encoder(x - self.b_dec))
            values, indices = activations.flatten().topk(k * x.shape[0], sorted=False)
            return torch.zeros_like(activations).flatten().scatter(0, indices, values).view_as(activations)

        def decode(self, z):
            return self.decoder(z) + self.b_dec

        def forward(self, x):
            return self.decode(self.encode(x))

    return BatchTopKSAE()


def load_model(checkpoint_path, device="cpu"):
    import torch

    checkpoint = torch.load(os.fspath(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def learning_rate(update):
    return 2e-4 * min(1.0, (200000 - update) / 40000)


def train(feature_hdfs, checkpoint_path, metrics_path):
    import h5py
    import numpy as np
    import torch

    torch.manual_seed(0)
    model_config = {"input_dim": 1024, "num_features": 8192, "k": 50}
    model = build_model(**model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    sampler = torch.Generator().manual_seed(0)
    completed_updates = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        completed_updates = checkpoint["completed_updates"]
        del checkpoint

    sizes = []
    utterance_ids = []
    for path in feature_hdfs:
        with h5py.File(path, "r") as hdf:
            sizes.append(len(hdf["inputs"]))
            utterance_ids.extend(hdf["seqTags"].asstr()[:].tolist())
    num_utterances = len(utterance_ids)
    frames = np.empty((sum(sizes), 1024), dtype=np.float16)
    offset = 0
    for path, size in zip(feature_hdfs, sizes):
        with h5py.File(path, "r") as hdf:
            hdf["inputs"].read_direct(frames, dest_sel=np.s_[offset:offset + size])
        offset += size
    print(f"cache loaded: utterances={num_utterances} frames={len(frames)} storage=float16 compute=float32", flush=True)

    def save_checkpoint():
        state = {
            "model": model.state_dict(), "model_config": model_config,
            "optimizer": optimizer.state_dict(), "completed_updates": completed_updates,
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "source_hdfs": feature_hdfs, "source_frames": len(frames), "source_utterances": num_utterances,
            "source_utterance_ids": utterance_ids,
            "source_speaker_ids": sorted({utt.split("-")[0] for utt in utterance_ids}),
        }
        torch.save(state, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)

    print(f"training from update={completed_updates} to=200000", flush=True)
    model.train()
    for update in range(completed_updates + 1, 200001):
        indices = torch.randint(len(frames), (2500,), generator=sampler).numpy()
        x = normalize_frames(torch.from_numpy(frames[indices]).cuda().float())
        optimizer.param_groups[0]["lr"] = learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        loss = (model(x) - x).square().sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        completed_updates = update
        if update % 1000 == 0:
            save_checkpoint()
            print(f"update={update} normalized_squared_l2={loss.item():.8f} lr={learning_rate(update):.8g}", flush=True)

    model.eval()
    squared_error = torch.zeros((), device="cuda", dtype=torch.float64)
    active_counts = torch.zeros(8192, device="cuda", dtype=torch.int64)
    with torch.no_grad():
        for start in range(0, len(frames), 2500):
            x = normalize_frames(torch.from_numpy(frames[start:start + 2500]).cuda().float())
            codes = model.encode(x)
            squared_error += (model.decode(codes) - x).square().sum(dim=-1).double().sum()
            active_counts += (codes > 0).sum(dim=0)
    metrics = {
        "completed_updates": completed_updates,
        "frame_presentations": completed_updates * 2500,
        "source_hdfs": feature_hdfs, "source_utterances": num_utterances, "source_frames": len(frames),
        "input_dim": 1024, "num_features": 8192, "k": 50, "seed": 0,
        "diagnostic_operating_point": "final endpoint; full train pool in stored order; global top50*B in groups B<=2500",
        "mean_normalized_input_squared_l2_error": squared_error.item() / len(frames),
        "mean_active_features_per_frame": active_counts.sum().item() / len(frames),
        "dead_features_on_full_train_pool": (active_counts == 0).sum().item(),
        "feature_active_frame_counts": active_counts.cpu().tolist(),
        "storage_dtype": "float16", "compute_dtype": "float32",
        "torch_version": str(torch.__version__),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "activation_normalization": "per-frame L2", "decoder_normalization": "initialization only",
    }
    with open(metrics_path + ".tmp", "w") as stream:
        json.dump(metrics, stream, indent=2)
    os.replace(metrics_path + ".tmp", metrics_path)
    print(f"final diagnostics: normalized_squared_l2={metrics['mean_normalized_input_squared_l2_error']:.8f} dead={metrics['dead_features_on_full_train_pool']}", flush=True)


class BatchTopKTrainingJob(Job):
    def __init__(self, *, feature_hdfs, artifact_root):
        self.feature_hdfs = list(feature_hdfs)
        self.artifact_root = artifact_root
        self.out_checkpoint = self.output_path("checkpoint.pt")
        self.out_metrics = self.output_path("metrics.json")
        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 64, "time": 11.5}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        artifact_dir = os.path.join(self.artifact_root, self._sis_id())
        os.makedirs(artifact_dir, exist_ok=True)
        paths = []
        for output in (self.out_checkpoint, self.out_metrics):
            path = os.path.join(artifact_dir, os.path.basename(output.get_path()))
            if not os.path.lexists(output.get_path()):
                os.symlink(path, output.get_path())
            paths.append(path)
        train([path.get_path() for path in self.feature_hdfs], *paths)
