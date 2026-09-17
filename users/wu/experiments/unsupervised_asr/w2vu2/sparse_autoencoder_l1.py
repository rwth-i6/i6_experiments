import json
import os
import time

from sisyphus import Job, Task


def normalize_frames(x, scale):
    return x.float() * scale


def compute_input_scale(frames):
    import numpy as np

    squared_norm_sum = np.float64(0)
    for start in range(0, len(frames), 2500):
        batch = frames[start:start + 2500].astype(np.float64)
        squared_norm_sum += np.square(batch).sum(dtype=np.float64)
    return float(np.sqrt(frames.shape[1] / (squared_norm_sum / len(frames))))


def build_model(input_dim=1024, num_features=8192):
    import torch
    from torch import nn

    class WeightedL1SAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = nn.Linear(num_features, input_dim, bias=False)
            with torch.no_grad():
                weight = self.decoder.weight
                weight.div_(weight.norm(dim=0, keepdim=True) + torch.finfo(weight.dtype).eps)
                weight.mul_(0.1)
            self.encoder = nn.Linear(input_dim, num_features)
            with torch.no_grad():
                self.encoder.weight.copy_(self.decoder.weight.T)
                self.encoder.bias.zero_()
            self.b_dec = nn.Parameter(torch.zeros(input_dim))

        def encode(self, x):
            return torch.relu(self.encoder(x))

        def decode(self, z):
            return self.decoder(z) + self.b_dec

        def forward(self, x):
            return self.decode(self.encode(x))

    return WeightedL1SAE()


def load_model(checkpoint_path, device="cpu"):
    import torch

    checkpoint = torch.load(os.fspath(checkpoint_path), map_location="cpu", weights_only=False)
    model = build_model(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.input_scale = float(checkpoint["input_scale"])
    return model.to(device).eval()


def learning_rate(update):
    return 5e-5 * min(1.0, (200000 - update) / 40000)


def sparsity_coefficient(update):
    return 5.0 * min(1.0, update / 10000)


def loss_terms(model, x):
    codes = model.encode(x)
    reconstruction = (model.decode(codes) - x).square().sum(dim=-1).mean()
    weighted_l1 = (codes * model.decoder.weight.norm(dim=0)).sum(dim=-1).mean()
    return reconstruction, weighted_l1


def train(feature_hdfs, checkpoint_path, metrics_path):
    import h5py
    import numpy as np
    import torch

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model_config = {"input_dim": 1024, "num_features": 8192}
    model = build_model(**model_config).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    sampler = torch.Generator().manual_seed(0)
    completed_updates = 0
    training_seconds = 0.0
    input_scale = None
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.set_state(checkpoint["sampler_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        completed_updates = checkpoint["completed_updates"]
        training_seconds = checkpoint["training_seconds"]
        input_scale = checkpoint["input_scale"]
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
    if input_scale is None:
        input_scale = compute_input_scale(frames)
    model.input_scale = input_scale
    print(f"cache loaded: utterances={num_utterances} frames={len(frames)} input_scale={input_scale:.12g} storage=float16 compute=float32", flush=True)

    metadata = {
        "sae_type": "weighted_l1",
        "objective": "mean_frame[sum_dim((reconstruction-x)^2) + lambda*sum_feature(ReLU(encoder(x))*decoder_column_norm)]",
        "lambda_target": 5.0, "lambda_warmup_updates": 10000,
        "activation_normalization": "global train-only scale sqrt(input_dim / mean_train_raw_squared_norm)",
        "input_scale": input_scale, "normalization_source": "all training frames; no dev or gold",
        "decoder_initial_column_norm": 0.1,
        "decoder_initial_direction_distribution": "inherited torch Linear uniform initialization, normalized by column",
        "encoder_initialization": "decoder transpose",
        "bias_initialization": "zero", "decoder_normalization": "initialization only",
        "optimizer_config": {"name": "Adam", "lr": 5e-5, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0},
        "gradient_clip_global_norm": 1.0,
        "lr_decay_start_update": 160000, "max_updates": 200000,
        "lr_schedule": "constant through 160000; linear decay to zero at 200000",
        "batch_size": 2500, "sampling": "uniform with replacement", "seed": 0,
        "auxiliary_loss": False, "feature_resampling": False,
        "source_hdfs": feature_hdfs, "source_frames": len(frames), "source_utterances": num_utterances,
        "storage_dtype": "float16", "compute_dtype": "float32",
        "torch_version": str(torch.__version__),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    training_start = time.monotonic()

    def save_checkpoint():
        state = {
            **metadata,
            "model": model.state_dict(), "model_config": model_config,
            "optimizer": optimizer.state_dict(), "completed_updates": completed_updates,
            "training_seconds": training_seconds + time.monotonic() - training_start,
            "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "source_utterance_ids": utterance_ids,
            "source_speaker_ids": sorted({utt.split("-")[0] for utt in utterance_ids}),
        }
        torch.save(state, checkpoint_path + ".tmp")
        os.replace(checkpoint_path + ".tmp", checkpoint_path)

    print(f"training from update={completed_updates} to=200000", flush=True)
    model.train()
    for update in range(completed_updates + 1, 200001):
        indices = torch.randint(len(frames), (2500,), generator=sampler).numpy()
        x = normalize_frames(torch.from_numpy(frames[indices]).cuda(), input_scale)
        optimizer.param_groups[0]["lr"] = learning_rate(update)
        optimizer.zero_grad(set_to_none=True)
        reconstruction, weighted_l1 = loss_terms(model, x)
        loss = reconstruction + sparsity_coefficient(update) * weighted_l1
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed_updates = update
        if update % 1000 == 0:
            save_checkpoint()
            print(f"update={update} reconstruction={reconstruction.item():.8f} weighted_l1={weighted_l1.item():.8f} lambda={sparsity_coefficient(update):.8g} lr={learning_rate(update):.8g}", flush=True)
    training_seconds += time.monotonic() - training_start

    model.eval()
    squared_error = torch.zeros((), device="cuda", dtype=torch.float64)
    input_squared_norm = torch.zeros((), device="cuda", dtype=torch.float64)
    weighted_l1_sum = torch.zeros((), device="cuda", dtype=torch.float64)
    active_counts = torch.zeros(8192, device="cuda", dtype=torch.int64)
    l0_histogram = torch.zeros(8193, device="cuda", dtype=torch.int64)
    with torch.no_grad():
        decoder_norms = model.decoder.weight.norm(dim=0)
        for start in range(0, len(frames), 2500):
            x = normalize_frames(torch.from_numpy(frames[start:start + 2500]).cuda(), input_scale)
            codes = model.encode(x)
            squared_error += (model.decode(codes) - x).square().sum(dim=-1).double().sum()
            input_squared_norm += x.square().sum(dim=-1).double().sum()
            weighted_l1_sum += (codes * decoder_norms).sum(dim=-1).double().sum()
            active = codes > 0
            active_counts += active.sum(dim=0)
            l0_histogram += torch.bincount(active.sum(dim=-1), minlength=8193)
    histogram = l0_histogram.cpu().numpy()
    cumulative = histogram.cumsum()
    quantiles = {}
    for name, quantile in (("min", 0.0), ("median", 0.5), ("p90", 0.9), ("max", 1.0)):
        position = quantile * (len(frames) - 1)
        lower = np.searchsorted(cumulative, int(np.floor(position)), side="right")
        upper = np.searchsorted(cumulative, int(np.ceil(position)), side="right")
        quantiles[name] = float(lower + (upper - lower) * (position - np.floor(position)))
    mean_reconstruction = squared_error.item() / len(frames)
    mean_weighted_l1 = weighted_l1_sum.item() / len(frames)
    metrics = {
        **metadata,
        "completed_updates": completed_updates, "training_seconds": training_seconds,
        "frame_presentations": completed_updates * 2500,
        "input_dim": 1024, "num_features": 8192,
        "diagnostic_operating_point": "fixed final endpoint; full train pool; direct ReLU, no TopK or threshold",
        "mean_reconstruction_squared_l2": mean_reconstruction,
        "mean_weighted_l1": mean_weighted_l1,
        "lambda_at_endpoint": sparsity_coefficient(completed_updates),
        "mean_loss": mean_reconstruction + sparsity_coefficient(completed_updates) * mean_weighted_l1,
        "normalized_mse": squared_error.item() / input_squared_norm.item(),
        "normalized_mse_definition": "sum reconstruction squared error / sum scaled input squared norm",
        "mean_scaled_input_squared_norm": input_squared_norm.item() / len(frames),
        "mean_active_features_per_frame": active_counts.sum().item() / len(frames),
        "dead_features_on_full_train_pool": (active_counts == 0).sum().item(),
        "decoder_column_norms": {
            "min": decoder_norms.min().item(),
            "median": float(np.median(decoder_norms.cpu().numpy())),
            "max": decoder_norms.max().item(),
            "zero_count": (decoder_norms == 0).sum().item(),
        },
        "feature_active_frame_counts": active_counts.cpu().tolist(),
        "l0_histogram": histogram.tolist(), "l0_quantiles": quantiles,
        "l0_quantile_method": "linear interpolation between adjacent ordered frames",
        "zero_activation_frame_fraction": int(histogram[0]) / len(frames),
    }
    with open(metrics_path + ".tmp", "w") as stream:
        json.dump(metrics, stream, indent=2)
    os.replace(metrics_path + ".tmp", metrics_path)
    print(f"final diagnostics: reconstruction={mean_reconstruction:.8f} weighted_l1={mean_weighted_l1:.8f} mean_l0={metrics['mean_active_features_per_frame']:.4f} dead={metrics['dead_features_on_full_train_pool']}", flush=True)


class WeightedL1TrainingJob(Job):
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
