"""
HuBERT layer 6 feature extraction, K-Means clustering, and run-length collapsed token generation.
"""

from typing import Optional, List, Dict, Iterator, Set, Union
import os
import glob
import logging
import numpy as np

from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase


SPIN_HUBERT_CKPT_URLS = {
    128: "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_128.ckpt",
    256: "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_256.ckpt",
    512: "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_512.ckpt",
    1024: "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_1024.ckpt",
    2048: "https://huggingface.co/datasets/vectominist/spin_ckpt/resolve/main/spin_hubert_2048.ckpt",
}


class DownloadSpinHuBERTCheckpointJob(Job):
    """
    Downloads a pre-trained Spin-HuBERT PyTorch Lightning checkpoint (.ckpt)
    from vectominist/spin_ckpt on Hugging Face (Chang et al., Interspeech 2023, arXiv:2305.11072).
    Runs as a mini_task on the local engine so internet access is available.
    """

    def __init__(
        self,
        codebook_size: int = 256,
        ckpt_url: Optional[str] = None,
    ):
        super().__init__()
        self.codebook_size = int(codebook_size)
        self.ckpt_url = ckpt_url or SPIN_HUBERT_CKPT_URLS[self.codebook_size]
        self.out_ckpt = self.output_path(f"spin_hubert_{self.codebook_size}.ckpt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        import shutil
        import subprocess
        import urllib.request

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        target_path = self.out_ckpt.get_path()
        tmp_path = target_path + ".tmp"
        logging.info(f"Downloading Spin-HuBERT (codebook={self.codebook_size}) from {self.ckpt_url} to {target_path}...")

        try:
            subprocess.run(
                ["wget", "-q", "-O", tmp_path, self.ckpt_url],
                check=True,
            )
        except Exception as e_wget:
            logging.warning(f"wget failed ({e_wget}), falling back to urllib.request...")
            urllib.request.urlretrieve(self.ckpt_url, tmp_path)

        if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 1_000_000:
            raise RuntimeError(f"Downloaded checkpoint {tmp_path} is missing or unexpectedly small!")

        shutil.move(tmp_path, target_path)
        logging.info(
            f"Successfully downloaded Spin-HuBERT checkpoint to {target_path} "
            f"({os.path.getsize(target_path) / (1024 ** 2):.1f} MB)."
        )


class ExtractHuBERTLayerFeaturesJob(Job):
    """
    Extracts intermediate representations from HuBERT-base (e.g., Layer 6)
    or Spin fine-tuned HuBERT (e.g., Layer 12, arXiv:2305.11072).
    """

    __sis_hash_exclude__ = {
        "cache_dir": "/work/smt4/benjamin.oyarzun/.cache/huggingface",
        "batch_size": 16,
        "spin_ckpt": None,
    }

    def __init__(
        self,
        audio_dir: Optional[tk.Path] = None,
        audio_tsv_manifest: Optional[tk.Path] = None,
        model_name: str = "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        layer: int = 6,
        batch_size: int = 16,
        cache_dir: Optional[str] = "/work/smt4/benjamin.oyarzun/.cache/huggingface",
        spin_ckpt: Optional[tk.Path] = None,
    ):
        super().__init__()
        assert audio_dir is not None or audio_tsv_manifest is not None, "Either audio_dir or audio_tsv_manifest must be given"
        self.audio_dir = audio_dir
        self.audio_tsv_manifest = audio_tsv_manifest
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.spin_ckpt = spin_ckpt

        self.out_features_npy = self.output_path("features.npy")
        self.out_lengths_txt = self.output_path("lengths.txt")
        self.out_seq_tags_txt = self.output_path("seq_tags.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 12})

    @staticmethod
    def _load_spin_weights_into_hf_hubert(model, ckpt_path: str) -> None:
        import pickle
        import torch

        class _DummyCheckpointObject:
            def __init__(self, *args, **kwargs):
                pass

            def __setstate__(self, state):
                pass

            def __call__(self, *args, **kwargs):
                return self

        class _SafeLightningUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                try:
                    return super().find_class(module, name)
                except (ModuleNotFoundError, AttributeError, ImportError):
                    return _DummyCheckpointObject

        class _SafePickleModule:
            Unpickler = _SafeLightningUnpickler
            load = staticmethod(pickle.load)
            loads = staticmethod(pickle.loads)
            UnpicklingError = pickle.UnpicklingError

        logging.info(f"Loading Spin fine-tuned HuBERT weights from {ckpt_path}...")
        try:
            ckpt_obj = torch.load(
                ckpt_path,
                map_location="cpu",
                pickle_module=_SafePickleModule,
                weights_only=False,
            )
        except TypeError:
            ckpt_obj = torch.load(
                ckpt_path,
                map_location="cpu",
                pickle_module=_SafePickleModule,
            )

        if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj:
            raw_sd = ckpt_obj["state_dict"]
        elif isinstance(ckpt_obj, dict) and "model" in ckpt_obj:
            raw_sd = ckpt_obj["model"]
        else:
            raw_sd = ckpt_obj

        # Normalize Spin Lightning key prefixes ('encoder.model.*' -> 'encoder.*')
        spin_sd = {}
        for k, v in raw_sd.items():
            if k.startswith("encoder.model."):
                spin_sd["encoder." + k[len("encoder.model."):]] = v
            elif k.startswith("model.encoder.model."):
                spin_sd["encoder." + k[len("model.encoder.model."):]] = v
            else:
                spin_sd[k] = v

        model_sd = model.state_dict()
        mapping = {
            "masked_spec_embed": "encoder.mask_emb",
            "encoder.layer_norm.bias": "encoder.encoder.layer_norm.bias",
            "encoder.layer_norm.weight": "encoder.encoder.layer_norm.weight",
            "encoder.pos_conv_embed.conv.bias": "encoder.encoder.pos_conv.0.bias",
            "feature_projection.layer_norm.bias": "encoder.layer_norm.bias",
            "feature_projection.layer_norm.weight": "encoder.layer_norm.weight",
            "feature_projection.projection.bias": "encoder.post_extract_proj.bias",
            "feature_projection.projection.weight": "encoder.post_extract_proj.weight",
        }
        if "encoder.pos_conv_embed.conv.parametrizations.weight.original0" in model_sd:
            mapping["encoder.pos_conv_embed.conv.parametrizations.weight.original0"] = "encoder.encoder.pos_conv.0.weight_g"
            mapping["encoder.pos_conv_embed.conv.parametrizations.weight.original1"] = "encoder.encoder.pos_conv.0.weight_v"
        else:
            mapping["encoder.pos_conv_embed.conv.weight_g"] = "encoder.encoder.pos_conv.0.weight_g"
            mapping["encoder.pos_conv_embed.conv.weight_v"] = "encoder.encoder.pos_conv.0.weight_v"

        for layer_idx in range(12):
            for proj in ("q", "k", "v", "out"):
                mapping[f"encoder.layers.{layer_idx}.attention.{proj}_proj.weight"] = (
                    f"encoder.encoder.layers.{layer_idx}.self_attn.{proj}_proj.weight"
                )
                mapping[f"encoder.layers.{layer_idx}.attention.{proj}_proj.bias"] = (
                    f"encoder.encoder.layers.{layer_idx}.self_attn.{proj}_proj.bias"
                )
            mapping[f"encoder.layers.{layer_idx}.layer_norm.weight"] = (
                f"encoder.encoder.layers.{layer_idx}.self_attn_layer_norm.weight"
            )
            mapping[f"encoder.layers.{layer_idx}.layer_norm.bias"] = (
                f"encoder.encoder.layers.{layer_idx}.self_attn_layer_norm.bias"
            )
            mapping[f"encoder.layers.{layer_idx}.feed_forward.intermediate_dense.weight"] = (
                f"encoder.encoder.layers.{layer_idx}.fc1.weight"
            )
            mapping[f"encoder.layers.{layer_idx}.feed_forward.intermediate_dense.bias"] = (
                f"encoder.encoder.layers.{layer_idx}.fc1.bias"
            )
            mapping[f"encoder.layers.{layer_idx}.feed_forward.output_dense.weight"] = (
                f"encoder.encoder.layers.{layer_idx}.fc2.weight"
            )
            mapping[f"encoder.layers.{layer_idx}.feed_forward.output_dense.bias"] = (
                f"encoder.encoder.layers.{layer_idx}.fc2.bias"
            )
            mapping[f"encoder.layers.{layer_idx}.final_layer_norm.weight"] = (
                f"encoder.encoder.layers.{layer_idx}.final_layer_norm.weight"
            )
            mapping[f"encoder.layers.{layer_idx}.final_layer_norm.bias"] = (
                f"encoder.encoder.layers.{layer_idx}.final_layer_norm.bias"
            )

        for conv_idx in range(7):
            mapping[f"feature_extractor.conv_layers.{conv_idx}.conv.weight"] = (
                f"encoder.feature_extractor.conv_layers.{conv_idx}.0.weight"
            )
            if conv_idx == 0:
                mapping[f"feature_extractor.conv_layers.0.layer_norm.weight"] = (
                    "encoder.feature_extractor.conv_layers.0.2.weight"
                )
                mapping[f"feature_extractor.conv_layers.0.layer_norm.bias"] = (
                    "encoder.feature_extractor.conv_layers.0.2.bias"
                )

        mapped_sd = {}
        missing_in_spin = []
        for hf_key, spin_key in mapping.items():
            if spin_key in spin_sd:
                mapped_sd[hf_key] = spin_sd[spin_key]
            else:
                missing_in_spin.append((hf_key, spin_key))

        if len(mapped_sd) < 190:
            raise RuntimeError(
                f"Failed to map Spin checkpoint keys! Only mapped {len(mapped_sd)}/{len(mapping)} keys. "
                f"Sample spin_sd keys: {list(spin_sd.keys())[:10]}"
            )
        if missing_in_spin:
            logging.warning(f"Keys not present in Spin ckpt (kept from base HuBERT): {missing_in_spin}")

        incompatible = model.load_state_dict(mapped_sd, strict=False)
        logging.info(
            f"Successfully loaded {len(mapped_sd)}/{len(mapping)} Spin-HuBERT parameter tensors "
            f"(unexpected={len(incompatible.unexpected_keys)}, missing={len(incompatible.missing_keys)})."
        )

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import tempfile
        import shutil
        import torch
        import torchaudio
        from transformers import AutoModel, AutoConfig
        from numpy.lib.format import open_memmap

        os.environ["HF_HOME"] = self.cache_dir or "/work/smt4/benjamin.oyarzun/.cache/huggingface"
        os.environ["TRANSFORMERS_CACHE"] = self.cache_dir or "/work/smt4/benjamin.oyarzun/.cache/huggingface"

        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA is not available on {os.uname().nodename}! Failing fast instead of falling back to CPU.")
        device = torch.device("cuda")
        logging.info(f"Loading model: {self.model_name} onto {device}...")

        if os.path.isdir(self.model_name):
            config = AutoConfig.from_pretrained(self.model_name, local_files_only=True)
            config.output_hidden_states = True
            model = AutoModel.from_pretrained(self.model_name, config=config, local_files_only=True)
        else:
            config = AutoConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir)
            config.output_hidden_states = True
            model = AutoModel.from_pretrained(self.model_name, config=config, cache_dir=self.cache_dir)

        if self.spin_ckpt is not None:
            ckpt_path = self.spin_ckpt.get_path() if hasattr(self.spin_ckpt, "get_path") else str(self.spin_ckpt)
            self._load_spin_weights_into_hf_hubert(model, ckpt_path)

        model.to(device)
        model.eval()

        audio_files = []
        if self.audio_dir is not None:
            root = self.audio_dir.get_path()
            for r, dirs, files in os.walk(root):
                for f in files:
                    if f.endswith(".flac") or f.endswith(".wav"):
                        audio_files.append(os.path.join(r, f))
            audio_files.sort()
        else:
            with open(self.audio_tsv_manifest.get_path(), "r") as f:
                lines = [line.strip() for line in f.readlines()]
            root_dir = lines[0] if lines else ""
            for entry in lines[1:]:
                rel_path = entry.split("\t")[0]
                full_path = os.path.join(root_dir, rel_path) if not os.path.isabs(rel_path) else rel_path
                audio_files.append(full_path)

        temp_dir = tempfile.mkdtemp(prefix="hubert_feats_")
        chunk_files = []
        lengths = []
        seq_tags = []
        total_frames = 0
        feat_dim = 768

        batch_feats = []
        batch_frames = 0
        chunk_idx = 0

        with torch.no_grad():
            for i, full_path in enumerate(audio_files):
                tag = os.path.splitext(os.path.basename(full_path))[0]
                seq_tags.append(tag)

                waveform, sample_rate = torchaudio.load(full_path)
                if sample_rate != 16000:
                    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

                if waveform.shape[0] > 1:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)

                input_values = waveform.to(device)
                outputs = model(input_values=input_values)
                # hidden_states contains (layer 0 embedding, layer 1, ..., layer 12)
                # Layer 6 is outputs.hidden_states[6]
                layer_feat = outputs.hidden_states[self.layer].squeeze(0).cpu().numpy().astype(np.float32)  # (T, d)
                feat_dim = layer_feat.shape[1]
                t_len = layer_feat.shape[0]
                lengths.append(t_len)
                total_frames += t_len
                batch_feats.append(layer_feat)
                batch_frames += t_len

                if batch_frames >= 100000 or i == len(audio_files) - 1:
                    chunk_arr = np.concatenate(batch_feats, axis=0)
                    chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.npy")
                    np.save(chunk_path, chunk_arr)
                    chunk_files.append(chunk_path)
                    chunk_idx += 1
                    batch_feats = []
                    batch_frames = 0

        logging.info(f"Writing {total_frames} frames (dim={feat_dim}) to {self.out_features_npy.get_path()} via memmap...")
        out_npy = open_memmap(self.out_features_npy.get_path(), mode='w+', dtype='float32', shape=(total_frames, feat_dim))
        offset = 0
        for chunk_path in chunk_files:
            chunk_data = np.load(chunk_path)
            c_len = len(chunk_data)
            out_npy[offset : offset + c_len] = chunk_data
            offset += c_len
            os.remove(chunk_path)
        out_npy.flush()
        del out_npy
        shutil.rmtree(temp_dir, ignore_errors=True)

        with open(self.out_lengths_txt.get_path(), "w") as f:
            for l in lengths:
                f.write(f"{l}\n")

        with open(self.out_seq_tags_txt.get_path(), "w") as f:
            for tag in seq_tags:
                f.write(f"{tag}\n")

        logging.info(f"Feature extraction completed for {len(lengths)} sequences.")


class HuBERTClusterAndCollapseJob(Job):
    """
    Fits K-Means clustering with K centroids on HuBERT features, quantizes frame representations,
    and collapses consecutive duplicate cluster IDs: (c1, c1, c2 -> c1, c2).
    Outputs tokenized audio sequences for Word2Vec training.
    """

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: Optional[tk.Path] = None,
        num_clusters: int = 41,
        sample_pct: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.num_clusters = num_clusters
        self.sample_pct = sample_pct
        self.seed = seed

        # Outputs
        self.out_collapsed_tokens_txt = self.output_path("collapsed_clusters.txt")
        self.out_raw_clusters_txt = self.output_path("raw_clusters.txt")
        self.out_centroids_npy = self.output_path("centroids.npy")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 16, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        from sklearn.cluster import MiniBatchKMeans

        feats = np.load(self.features_npy.get_path(), mmap_mode="r")
        N_total, d = feats.shape
        logging.info(f"Total feature frames: {N_total}, dim: {d}, K = {self.num_clusters}")

        # Sample 10 contiguous chunks of 20,000 frames evenly spaced across dataset (200k frames total)
        num_sample_chunks = 10
        chunk_len = 20000
        step = max(1, (N_total - chunk_len) // num_sample_chunks)
        chunks = []
        for i in range(num_sample_chunks):
            start = i * step
            chunks.append(np.array(feats[start : start + chunk_len], dtype=np.float32))
        sample_feats = np.concatenate(chunks, axis=0)

        logging.info(f"Training MiniBatchKMeans with K={self.num_clusters} on {len(sample_feats)} samples...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.seed,
            batch_size=8192,
            max_iter=100,
            n_init=3,
        )
        kmeans.fit(sample_feats)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        np.save(self.out_centroids_npy.get_path(), centroids)
        centroid_norms = np.sum(centroids ** 2, axis=1)

        # Read sequence lengths
        with open(self.lengths_txt.get_path(), "r") as f:
            lengths = [int(line.strip()) for line in f if line.strip()]

        logging.info(f"Vectorized batch-quantizing {N_total:,} frames...")
        import time
        t_start = time.time()
        chunk_size = 200000
        all_preds = []
        for chunk_idx, start_idx in enumerate(range(0, N_total, chunk_size)):
            end_idx = min(start_idx + chunk_size, N_total)
            chunk_feats = np.array(feats[start_idx:end_idx], dtype=np.float32)
            # argmin ||x - c||^2 = argmax (2 * x @ c.T - ||c||^2)
            scores = 2.0 * np.dot(chunk_feats, centroids.T) - centroid_norms
            preds = np.argmax(scores, axis=1).astype(np.int32)
            all_preds.append(preds)

            if chunk_idx % 20 == 0 or end_idx == N_total:
                pct = (end_idx / N_total) * 100.0
                elapsed = time.time() - t_start
                eta = (elapsed / max(end_idx, 1)) * (N_total - end_idx)
                logging.info(f"Quantization progress: {end_idx:,}/{N_total:,} frames ({pct:.1f}%) [Elapsed: {elapsed:.1f}s, ETA: {eta:.1f}s]")

        all_cluster_ids = np.concatenate(all_preds, axis=0)

        # Run-length deduplication (collapse consecutive identical cluster IDs)
        logging.info(f"Run-length collapsing across {len(lengths):,} utterances...")
        cluster_counts = {k: 0 for k in range(self.num_clusters)}
        offset = 0
        total_utts = len(lengths)

        with open(self.out_collapsed_tokens_txt.get_path(), "w") as f_collapsed, \
             open(self.out_raw_clusters_txt.get_path(), "w") as f_raw:

            for u_idx, seq_len in enumerate(lengths):
                seq_clusters = all_cluster_ids[offset : offset + seq_len].tolist()
                offset += seq_len

                f_raw.write(" ".join(str(c) for c in seq_clusters) + "\n")

                collapsed = [seq_clusters[0]]
                for c in seq_clusters[1:]:
                    if c != collapsed[-1]:
                        collapsed.append(c)

                f_collapsed.write(" ".join(str(c) for c in collapsed) + "\n")
                for c in collapsed:
                    cluster_counts[c] += 1

                if (u_idx + 1) % 5000 == 0 or (u_idx + 1) == total_utts:
                    logging.info(f"Collapse progress: {u_idx + 1:,}/{total_utts:,} utterances ({(u_idx + 1) / total_utts * 100.0:.1f}%)")

        # Write vocab and count files
        with open(self.out_cluster_vocab_txt.get_path(), "w") as f_vocab, \
             open(self.out_cluster_counts_txt.get_path(), "w") as f_counts:
            for k in range(self.num_clusters):
                f_vocab.write(f"{k}\n")
                f_counts.write(f"{k} {cluster_counts[k]}\n")

        logging.info(f"Clustering & collapsing completed for K={self.num_clusters}.")


class HuBERTFramePCAFitJob(Job):
    """
    Fits PCA on frame representations extracted from HuBERT.
    Subsamples frames randomly from the memory-mapped features matrix.
    Outputs:
      - out_components_npy: projection matrix (n_components, feat_dim)
      - out_mean_npy: frame mean vector (feat_dim,)
    """

    def __init__(
        self,
        features_npy: tk.Path,
        n_components: int = 512,
        sample_frames: int = 1_000_000,
        seed: int = 42,
        _version: int = 1,
    ):
        super().__init__()
        self._version = _version
        self.features_npy = features_npy
        self.n_components = n_components
        self.sample_frames = sample_frames
        self.seed = seed

        self.out_components_npy = self.output_path("pca_components.npy")
        self.out_mean_npy = self.output_path("pca_mean.npy")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 32, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        from sklearn.decomposition import PCA

        features = np.load(self.features_npy.get_path(), mmap_mode="r")
        total_frames, feat_dim = features.shape
        logging.info(f"Loaded feature matrix for PCA fitting: shape {features.shape}")

        n_samples = min(total_frames, self.sample_frames)
        rng = np.random.default_rng(self.seed)
        indices = rng.choice(total_frames, size=n_samples, replace=False)
        indices.sort()  # sequential index sorting for optimal mmap page caching

        logging.info(f"Subsampling {n_samples:,} frames for PCA...")
        sample_data = features[indices].astype(np.float32)

        logging.info(f"Fitting PCA ({feat_dim} -> {self.n_components}) with randomized SVD...")
        pca = PCA(n_components=self.n_components, svd_solver="randomized", random_state=self.seed)
        pca.fit(sample_data)

        var_explained = float(np.sum(pca.explained_variance_ratio_))
        logging.info(f"Explained variance ratio sum: {var_explained:.4f}")
        np.save(self.out_components_npy.get_path(), pca.components_.astype(np.float32))
        np.save(self.out_mean_npy.get_path(), pca.mean_.astype(np.float32))
        logging.info("PCA components and mean saved successfully.")


def viterbi_decode_with_penalty(scores: np.ndarray, penalty: float) -> List[int]:
    """
    Decodes the optimal cluster sequence under a soft consecutive repetition penalty:
      min_{c_1..c_T} sum_t d(x_t, mu_{c_t}) + penalty * sum_t I(c_t == c_{t-1})
    Equivalently, maximizes sum_t score(x_t, c_t) - penalty * sum_t I(c_t == c_{t-1}).

    :param scores: [T, K] matrix of negative squared distances / dot product scores (higher is closer).
    :param penalty: Non-negative transition penalty lambda for consecutive duplicate cluster IDs.
    :return: List of decoded cluster IDs of length T.
    """
    T, K = scores.shape
    if T <= 0:
        return []
    if T == 1 or penalty <= 0.0:
        return np.argmax(scores, axis=1).tolist()

    V = np.empty((T, K), dtype=np.float32)
    backtrack = np.empty((T, K), dtype=np.int32)
    V[0] = scores[0]

    for t in range(1, T):
        prev_V = V[t - 1]
        best_j = int(np.argmax(prev_V))
        best_val = prev_V[best_j]

        prev_V_copy = prev_V.copy()
        prev_V_copy[best_j] = -np.inf
        second_best_j = int(np.argmax(prev_V_copy))
        second_best_val = prev_V_copy[second_best_j]

        trans_vals = np.full(K, best_val, dtype=np.float32)
        trans_preds = np.full(K, best_j, dtype=np.int32)

        if penalty >= 1e6 or np.isinf(penalty) or (best_val - penalty < second_best_val):
            trans_vals[best_j] = second_best_val
            trans_preds[best_j] = second_best_j
        else:
            trans_vals[best_j] = best_val - penalty
            trans_preds[best_j] = best_j

        V[t] = scores[t] + trans_vals
        backtrack[t] = trans_preds

    best_last = int(np.argmax(V[-1]))
    path = [best_last]
    for t in range(T - 1, 0, -1):
        best_last = int(backtrack[t, best_last])
        path.append(best_last)

    path.reverse()
    return path


def align_words_to_segments(words: List[str], ref_phones: List[str], wrd_to_phn: Dict[str, List[str]]) -> Set[int]:
    """
    Finds segment indices where words end by dynamic programming alignment
    between the word phonemes and the oracle segment phonemes.
    """
    import numpy as np
    word_phonemes = []
    phone_to_word = []
    for w_idx, w in enumerate(words):
        phns = wrd_to_phn.get(w.lower(), [])
        for p in phns:
            word_phonemes.append(p)
            phone_to_word.append(w_idx)
    N = len(word_phonemes)
    M = len(ref_phones)
    if N == 0 or M == 0:
        return set()
    dp = np.zeros((N + 1, M + 1), dtype=int)
    dp[:, 0] = np.arange(N + 1)
    dp[0, :] = np.arange(M + 1)
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            cost = 0 if word_phonemes[i - 1] == ref_phones[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    i, j = N, M
    aligned = [0] * M
    while i > 0 and j > 0:
        cost = 0 if word_phonemes[i - 1] == ref_phones[j - 1] else 1
        if dp[i, j] == dp[i - 1, j - 1] + cost:
            aligned[j - 1] = phone_to_word[i - 1]
            i -= 1
            j -= 1
        elif dp[i, j] == dp[i, j - 1] + 1:
            aligned[j - 1] = phone_to_word[min(i, N - 1)]
            j -= 1
        else:
            i -= 1
    wb = set()
    for k in range(len(aligned) - 1):
        if aligned[k] != aligned[k + 1]:
            wb.add(k)
    return wb


class HuBERTCheatingSegmentAveragingAndClusterJob(Job):
    """
    Ablation: Uses GMM oracle ("cheating") segmentation boundaries to pool HuBERT frame features.
    For each discovered non-silence phoneme segment, averages (or weighted-averages) the frame-level
    representations across the segment.
    Then trains K-Means with K centroids on the pooled segment representations and quantizes each segment.
    Outputs the sequence of segment cluster indices as pseudo-tokens for Word2Vec training.
    """

    __sis_hash_exclude__ = {
        "pooling_method": "mean",
        "clustering_method": "kmeans",
        "spectral_n_neighbors": 30,
        "spectral_sample_size": 25000,
    }

    @staticmethod
    def _compute_pooling_weights(L: int, method: str = "mean") -> np.ndarray:
        """
        Computes 1D pooling weights of length L normalized so sum(weights) == 1.
        Methods:
          - "mean": uniform weights (1/L).
          - "center_weighted" / "weighted" / "triangular": symmetric triangular window with positive boundary weights.
          - "gaussian": gaussian window centered at (L-1)/2 with sigma = max(1.0, (L-1)/3.0).
          - "hann": Hann / cosine window with pedestal.
        """
        if L <= 2 or method in (None, "mean"):
            return np.full((L,), 1.0 / max(L, 1), dtype=np.float32)

        indices = np.arange(L, dtype=np.float32)
        center = (L - 1.0) / 2.0

        if method in ("center_weighted", "weighted", "weighted_average", "triangular"):
            half_width = (L + 1.0) / 2.0
            weights = 1.0 - np.abs(indices - center) / half_width
        elif method == "gaussian":
            sigma = max(1.0, (L - 1.0) / 3.0)
            weights = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
        elif method == "hann":
            weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * (indices + 0.5) / L))
        else:
            raise ValueError(f"Unknown pooling_method: {method}")

        weights = weights / np.sum(weights)
        return weights.astype(np.float32)

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: tk.Path,
        segmentation_hdfs: List[tk.Path],
        num_clusters: int = 483,
        pca_components_npy: Optional[tk.Path] = None,
        pca_mean_npy: Optional[tk.Path] = None,
        normalize_segments: Optional[str] = None,
        sample_pct: float = 0.2,
        seed: int = 42,
        collapse_consecutive: bool = True,
        repetition_penalty: float = 0.0,
        word_boundary_token: Optional[str] = None,
        transcript_text_file: Optional[Union[DelayedBase, tk.Path]] = None,
        lexicon_file: Optional[Union[DelayedBase, tk.Path]] = None,
        idx_to_phone_file: Optional[Union[DelayedBase, tk.Path]] = None,
        pooling_method: str = "mean",
        clustering_method: str = "kmeans",
        spectral_n_neighbors: int = 30,
        spectral_sample_size: int = 25000,
        _version: int = 2,
    ):
        super().__init__()
        ver = _version
        if pca_components_npy is not None:
            ver = max(ver, 3)
        if normalize_segments is not None:
            ver = max(ver, 4)
        if repetition_penalty > 0.0:
            ver = max(ver, 5)
        if word_boundary_token is not None:
            ver = max(ver, 6)
        if pooling_method not in (None, "mean"):
            ver = max(ver, 7)
        if clustering_method not in (None, "kmeans"):
            ver = max(ver, 8)
        self._version = ver
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.segmentation_hdfs = segmentation_hdfs
        self.num_clusters = num_clusters
        self.pca_components_npy = pca_components_npy
        self.pca_mean_npy = pca_mean_npy
        self.normalize_segments = normalize_segments
        self.sample_pct = sample_pct
        self.seed = seed
        self.collapse_consecutive = collapse_consecutive
        self.repetition_penalty = float(repetition_penalty)
        self.word_boundary_token = word_boundary_token
        self.transcript_text_file = transcript_text_file
        self.lexicon_file = lexicon_file
        self.idx_to_phone_file = idx_to_phone_file
        self.pooling_method = pooling_method or "mean"
        self.clustering_method = clustering_method or "kmeans"
        self.spectral_n_neighbors = int(spectral_n_neighbors)
        self.spectral_sample_size = int(spectral_sample_size)

        # Outputs (matches HuBERTClusterAndCollapseJob interface for downstream compatibility)
        self.out_collapsed_tokens_txt = self.output_path("collapsed_clusters.txt")
        self.out_raw_clusters_txt = self.output_path("raw_clusters.txt")
        self.out_centroids_npy = self.output_path("centroids.npy")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 64, "time": 6})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        from collections import defaultdict
        import h5py
        from sklearn.cluster import MiniBatchKMeans

        # 1. Load sequence tags and utterance offsets in features_npy
        with open(self.seq_tags_txt.get_path(), "r") as f:
            seq_tags = [line.strip().split()[0] for line in f if line.strip()]

        with open(self.lengths_txt.get_path(), "r") as f:
            lengths = [int(line.strip()) for line in f if line.strip()]

        assert len(seq_tags) == len(lengths), f"Mismatch between tags ({len(seq_tags)}) and lengths ({len(lengths)})"
        logging.info(f"Loaded {len(seq_tags):,} sequences from manifest.")

        tag_to_indices = defaultdict(list)
        utt_offsets = []
        offset = 0
        for idx, (tag, l) in enumerate(zip(seq_tags, lengths)):
            clean_tag = tag.split("/")[-1]
            tag_to_indices[tag].append(idx)
            tag_to_indices[clean_tag].append(idx)
            utt_offsets.append(offset)
            offset += l

        total_frames = offset
        logging.info(f"Total dataset frames: {total_frames:,}")

        # Open features_npy with memory mapping
        features = np.load(self.features_npy.get_path(), mmap_mode="r")
        feat_dim = features.shape[1]
        logging.info(f"Loaded feature matrix: shape {features.shape}")

        # 2. Extract segments from GMM phoneme reference HDFs
        logging.info(f"Scanning {len(self.segmentation_hdfs)} segmentation HDF shards for {len(seq_tags):,} utterance instances...")

        utt_segments = [[] for _ in range(len(seq_tags))]
        utt_seg_labels = [[] for _ in range(len(seq_tags))]
        matched_tags_count = 0

        for hdf_path in self.segmentation_hdfs:
            resolved_p = hdf_path.get_path() if hasattr(hdf_path, "get_path") else str(hdf_path)
            if resolved_p.startswith("/hpcwork/"):
                resolved_p = "/rwthfs/rz/cluster/hpcwork/" + resolved_p[len("/hpcwork/"):]
            if not os.path.exists(resolved_p):
                logging.warning(f"Segmentation shard not found: {resolved_p}")
                continue

            with h5py.File(resolved_p, "r") as f:
                if "seqTags" not in f or "seqLengths" not in f:
                    continue
                shard_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                clean_shard_tags = [t.split("/")[-1] for t in shard_tags]
                shard_matching_indices = [
                    i for i, (t, ct) in enumerate(zip(shard_tags, clean_shard_tags))
                    if t in tag_to_indices or ct in tag_to_indices
                ]
                if not shard_matching_indices:
                    continue

                shard_lens = f["seqLengths"][:]
                target_key = "inputs" if "inputs" in f else ("data" if "data" in f else None)
                target_data = f[target_key]

                if shard_lens.ndim == 2:
                    lengths_1d = shard_lens[:, 0].astype(np.int64)
                else:
                    lengths_1d = shard_lens[:].astype(np.int64)
                shard_offsets = np.zeros(len(lengths_1d), dtype=np.int64)
                np.cumsum(lengths_1d[:-1], out=shard_offsets[1:])

                for i in shard_matching_indices:
                    t = shard_tags[i]
                    ct = clean_shard_tags[i]
                    target_indices = tag_to_indices[t] if t in tag_to_indices else tag_to_indices[ct]

                    s_off = shard_offsets[i]
                    s_len = lengths_1d[i]
                    labels = target_data[s_off : s_off + s_len]

                    # Group consecutive identical non-silence labels (label != 0) into segments
                    cur_label = None
                    cur_start = None
                    segs = []
                    seg_labels = []
                    for t_idx, lab in enumerate(labels):
                        lab_val = int(lab)
                        if lab_val != cur_label:
                            if cur_label is not None and cur_label != 0:
                                segs.append((cur_start, t_idx - 1))
                                seg_labels.append(cur_label)
                            cur_label = lab_val
                            cur_start = t_idx
                    if cur_label is not None and cur_label != 0:
                        segs.append((cur_start, len(labels) - 1))
                        seg_labels.append(cur_label)

                    for u_idx in target_indices:
                        u_len = lengths[u_idx]
                        if not segs:
                            utt_segments[u_idx] = [(0, min(u_len - 1, len(labels) - 1))]
                            utt_seg_labels[u_idx] = [0]
                        else:
                            utt_segments[u_idx] = segs
                            utt_seg_labels[u_idx] = seg_labels
                        matched_tags_count += 1

        logging.info(f"Matched {matched_tags_count:,}/{len(seq_tags):,} utterance instances with oracle segmentation.")

        # Optional: Compute word boundaries for audio segments
        word_boundaries_by_utt = {}
        if self.word_boundary_token is not None:
            logging.info("Computing word boundary segment alignments for audio utterances...")
            idx_to_phone = {}
            idx_phone_p = (
                self.idx_to_phone_file.get_path()
                if self.idx_to_phone_file is not None and hasattr(self.idx_to_phone_file, "get_path")
                else (self.idx_to_phone_file if self.idx_to_phone_file is not None else "/u/zyang/setups/mini/output/lbs_mono_phone_eow/idx_to_phoneme.txt")
            )
            if os.path.exists(idx_phone_p):
                with open(idx_phone_p, "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            idx_to_phone[int(parts[0])] = parts[1]

            wrd_to_phn = {}
            if self.lexicon_file is not None:
                lex_p = self.lexicon_file.get_path() if hasattr(self.lexicon_file, "get_path") else str(self.lexicon_file)
                if os.path.exists(lex_p):
                    with open(lex_p, "r", encoding="utf-8") as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                wrd = parts[0].lower()
                                phones = ["".join(c for c in p if not c.isdigit()) for p in parts[1:]]
                                if wrd not in wrd_to_phn:
                                    wrd_to_phn[wrd] = phones

            text_by_tag = {}
            if self.transcript_text_file is not None:
                text_p = self.transcript_text_file.get_path() if hasattr(self.transcript_text_file, "get_path") else str(self.transcript_text_file)
                if os.path.exists(text_p):
                    with open(self.seq_tags_txt.get_path(), "r") as f_tags, open(text_p, "r", encoding="utf-8") as f_text:
                        for t_line, text_line in zip(f_tags, f_text):
                            t_clean = t_line.strip().split()[0].split("/")[-1]
                            text_by_tag[t_clean] = text_line.strip()

            for u_idx in range(len(seq_tags)):
                t_clean = seq_tags[u_idx].split("/")[-1]
                words = text_by_tag.get(t_clean, "").split()
                ref_pids = utt_seg_labels[u_idx]
                ref_phones = [idx_to_phone.get(pid, str(pid)) for pid in ref_pids]
                wb = align_words_to_segments(words, ref_phones, wrd_to_phn)
                if not wb and len(utt_segments[u_idx]) > 1:
                    u_segs = utt_segments[u_idx]
                    for s_i in range(len(u_segs) - 1):
                        if u_segs[s_i + 1][0] > u_segs[s_i][1] + 1:
                            wb.add(s_i)
                word_boundaries_by_utt[u_idx] = wb
            logging.info(f"Computed word boundaries across {len(word_boundaries_by_utt):,} utterances.")

        # 3. Compute Segment Average / Weighted-Average Representations
        p_method = (self.pooling_method or "mean").strip().lower()
        use_weighted_pooling = p_method not in ("mean", "uniform")
        weights_cache: Dict[int, np.ndarray] = {}
        logging.info(f"Pooling oracle segments with pooling_method='{p_method}'...")

        all_segment_reps = []
        utt_segment_counts = []
        total_segments = 0

        for idx in range(len(seq_tags)):
            utt_offset = utt_offsets[idx]
            utt_len = lengths[idx]
            segs = utt_segments[idx]
            if not segs:
                segs = [(0, max(0, utt_len - 1))]
                utt_segments[idx] = segs

            utt_segment_counts.append(len(segs))
            for s, e in segs:
                s_clamped = max(0, min(s, utt_len - 1))
                e_clamped = max(s_clamped, min(e, utt_len - 1))
                seg_feats = features[utt_offset + s_clamped : utt_offset + e_clamped + 1]
                L = e_clamped - s_clamped + 1
                if not use_weighted_pooling or L <= 2:
                    pooled_vec = np.mean(seg_feats, axis=0, dtype=np.float32)
                else:
                    w_vec = weights_cache.get(L)
                    if w_vec is None:
                        w_vec = self._compute_pooling_weights(L, p_method)
                        weights_cache[L] = w_vec
                    pooled_vec = np.dot(w_vec, seg_feats).astype(np.float32)
                all_segment_reps.append(pooled_vec)
                total_segments += 1

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(seq_tags):
                logging.info(
                    f"Segment pooling ({p_method}) progress: {idx + 1:,}/{len(seq_tags):,} utts ({total_segments:,} segments pooled)"
                )

        all_segment_reps = np.stack(all_segment_reps, axis=0)
        logging.info(f"Total pooled segment representations: {all_segment_reps.shape}")

        # 3b. Apply Frame-Fitted PCA Projection (if configured)
        if self.pca_components_npy is not None and self.pca_mean_npy is not None:
            logging.info("Applying frame-fitted PCA projection to segment representations...")
            pca_comp = np.load(self.pca_components_npy.get_path()).astype(np.float32)
            pca_mean = np.load(self.pca_mean_npy.get_path()).astype(np.float32)
            # Linearity: Average( (x - μ) @ W ) == (Average(x) - μ) @ W
            # Project in chunks to avoid allocating ~60GB of temporary arrays
            n_segs = len(all_segment_reps)
            out_dim = pca_comp.shape[0]
            projected = np.empty((n_segs, out_dim), dtype=np.float32)
            chunk_size = 500000
            for c_start in range(0, n_segs, chunk_size):
                c_end = min(c_start + chunk_size, n_segs)
                chunk = all_segment_reps[c_start:c_end] - pca_mean
                projected[c_start:c_end] = np.dot(chunk, pca_comp.T)
            del all_segment_reps
            all_segment_reps = projected
            logging.info(f"Projected segment representations to shape: {all_segment_reps.shape}")

        # 3c. Optional Normalization to Unit Sphere (v5.4 setup)
        if self.normalize_segments:
            norm_mode = str(self.normalize_segments).lower().strip()
            if norm_mode in ("unit_sphere", "l2", "unit", "sphere", "true"):
                logging.info("Normalizing segment representations to the unit sphere (L2 norm = 1.0)...")
                norms = np.linalg.norm(all_segment_reps, axis=1, keepdims=True)
                all_segment_reps /= np.maximum(norms, 1e-12)
            elif norm_mode in ("z_score", "z_transform", "standardize"):
                logging.info("Applying z-score standardization (zero mean, unit variance) across segment representations...")
                mean = np.mean(all_segment_reps, axis=0, keepdims=True)
                std = np.std(all_segment_reps, axis=0, keepdims=True)
                all_segment_reps -= mean
                all_segment_reps /= np.maximum(std, 1e-12)
            elif norm_mode in ("center_unit", "unit_center_unit"):
                logging.info("Applying unit-norm + mean-centering + unit-norm to segment representations...")
                norms = np.linalg.norm(all_segment_reps, axis=1, keepdims=True)
                all_segment_reps /= np.maximum(norms, 1e-12)
                mean = np.mean(all_segment_reps, axis=0, keepdims=True)
                all_segment_reps -= mean
                norms = np.linalg.norm(all_segment_reps, axis=1, keepdims=True)
                all_segment_reps /= np.maximum(norms, 1e-12)
            else:
                raise ValueError(f"Unknown normalize_segments mode: {self.normalize_segments}")

        # 4. Clustering on Segment Representations (K-Means or Spectral Clustering)
        c_method = (self.clustering_method or "kmeans").strip().lower()
        rng = np.random.default_rng(self.seed)

        if c_method in ("spectral", "spectral_clustering"):
            from sklearn.cluster import SpectralClustering

            n_spec_sample = min(len(all_segment_reps), max(self.num_clusters * 10, self.spectral_sample_size))
            sample_indices = rng.choice(len(all_segment_reps), size=n_spec_sample, replace=False)
            sample_reps = all_segment_reps[sample_indices]

            logging.info(
                f"Training SpectralClustering (K={self.num_clusters}, affinity='nearest_neighbors', "
                f"n_neighbors={self.spectral_n_neighbors}) on {n_spec_sample:,} segment representations..."
            )
            spectral = SpectralClustering(
                n_clusters=self.num_clusters,
                affinity="nearest_neighbors",
                n_neighbors=min(self.spectral_n_neighbors, n_spec_sample - 1),
                assign_labels="kmeans",
                random_state=self.seed,
                n_init=10,
                n_jobs=8,
            )
            sample_labels = spectral.fit_predict(sample_reps)

            # Compute centroids in the segment representation space from the spectral partition
            feat_dim_proj = all_segment_reps.shape[1]
            centroids = np.zeros((self.num_clusters, feat_dim_proj), dtype=np.float32)
            global_mean = np.mean(sample_reps, axis=0, dtype=np.float32)
            for k in range(self.num_clusters):
                mask_k = sample_labels == k
                if np.any(mask_k):
                    c_vec = np.mean(sample_reps[mask_k], axis=0, dtype=np.float32)
                    if self.normalize_segments and str(self.normalize_segments).lower().strip() in (
                        "unit_sphere",
                        "l2",
                        "unit",
                        "sphere",
                        "true",
                        "center_unit",
                        "unit_center_unit",
                    ):
                        c_norm = float(np.linalg.norm(c_vec))
                        if c_norm > 1e-12:
                            c_vec = c_vec / c_norm
                    centroids[k] = c_vec
                else:
                    centroids[k] = global_mean
            np.save(self.out_centroids_npy.get_path(), centroids)
            centroid_norms = np.sum(centroids ** 2, axis=1)
        else:
            n_sample = min(len(all_segment_reps), max(200000, int(len(all_segment_reps) * self.sample_pct)))
            sample_indices = rng.choice(len(all_segment_reps), size=n_sample, replace=False)
            sample_reps = all_segment_reps[sample_indices]

            logging.info(f"Training MiniBatchKMeans (K={self.num_clusters}) on {n_sample:,} segment representations...")
            kmeans = MiniBatchKMeans(
                n_clusters=self.num_clusters,
                random_state=self.seed,
                batch_size=8192,
                max_iter=100,
                n_init=3,
            )
            kmeans.fit(sample_reps)
            centroids = kmeans.cluster_centers_.astype(np.float32)
            np.save(self.out_centroids_npy.get_path(), centroids)
            centroid_norms = np.sum(centroids ** 2, axis=1)

        # 5. Quantize all segment representations to nearest centroid (or via Viterbi with soft repetition penalty)
        logging.info(
            f"Quantizing {len(all_segment_reps):,} segments to K={self.num_clusters} centroids "
            f"(repetition_penalty={self.repetition_penalty})..."
        )
        chunk_size = 100000
        quantized_clusters = []
        all_segment_scores = []
        for c_start in range(0, len(all_segment_reps), chunk_size):
            c_end = min(c_start + chunk_size, len(all_segment_reps))
            sub = all_segment_reps[c_start:c_end]
            scores = 2.0 * np.dot(sub, centroids.T) - centroid_norms
            if self.repetition_penalty > 0.0:
                all_segment_scores.append(scores)
            else:
                preds = np.argmax(scores, axis=1).astype(np.int32)
                quantized_clusters.append(preds)

        if self.repetition_penalty > 0.0:
            all_scores_mat = np.concatenate(all_segment_scores, axis=0)
            all_cluster_ids = None
        else:
            all_scores_mat = None
            all_cluster_ids = np.concatenate(quantized_clusters, axis=0)

        # 6. Reconstruct sequence tokens per utterance and compute counts
        cluster_counts = {k: 0 for k in range(self.num_clusters)}
        seg_ptr = 0

        with open(self.out_collapsed_tokens_txt.get_path(), "w") as f_tokens, \
             open(self.out_raw_clusters_txt.get_path(), "w") as f_raw:
            for u_idx, n_segs in enumerate(utt_segment_counts):
                if self.repetition_penalty > 0.0 and n_segs > 0:
                    u_scores = all_scores_mat[seg_ptr : seg_ptr + n_segs]
                    u_cids = viterbi_decode_with_penalty(u_scores, self.repetition_penalty)
                else:
                    u_cids = all_cluster_ids[seg_ptr : seg_ptr + n_segs].tolist() if all_cluster_ids is not None else []
                seg_ptr += n_segs

                if self.collapse_consecutive and u_cids:
                    collapsed = [u_cids[0]]
                    for c in u_cids[1:]:
                        if c != collapsed[-1]:
                            collapsed.append(c)
                else:
                    collapsed = u_cids

                if self.word_boundary_token is not None and u_idx in word_boundaries_by_utt:
                    wb_indices = word_boundaries_by_utt[u_idx]
                    raw_with_wb = []
                    for s_i, c in enumerate(u_cids):
                        raw_with_wb.append(str(c))
                        if s_i in wb_indices and s_i < len(u_cids) - 1:
                            raw_with_wb.append(self.word_boundary_token)
                    f_raw.write(" ".join(raw_with_wb) + "\n")

                    coll_with_wb = []
                    for s_i, c in enumerate(collapsed):
                        coll_with_wb.append(str(c))
                        if s_i in wb_indices and s_i < len(collapsed) - 1:
                            coll_with_wb.append(self.word_boundary_token)
                    f_tokens.write(" ".join(coll_with_wb) + "\n")
                else:
                    f_raw.write(" ".join(str(c) for c in u_cids) + "\n")
                    f_tokens.write(" ".join(str(c) for c in collapsed) + "\n")

                for c in collapsed:
                    cluster_counts[c] += 1

        # 7. Write vocabulary and counts files
        with open(self.out_cluster_vocab_txt.get_path(), "w") as f_vocab, \
             open(self.out_cluster_counts_txt.get_path(), "w") as f_counts:
            for k in range(self.num_clusters):
                f_vocab.write(f"{k}\n")
                f_counts.write(f"{k} {cluster_counts[k]}\n")

        logging.info(
            f"Cheating segmentation clustering completed successfully. "
            f"Total segments: {total_segments:,}, Unique clusters used: {sum(1 for c in cluster_counts.values() if c > 0)}/{self.num_clusters}"
        )


class HuBERTCheatingSegmentCOPClusterJob(Job):
    """
    Constrained COP-KMeans: Alternating Viterbi decoding with soft repetition penalty
    and centroid re-estimation on pooled GMM oracle phoneme segments.
    Iteratively updates cluster centroids so that segments that occur consecutively
    are pushed into distinct acoustic clusters.
    """
    __sis_version__ = 1

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: tk.Path,
        segmentation_hdfs: List[tk.Path],
        num_clusters: int = 40,
        pca_components_npy: Optional[tk.Path] = None,
        pca_mean_npy: Optional[tk.Path] = None,
        normalize_segments: Optional[str] = "unit_sphere",
        sample_pct: float = 0.2,
        seed: int = 42,
        collapse_consecutive: bool = False,
        repetition_penalty: float = 0.25,
        max_iter: int = 15,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.segmentation_hdfs = segmentation_hdfs
        self.num_clusters = num_clusters
        self.pca_components_npy = pca_components_npy
        self.pca_mean_npy = pca_mean_npy
        self.normalize_segments = normalize_segments
        self.sample_pct = sample_pct
        self.seed = seed
        self.collapse_consecutive = collapse_consecutive
        self.repetition_penalty = float(repetition_penalty)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        # Outputs matching HuBERTCheatingSegmentAveragingAndClusterJob interface
        self.out_collapsed_tokens_txt = self.output_path("collapsed_clusters.txt")
        self.out_raw_clusters_txt = self.output_path("raw_clusters.txt")
        self.out_centroids_npy = self.output_path("centroids.npy")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 64, "time": 6})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        from collections import defaultdict
        import h5py
        from sklearn.cluster import MiniBatchKMeans

        # 1. Load sequence tags and utterance offsets in features_npy
        with open(self.seq_tags_txt.get_path(), "r") as f:
            seq_tags = [line.strip().split()[0] for line in f if line.strip()]

        with open(self.lengths_txt.get_path(), "r") as f:
            lengths = [int(line.strip()) for line in f if line.strip()]

        assert len(seq_tags) == len(lengths), f"Mismatch between tags ({len(seq_tags)}) and lengths ({len(lengths)})"
        logging.info(f"Loaded {len(seq_tags):,} sequences from manifest.")

        tag_to_indices = defaultdict(list)
        utt_offsets = []
        offset = 0
        for idx, (tag, l) in enumerate(zip(seq_tags, lengths)):
            clean_tag = tag.split("/")[-1]
            tag_to_indices[tag].append(idx)
            tag_to_indices[clean_tag].append(idx)
            utt_offsets.append(offset)
            offset += l

        total_frames = offset
        logging.info(f"Total dataset frames: {total_frames:,}")

        features = np.load(self.features_npy.get_path(), mmap_mode="r")
        logging.info(f"Loaded feature matrix: shape {features.shape}")

        # 2. Extract segments from GMM phoneme reference HDFs
        logging.info(f"Scanning {len(self.segmentation_hdfs)} segmentation HDF shards for {len(seq_tags):,} utterance instances...")
        utt_segments = [[] for _ in range(len(seq_tags))]
        matched_tags_count = 0

        for hdf_path in self.segmentation_hdfs:
            resolved_p = hdf_path.get_path() if hasattr(hdf_path, "get_path") else str(hdf_path)
            if resolved_p.startswith("/hpcwork/"):
                resolved_p = "/rwthfs/rz/cluster/hpcwork/" + resolved_p[len("/hpcwork/"):]
            if not os.path.exists(resolved_p):
                logging.warning(f"Segmentation shard not found: {resolved_p}")
                continue

            with h5py.File(resolved_p, "r") as f:
                if "seqTags" not in f or "seqLengths" not in f:
                    continue
                shard_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                clean_shard_tags = [t.split("/")[-1] for t in shard_tags]
                shard_matching_indices = [
                    i for i, (t, ct) in enumerate(zip(shard_tags, clean_shard_tags))
                    if t in tag_to_indices or ct in tag_to_indices
                ]
                if not shard_matching_indices:
                    continue

                shard_lens = f["seqLengths"][:]
                target_key = "inputs" if "inputs" in f else ("data" if "data" in f else None)
                target_data = f[target_key]

                if shard_lens.ndim == 2:
                    lengths_1d = shard_lens[:, 0].astype(np.int64)
                else:
                    lengths_1d = shard_lens[:].astype(np.int64)
                shard_offsets = np.zeros(len(lengths_1d), dtype=np.int64)
                np.cumsum(lengths_1d[:-1], out=shard_offsets[1:])

                for i in shard_matching_indices:
                    t = shard_tags[i]
                    ct = clean_shard_tags[i]
                    target_indices = tag_to_indices[t] if t in tag_to_indices else tag_to_indices[ct]
                    s_off = shard_offsets[i]
                    s_len = lengths_1d[i]
                    labels = target_data[s_off : s_off + s_len]

                    cur_label = None
                    cur_start = None
                    segs = []
                    for t_idx, lab in enumerate(labels):
                        lab_val = int(lab)
                        if lab_val != cur_label:
                            if cur_label is not None and cur_label != 0:
                                segs.append((cur_start, t_idx - 1))
                            cur_label = lab_val
                            cur_start = t_idx
                    if cur_label is not None and cur_label != 0:
                        segs.append((cur_start, len(labels) - 1))

                    for u_idx in target_indices:
                        u_len = lengths[u_idx]
                        if not segs:
                            utt_segments[u_idx] = [(0, min(u_len - 1, len(labels) - 1))]
                        else:
                            utt_segments[u_idx] = segs
                        matched_tags_count += 1

        logging.info(f"Matched {matched_tags_count:,}/{len(seq_tags):,} utterance instances with oracle segmentation.")

        # 3. Compute Segment Average Representations
        all_segment_reps = []
        utt_segment_counts = []
        total_segments = 0

        for idx in range(len(seq_tags)):
            utt_offset = utt_offsets[idx]
            utt_len = lengths[idx]
            segs = utt_segments[idx]
            if not segs:
                segs = [(0, max(0, utt_len - 1))]
                utt_segments[idx] = segs

            utt_segment_counts.append(len(segs))
            for s, e in segs:
                s_clamped = max(0, min(s, utt_len - 1))
                e_clamped = max(s_clamped, min(e, utt_len - 1))
                seg_feats = features[utt_offset + s_clamped : utt_offset + e_clamped + 1]
                mean_vec = np.mean(seg_feats, axis=0, dtype=np.float32)
                all_segment_reps.append(mean_vec)
                total_segments += 1

        all_segment_reps = np.stack(all_segment_reps, axis=0)
        logging.info(f"Total pooled segment representations: {all_segment_reps.shape}")

        # 3b. Apply Frame-Fitted PCA Projection
        if self.pca_components_npy is not None and self.pca_mean_npy is not None:
            logging.info("Applying frame-fitted PCA projection to segment representations...")
            pca_comp = np.load(self.pca_components_npy.get_path()).astype(np.float32)
            pca_mean = np.load(self.pca_mean_npy.get_path()).astype(np.float32)
            n_segs = len(all_segment_reps)
            out_dim = pca_comp.shape[0]
            projected = np.empty((n_segs, out_dim), dtype=np.float32)
            chunk_size = 500000
            for c_start in range(0, n_segs, chunk_size):
                c_end = min(c_start + chunk_size, n_segs)
                chunk = all_segment_reps[c_start:c_end] - pca_mean
                projected[c_start:c_end] = np.dot(chunk, pca_comp.T)
            del all_segment_reps
            all_segment_reps = projected

        # 3c. Normalization to Unit Sphere
        if self.normalize_segments:
            norm_mode = str(self.normalize_segments).lower().strip()
            if norm_mode in ("unit_sphere", "l2", "unit", "sphere", "true"):
                logging.info("Normalizing segment representations to the unit sphere (L2 norm = 1.0)...")
                norms = np.linalg.norm(all_segment_reps, axis=1, keepdims=True)
                all_segment_reps /= np.maximum(norms, 1e-12)
            elif norm_mode in ("z_score", "z_transform", "standardize"):
                mean = np.mean(all_segment_reps, axis=0, keepdims=True)
                std = np.std(all_segment_reps, axis=0, keepdims=True)
                all_segment_reps -= mean
                all_segment_reps /= np.maximum(std, 1e-12)

        # 4. Initial K-Means Centroids
        n_sample = min(len(all_segment_reps), max(200000, int(len(all_segment_reps) * self.sample_pct)))
        rng = np.random.default_rng(self.seed)
        sample_indices = rng.choice(len(all_segment_reps), size=n_sample, replace=False)
        sample_reps = all_segment_reps[sample_indices]

        logging.info(f"Initializing COP-KMeans with MiniBatchKMeans (K={self.num_clusters}) on {n_sample:,} segments...")
        kmeans = MiniBatchKMeans(
            n_clusters=self.num_clusters,
            random_state=self.seed,
            batch_size=8192,
            max_iter=100,
            n_init=3,
        )
        kmeans.fit(sample_reps)
        centroids = kmeans.cluster_centers_.astype(np.float32)
        if self.normalize_segments:
            centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)

        # 5. Alternating Constrained COP-KMeans: Viterbi Assignment + Centroid Re-Estimation
        logging.info(
            f"Starting alternating COP-KMeans for max {self.max_iter} iterations "
            f"(K={self.num_clusters}, repetition_penalty={self.repetition_penalty})..."
        )
        all_cluster_ids_arr = None

        for it in range(1, self.max_iter + 1):
            centroid_norms = np.sum(centroids ** 2, axis=1)

            # E-step: Score and Viterbi decode
            all_segment_scores = []
            chunk_size = 100000
            for c_start in range(0, len(all_segment_reps), chunk_size):
                c_end = min(c_start + chunk_size, len(all_segment_reps))
                sub = all_segment_reps[c_start:c_end]
                scores = 2.0 * np.dot(sub, centroids.T) - centroid_norms
                all_segment_scores.append(scores)
            all_scores_mat = np.concatenate(all_segment_scores, axis=0)

            all_cluster_ids = []
            seg_ptr = 0
            total_reps = 0
            total_trans = 0
            for n_segs in utt_segment_counts:
                if n_segs > 0:
                    u_scores = all_scores_mat[seg_ptr : seg_ptr + n_segs]
                    u_cids = viterbi_decode_with_penalty(u_scores, self.repetition_penalty)
                    for i in range(1, len(u_cids)):
                        total_trans += 1
                        if u_cids[i] == u_cids[i - 1]:
                            total_reps += 1
                else:
                    u_cids = []
                all_cluster_ids.extend(u_cids)
                seg_ptr += n_segs

            all_cluster_ids_arr = np.array(all_cluster_ids, dtype=np.int32)
            rep_rate = (total_reps / total_trans * 100.0) if total_trans > 0 else 0.0

            # M-step: Centroid Re-Estimation
            new_centroids = np.empty_like(centroids)
            for k in range(self.num_clusters):
                mask = (all_cluster_ids_arr == k)
                k_count = np.sum(mask)
                if k_count > 0:
                    new_centroids[k] = np.mean(all_segment_reps[mask], axis=0)
                    if self.normalize_segments:
                        norm_c = np.linalg.norm(new_centroids[k])
                        if norm_c > 1e-12:
                            new_centroids[k] /= norm_c
                else:
                    new_centroids[k] = centroids[k]

            shift = float(np.max(np.linalg.norm(new_centroids - centroids, axis=1)))
            logging.info(
                f"COP-KMeans iter {it}/{self.max_iter}: max centroid shift = {shift:.6f}, "
                f"repetition rate = {rep_rate:.2f}% ({total_reps:,}/{total_trans:,})"
            )
            centroids = new_centroids
            if shift < self.tol:
                logging.info(f"COP-KMeans converged at iteration {it} (shift < {self.tol}).")
                break

        np.save(self.out_centroids_npy.get_path(), centroids)

        # 6. Reconstruct sequence tokens per utterance and compute counts
        cluster_counts = {k: 0 for k in range(self.num_clusters)}
        seg_ptr = 0

        with open(self.out_collapsed_tokens_txt.get_path(), "w") as f_tokens, \
             open(self.out_raw_clusters_txt.get_path(), "w") as f_raw:
            for n_segs in utt_segment_counts:
                u_cids = all_cluster_ids_arr[seg_ptr : seg_ptr + n_segs].tolist() if n_segs > 0 else []
                seg_ptr += n_segs

                f_raw.write(" ".join(str(c) for c in u_cids) + "\n")

                if self.collapse_consecutive and u_cids:
                    collapsed = [u_cids[0]]
                    for c in u_cids[1:]:
                        if c != collapsed[-1]:
                            collapsed.append(c)
                else:
                    collapsed = u_cids

                f_tokens.write(" ".join(str(c) for c in collapsed) + "\n")
                for c in collapsed:
                    cluster_counts[c] += 1

        # 7. Write vocabulary and counts files
        with open(self.out_cluster_vocab_txt.get_path(), "w") as f_vocab, \
             open(self.out_cluster_counts_txt.get_path(), "w") as f_counts:
            for k in range(self.num_clusters):
                f_vocab.write(f"{k}\n")
                f_counts.write(f"{k} {cluster_counts[k]}\n")

        logging.info(
            f"COP-KMeans completed successfully. Total segments: {total_segments:,}, "
            f"Unique clusters used: {sum(1 for c in cluster_counts.values() if c > 0)}/{self.num_clusters}"
        )


class HuBERTCheatingSegmentPCKClusterJob(HuBERTCheatingSegmentCOPClusterJob):
    """
    Pairwise Constrained K-Means (PCK-Means, Basu, Banerjee, Mooney, SDM 2004):
    Alternating cluster assignment with soft cannot-link penalties and centroid re-estimation
    over pooled GMM oracle phoneme segments.
    Minimizes the penalized objective:
      J_PCKMeans = sum_t 1/2 ||x_t - mu_{c_t}||^2 + lambda * sum_t I(c_t == c_{t-1})
    where lambda (repetition_penalty) soft-penalizes consecutive duplicate cluster assignments.
    """
    __sis_version__ = 1


class GenerateAudioClusterNgramsJob(Job):
    """
    Takes base monophone-scale audio cluster token sequences (e.g. from K=40 segment K-Means)
    and generates compositional audio n-gram sequences (2-grams for diphones, 3-grams for triphones),
    filtering the vocabulary by cumulative coverage_threshold and/or max_vocab_size (matching GenerateTextNgramsJob).

    Exports a drop-in compatible interface with HuBERTCheatingSegmentAveragingAndClusterJob & GenerateTextNgramsJob:
      - out_ngram_text: unpruned sliding-window n-gram strings per utterance (for Word2Vec training)
      - out_collapsed_tokens_txt: integer n-gram IDs (0..V-1) per utterance for segment-level evaluation
      - out_raw_clusters_txt: alias to out_collapsed_tokens_txt
      - out_cluster_vocab_txt: top-V n-gram strings sorted by frequency descending (0..V-1 lines)
      - out_cluster_counts_txt: integer-indexed counts 'idx count' (0..V-1) for distribution & alignment analysis
      - out_ngram_counts_txt: string-keyed counts 'ngram_str count'
    """

    def __init__(
        self,
        base_clusters_txt: tk.Path,
        ngram_order: int = 2,
        delimiter: str = "_",
        min_count: int = 1,
        coverage_threshold: Optional[float] = 0.80,
        max_vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.base_clusters_txt = base_clusters_txt
        self.ngram_order = ngram_order
        self.delimiter = delimiter
        self.min_count = min_count
        self.coverage_threshold = coverage_threshold
        self.max_vocab_size = max_vocab_size

        self.out_ngram_text = self.output_path(f"audio.ngrams_{ngram_order}.txt")
        self.out_collapsed_tokens_txt = self.output_path("collapsed_clusters.txt")
        self.out_raw_clusters_txt = self.output_path("raw_clusters.txt")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")
        self.out_ngram_counts_txt = self.output_path("ngram_counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 1})

    def run(self):
        from collections import Counter

        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        counts = Counter()
        n = self.ngram_order
        delim = self.delimiter
        in_path = self.base_clusters_txt.get_path()

        # 1. Pass 1: Generate sliding-window n-grams (matching GenerateTextNgramsJob) and count occurrences
        with open(in_path, "r", encoding="utf-8") as f_in, open(
            self.out_ngram_text.get_path(), "w", encoding="utf-8"
        ) as f_out:
            for line in f_in:
                tokens = line.strip().split()
                if not tokens:
                    f_out.write("\n")
                    continue
                if len(tokens) < n:
                    ngrams = [delim.join(tokens)]
                else:
                    ngrams = [delim.join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
                counts.update(ngrams)
                f_out.write(" ".join(ngrams) + "\n")

        # 2. Filter vocabulary by min_count, coverage_threshold, and max_vocab_size
        total_occurrences = sum(counts.values())
        vocab = []
        cum_count = 0
        for w, c in counts.most_common():
            if c < self.min_count:
                break
            vocab.append(w)
            cum_count += c
            if self.coverage_threshold is not None and total_occurrences > 0:
                if (cum_count / total_occurrences) >= self.coverage_threshold:
                    break
            if self.max_vocab_size is not None and len(vocab) >= self.max_vocab_size:
                break

        logging.info(
            f"Generated audio {n}-grams: {len(counts):,} unique total, kept {len(vocab):,} "
            f"(coverage={cum_count / max(1, total_occurrences):.4f}, max_count={counts[vocab[0]] if vocab else 0:,})."
        )

        # 3. Build integer lookup and center-phone fallbacks for OOV / boundary positions
        ngram_to_id = {w: idx for idx, w in enumerate(vocab)}
        best_by_center = {}
        best_by_left_pair = {}
        best_by_right_pair = {}

        for idx, w in enumerate(vocab):
            parts = w.split(delim)
            if n >= 3 and len(parts) >= 3:
                mid = parts[len(parts) // 2]
                best_by_center.setdefault(mid, idx)
                best_by_left_pair.setdefault((parts[0], parts[1]), idx)
                best_by_right_pair.setdefault((parts[-2], parts[-1]), idx)
            else:
                best_by_center.setdefault(parts[0], idx)

        # 4. Pass 2: Write integer n-gram IDs per segment for evaluation (out_collapsed_tokens_txt / out_raw_clusters_txt)
        with open(in_path, "r", encoding="utf-8") as f_in, open(
            self.out_collapsed_tokens_txt.get_path(), "w", encoding="utf-8"
        ) as f_coll, open(self.out_raw_clusters_txt.get_path(), "w", encoding="utf-8") as f_raw:
            for line in f_in:
                tokens = line.strip().split()
                L = len(tokens)
                if L == 0:
                    f_coll.write("\n")
                    f_raw.write("\n")
                    continue

                seg_ids = []
                if n == 2:
                    for i in range(L):
                        if i < L - 1:
                            cand = f"{tokens[i]}{delim}{tokens[i + 1]}"
                            idx = ngram_to_id.get(cand, best_by_center.get(tokens[i], 0))
                        else:
                            idx = best_by_center.get(tokens[i], 0)
                        seg_ids.append(idx)
                elif n >= 3:
                    for i in range(L):
                        if 0 < i < L - 1:
                            cand = f"{tokens[i - 1]}{delim}{tokens[i]}{delim}{tokens[i + 1]}"
                            idx = ngram_to_id.get(cand)
                            if idx is None:
                                idx = best_by_left_pair.get(
                                    (tokens[i - 1], tokens[i]),
                                    best_by_right_pair.get(
                                        (tokens[i], tokens[i + 1]),
                                        best_by_center.get(tokens[i], 0),
                                    ),
                                )
                        elif i == 0:
                            if L > 1:
                                idx = best_by_right_pair.get(
                                    (tokens[0], tokens[1]),
                                    best_by_center.get(tokens[0], 0),
                                )
                            else:
                                idx = best_by_center.get(tokens[0], 0)
                        else:
                            if L > 1:
                                idx = best_by_left_pair.get(
                                    (tokens[L - 2], tokens[L - 1]),
                                    best_by_center.get(tokens[L - 1], 0),
                                )
                            else:
                                idx = best_by_center.get(tokens[L - 1], 0)
                        seg_ids.append(idx)
                else:
                    for t in tokens:
                        seg_ids.append(ngram_to_id.get(t, best_by_center.get(t, 0)))

                out_line = " ".join(str(idx) for idx in seg_ids) + "\n"
                f_coll.write(out_line)
                f_raw.write(out_line)

        # 5. Write vocabulary and counts files
        with open(self.out_cluster_vocab_txt.get_path(), "w", encoding="utf-8") as f_vocab, open(
            self.out_cluster_counts_txt.get_path(), "w", encoding="utf-8"
        ) as f_counts, open(self.out_ngram_counts_txt.get_path(), "w", encoding="utf-8") as f_ng_counts:
            for idx, w in enumerate(vocab):
                cnt = counts[w]
                f_vocab.write(f"{w}\n")
                f_counts.write(f"{idx} {cnt}\n")
                f_ng_counts.write(f"{w} {cnt}\n")
