"""
HuBERT layer 6 feature extraction, K-Means clustering, and run-length collapsed token generation.
"""

from typing import Optional, List, Dict, Iterator
import os
import glob
import logging
import numpy as np

from sisyphus import Job, Task, tk


class ExtractHuBERTLayerFeaturesJob(Job):
    """
    Extracts intermediate representations from HuBERT-base (e.g., Layer 6).
    """

    def __init__(
        self,
        audio_dir: Optional[tk.Path] = None,
        audio_tsv_manifest: Optional[tk.Path] = None,
        model_name: str = "facebook/hubert-base-ls960",
        layer: int = 6,
        batch_size: int = 16,
        cache_dir: Optional[str] = "/u/benjamin.oyarzun/cache/huggingface",
    ):
        super().__init__()
        assert audio_dir is not None or audio_tsv_manifest is not None, "Either audio_dir or audio_tsv_manifest must be given"
        self.audio_dir = audio_dir
        self.audio_tsv_manifest = audio_tsv_manifest
        self.model_name = model_name
        self.layer = layer
        self.batch_size = batch_size
        self.cache_dir = cache_dir

        self.out_features_npy = self.output_path("features.npy")
        self.out_lengths_txt = self.output_path("lengths.txt")
        self.out_seq_tags_txt = self.output_path("seq_tags.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 12})

    def run(self):
        logging.basicConfig(level=logging.INFO)
        import tempfile
        import shutil
        import torch
        import torchaudio
        from transformers import HubertModel, HubertConfig
        from numpy.lib.format import open_memmap

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Loading HuBERT model: {self.model_name} onto {device}...")

        config = HubertConfig.from_pretrained(self.model_name, cache_dir=self.cache_dir)
        config.output_hidden_states = True
        model = HubertModel.from_pretrained(self.model_name, config=config, cache_dir=self.cache_dir)
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
