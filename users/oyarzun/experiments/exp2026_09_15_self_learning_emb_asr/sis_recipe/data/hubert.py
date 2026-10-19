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

    __sis_hash_exclude__ = {"cache_dir": "/work/smt4/benjamin.oyarzun/.cache/huggingface", "batch_size": 16}

    def __init__(
        self,
        audio_dir: Optional[tk.Path] = None,
        audio_tsv_manifest: Optional[tk.Path] = None,
        model_name: str = "/work/smt4/benjamin.oyarzun/models/hubert-base-ls960",
        layer: int = 6,
        batch_size: int = 16,
        cache_dir: Optional[str] = "/work/smt4/benjamin.oyarzun/.cache/huggingface",
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


class HuBERTCheatingSegmentAveragingAndClusterJob(Job):
    """
    Ablation: Uses GMM oracle ("cheating") segmentation boundaries to pool HuBERT frame features.
    For each discovered non-silence phoneme segment, averages the frame-level representations across the segment.
    Then trains K-Means with K centroids on the pooled segment representations and quantizes each segment.
    Outputs the sequence of segment cluster indices as pseudo-tokens for Word2Vec training.
    """

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: tk.Path,
        segmentation_hdfs: List[tk.Path],
        num_clusters: int = 483,
        pca_components_npy: Optional[tk.Path] = None,
        pca_mean_npy: Optional[tk.Path] = None,
        sample_pct: float = 0.2,
        seed: int = 42,
        collapse_consecutive: bool = True,
        _version: int = 2,
    ):
        super().__init__()
        self._version = max(_version, 3) if pca_components_npy is not None else _version
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.segmentation_hdfs = segmentation_hdfs
        self.num_clusters = num_clusters
        self.pca_components_npy = pca_components_npy
        self.pca_mean_npy = pca_mean_npy
        self.sample_pct = sample_pct
        self.seed = seed
        self.collapse_consecutive = collapse_consecutive

        # Outputs (matches HuBERTClusterAndCollapseJob interface for downstream compatibility)
        self.out_collapsed_tokens_txt = self.output_path("collapsed_clusters.txt")
        self.out_raw_clusters_txt = self.output_path("raw_clusters.txt")
        self.out_centroids_npy = self.output_path("centroids.npy")
        self.out_cluster_vocab_txt = self.output_path("cluster_vocab.txt")
        self.out_cluster_counts_txt = self.output_path("cluster_counts.txt")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 8, "mem": 32, "time": 6})

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

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(seq_tags):
                logging.info(f"Segment pooling progress: {idx + 1:,}/{len(seq_tags):,} utts ({total_segments:,} segments pooled)")

        all_segment_reps = np.stack(all_segment_reps, axis=0)
        logging.info(f"Total pooled segment representations: {all_segment_reps.shape}")

        # 3b. Apply Frame-Fitted PCA Projection (if configured)
        if self.pca_components_npy is not None and self.pca_mean_npy is not None:
            logging.info("Applying frame-fitted PCA projection to segment representations...")
            pca_comp = np.load(self.pca_components_npy.get_path())
            pca_mean = np.load(self.pca_mean_npy.get_path())
            # Linearity: Average( (x - μ) @ W ) == (Average(x) - μ) @ W
            all_segment_reps = np.dot(all_segment_reps - pca_mean, pca_comp.T).astype(np.float32)
            logging.info(f"Projected segment representations to shape: {all_segment_reps.shape}")

        # 4. K-Means Clustering on Segment Representations
        n_sample = min(len(all_segment_reps), max(200000, int(len(all_segment_reps) * self.sample_pct)))
        rng = np.random.default_rng(self.seed)
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

        # 5. Quantize all segment representations to nearest centroid
        logging.info(f"Quantizing {len(all_segment_reps):,} segments to K={self.num_clusters} centroids...")
        chunk_size = 100000
        quantized_clusters = []
        for c_start in range(0, len(all_segment_reps), chunk_size):
            c_end = min(c_start + chunk_size, len(all_segment_reps))
            sub = all_segment_reps[c_start:c_end]
            scores = 2.0 * np.dot(sub, centroids.T) - centroid_norms
            preds = np.argmax(scores, axis=1).astype(np.int32)
            quantized_clusters.append(preds)
        all_cluster_ids = np.concatenate(quantized_clusters, axis=0)

        # 6. Reconstruct sequence tokens per utterance and compute counts
        cluster_counts = {k: 0 for k in range(self.num_clusters)}
        seg_ptr = 0

        with open(self.out_collapsed_tokens_txt.get_path(), "w") as f_tokens, \
             open(self.out_raw_clusters_txt.get_path(), "w") as f_raw:
            for n_segs in utt_segment_counts:
                u_cids = all_cluster_ids[seg_ptr : seg_ptr + n_segs].tolist()
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
            f"Cheating segmentation clustering completed successfully. "
            f"Total segments: {total_segments:,}, Unique clusters used: {sum(1 for c in cluster_counts.values() if c > 0)}/{self.num_clusters}"
        )
