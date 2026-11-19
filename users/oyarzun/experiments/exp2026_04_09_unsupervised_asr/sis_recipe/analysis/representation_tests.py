"""
Sisyphus Job: Consecutive Representation Tests for Cheating Segmentation (5.9.4+).
Evaluates whether segment representations over consecutive segments are sufficiently
discriminative to be mapped to distinct phonemes:
  - Test 1: Consecutive vs. Non-consecutive Distance (Contextual Smearing Index)
  - Test 2: Oracle Supervised Consecutive Discrimination (Spurious Geminate Rate)
  - Test 3: Unconstrained K-Means Collision Rate (K=40 vs K=512)
"""

import os
import sys
import json
import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sisyphus import tk, Task, Job


class ConsecutiveRepresentationTestJob(Job):
    """
    Evaluates Tests 1, 2, 3, and 4 on segment-averaged representations of consecutive oracle segments.
    Supports different pooling methods over segment frames (e.g. uniform mean vs center-weighted)
    and clustering methods (MiniBatchKMeans or SpectralClustering).
    """
    __sis_version__ = 4
    __sis_hash_exclude__ = {
        "clustering_method": "kmeans",
        "spectral_n_neighbors": 30,
        "spectral_sample_size": 25000,
    }

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: tk.Path,
        segmentation_hdfs: List[tk.Path],
        model_name: str = "hubert",
        pca_components_npy: Optional[tk.Path] = None,
        pca_mean_npy: Optional[tk.Path] = None,
        normalize_segments: str = "unit_sphere",
        num_utts: int = 1000,
        pooling_method: str = "mean",
        clustering_method: str = "kmeans",
        spectral_n_neighbors: int = 30,
        spectral_sample_size: int = 25000,
    ):
        super().__init__()
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.segmentation_hdfs = segmentation_hdfs
        self.model_name = model_name
        self.pca_components_npy = pca_components_npy
        self.pca_mean_npy = pca_mean_npy
        self.normalize_segments = normalize_segments
        self.num_utts = num_utts
        self.pooling_method = pooling_method
        self.clustering_method = clustering_method or "kmeans"
        self.spectral_n_neighbors = int(spectral_n_neighbors)
        self.spectral_sample_size = int(spectral_sample_size)

        # Test 1 outputs
        self.out_t1_summary_json = self.output_path("test1_consecutive_distance/distance_summary.json")
        self.out_t1_plot = self.output_path("test1_consecutive_distance/consecutive_vs_intra_distribution.png")

        # Test 2 outputs
        self.out_t2_summary_json = self.output_path("test2_supervised_discrimination/supervised_probe_metrics.json")
        self.out_t2_plot = self.output_path("test2_supervised_discrimination/supervised_probe_summary.png")

        # Test 3 outputs
        self.out_t3_summary_json = self.output_path("test3_kmeans_collision/kmeans_collision_summary.json")
        self.out_t3_plot = self.output_path("test3_kmeans_collision/collision_rate_vs_segment_duration.png")

        # Test 4 outputs (K=40 Cluster Ground-Truth Phoneme Purity & Composition + Word Boundary Extension)
        self.out_t4_summary_json = self.output_path("test4_cluster_phone_composition/cluster_phone_composition.json")
        self.out_t4_plot = self.output_path("test4_cluster_phone_composition/cluster_phone_composition.png")
        self.out_t4_wb_summary_json = self.output_path("test4_cluster_phone_composition/cluster_phone_composition_word_boundaries.json")
        self.out_t4_wb_plot = self.output_path("test4_cluster_phone_composition/cluster_phone_composition_word_boundaries.png")

    @staticmethod
    def _compute_pooling_weights(L: int, method: str = "mean") -> np.ndarray:
        """
        Computes 1D pooling weights of length L normalized so sum(weights) == 1.

        Methods:
          - "mean": uniform weights (1/L).
          - "center_weighted" or "triangular": symmetric triangular window with strictly positive
            boundary weights, tapering down towards segment boundaries.
          - "gaussian": gaussian window centered at (L-1)/2 with sigma = max(1.0, (L-1)/3.0).
          - "hann": Hann / cosine window with pedestal.
        """
        if L <= 2 or method == "mean":
            return np.full((L,), 1.0 / L, dtype=np.float32)

        indices = np.arange(L, dtype=np.float32)
        center = (L - 1.0) / 2.0

        if method in ("center_weighted", "triangular"):
            half_width = (L + 1.0) / 2.0
            weights = 1.0 - np.abs(indices - center) / half_width
        elif method == "gaussian":
            sigma = max(1.0, (L - 1.0) / 3.0)
            weights = np.exp(-0.5 * ((indices - center) / sigma) ** 2)
        elif method == "hann":
            weights = 0.5 * (1.0 - np.cos(2.0 * np.pi * (indices + 0.5) / L))
        else:
            raise ValueError(f"Unknown pooling method: {method}")

        weights = weights / np.sum(weights)
        return weights.astype(np.float32)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 32, "time": 2})

    def _pool_oracle_segments(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[set]]:
        import h5py

        # 1. Load sequence tags and lengths
        with open(self.seq_tags_txt.get_path(), "r") as f:
            seq_tags = [line.strip().split()[0] for line in f if line.strip()]
        with open(self.lengths_txt.get_path(), "r") as f:
            lengths = [int(line.strip()) for line in f if line.strip()]

        num_eval = min(self.num_utts, len(seq_tags))
        seq_tags = seq_tags[:num_eval]
        lengths = lengths[:num_eval]

        tag_to_idx = {tag: i for i, tag in enumerate(seq_tags)}
        clean_tag_to_idx = {tag.split("/")[-1]: i for i, tag in enumerate(seq_tags)}

        # Utterance offsets
        utt_offsets = [0]
        for l in lengths[:-1]:
            utt_offsets.append(utt_offsets[-1] + l)

        # 2. Extract oracle segments and labels from HDFs
        utt_segments = [[] for _ in range(num_eval)]
        utt_labels = [[] for _ in range(num_eval)]

        for hdf_path in self.segmentation_hdfs:
            p = hdf_path.get_path() if hasattr(hdf_path, "get_path") else str(hdf_path)
            if not os.path.exists(p):
                continue
            with h5py.File(p, "r") as f:
                if "seqTags" not in f or "seqLengths" not in f:
                    continue
                shard_tags = [t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in f["seqTags"][:]]
                clean_tags = [t.split("/")[-1] for t in shard_tags]
                matching = [i for i, (t, ct) in enumerate(zip(shard_tags, clean_tags)) if t in tag_to_idx or ct in clean_tag_to_idx]
                if not matching:
                    continue

                target_data = f["inputs"] if "inputs" in f else f["data"]
                shard_lens = f["seqLengths"][:, 0] if f["seqLengths"].ndim == 2 else f["seqLengths"][:]
                shard_offsets = np.concatenate([[0], np.cumsum(shard_lens[:-1])])

                for i in matching:
                    t = shard_tags[i]
                    ct = clean_tags[i]
                    u_idx = tag_to_idx.get(t, clean_tag_to_idx.get(ct))
                    s_off = shard_offsets[i]
                    s_len = shard_lens[i]
                    labels = target_data[s_off : s_off + s_len]

                    # Group runs of non-silence labels (label != 0)
                    cur_label, cur_start = None, None
                    segs, seg_labs = [], []
                    for t_idx, lab in enumerate(labels):
                        lab_val = int(lab)
                        if lab_val != cur_label:
                            if cur_label is not None and cur_label != 0:
                                segs.append((cur_start, t_idx - 1))
                                seg_labs.append(cur_label)
                            cur_label = lab_val
                            cur_start = t_idx
                    if cur_label is not None and cur_label != 0:
                        segs.append((cur_start, len(labels) - 1))
                        seg_labs.append(cur_label)

                    utt_segments[u_idx] = segs
                    utt_labels[u_idx] = seg_labs

        # 3. Load feature matrix and pool segments
        features = np.load(self.features_npy.get_path(), mmap_mode="r")
        pca_comp = np.load(self.pca_components_npy.get_path()).astype(np.float32) if self.pca_components_npy else None
        pca_mean = np.load(self.pca_mean_npy.get_path()).astype(np.float32) if self.pca_mean_npy else None

        segments_by_utt = []
        labels_by_utt = []
        durations_by_utt = []
        word_boundaries_by_utt = []

        # Optional transcript + lexicon loading for exact word boundary alignment (as in v5.9.1_spin_audio_word_boundaries)
        from i6_experiments.users.oyarzun.experiments.exp2026_09_15_self_learning_emb_asr.sis_recipe.data.hubert import (
            align_words_to_segments,
        )
        from .run_representation_tests import load_idx_to_phoneme_map

        idx_to_phone_map = load_idx_to_phoneme_map()
        wrd_to_phn = {}
        lex_candidates = [
            "/u/benjamin.oyarzun/experiments/2026_05_07_first_experiments/work/i6_core/tools/download/DownloadJob.Zx9QK2LMKdWk/output/librispeech-lexicon.txt",
            "/u/corpora/speech/LibriSpeech/librispeech-lexicon.txt",
        ]
        for lex_p in lex_candidates:
            if os.path.exists(lex_p):
                try:
                    with open(lex_p, "r", encoding="utf-8") as f_lex:
                        for line in f_lex:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                wrd = parts[0].lower()
                                phones = ["".join(c for c in p if not c.isdigit()) for p in parts[1:]]
                                if wrd not in wrd_to_phn:
                                    wrd_to_phn[wrd] = phones
                    break
                except Exception as e:
                    logging.warning(f"Could not load lexicon from {lex_p}: {e}")

        text_by_tag = {}
        lbs_audio_root = "/u/corpora/speech/LibriSpeech/LibriSpeech/train-clean-100"
        for u_idx in range(num_eval):
            t_clean = seq_tags[u_idx].split("/")[-1]
            t_parts = t_clean.split("-")
            if len(t_parts) >= 2:
                spk, chp = t_parts[0], t_parts[1]
                trans_p = os.path.join(lbs_audio_root, spk, chp, f"{spk}-{chp}.trans.txt")
                if trans_p not in text_by_tag and os.path.exists(trans_p):
                    try:
                        with open(trans_p, "r", encoding="utf-8") as f_tr:
                            for line in f_tr:
                                lp = line.strip().split(maxsplit=1)
                                if len(lp) == 2:
                                    text_by_tag[lp[0]] = lp[1].lower()
                        text_by_tag[trans_p] = True
                    except Exception:
                        pass

        for u_idx in range(num_eval):
            segs = utt_segments[u_idx]
            labs = utt_labels[u_idx]
            if not segs:
                continue
            off = utt_offsets[u_idx]
            u_len = lengths[u_idx]

            pooled = []
            valid_labs = []
            durs = []
            for (s, e), lab in zip(segs, labs):
                s_c = max(0, min(s, u_len - 1))
                e_c = max(s_c, min(e, u_len - 1))
                feat_block = features[off + s_c : off + e_c + 1]
                L = e_c - s_c + 1
                weights = self._compute_pooling_weights(L, self.pooling_method)
                pooled_v = np.sum(feat_block * weights[:, None], axis=0, dtype=np.float32)
                if pca_comp is not None and pca_mean is not None:
                    pooled_v = np.dot(pooled_v - pca_mean, pca_comp.T)
                # Unit sphere normalization
                if self.normalize_segments == "unit_sphere":
                    norm = np.linalg.norm(pooled_v)
                    pooled_v = pooled_v / max(norm, 1e-12)

                pooled.append(pooled_v)
                valid_labs.append(lab)
                durs.append(L)

            if pooled:
                segments_by_utt.append(np.array(pooled, dtype=np.float32))
                labels_by_utt.append(np.array(valid_labs, dtype=np.int32))
                durations_by_utt.append(np.array(durs, dtype=np.int32))

                t_clean = seq_tags[u_idx].split("/")[-1]
                words = text_by_tag.get(t_clean, "").split() if isinstance(text_by_tag.get(t_clean), str) else []
                ref_phones = [idx_to_phone_map.get(int(pid), str(pid)) for pid in valid_labs]
                wb = align_words_to_segments(words, ref_phones, wrd_to_phn) if (words and wrd_to_phn) else set()
                if not wb and len(segs) > 1:
                    for s_i in range(len(segs) - 1):
                        if segs[s_i + 1][0] > segs[s_i][1] + 1:
                            wb.add(s_i)
                word_boundaries_by_utt.append(wb)

        logging.info(f"Pooled segments for {len(segments_by_utt)} utterances (pooling_method='{self.pooling_method}').")
        if not segments_by_utt:
            raise RuntimeError(
                f"No segments were pooled across {num_eval} utterances! "
                f"Check sequence tag matching between {self.seq_tags_txt.get_path()} and segmentation HDFs."
            )
        return segments_by_utt, labels_by_utt, durations_by_utt, word_boundaries_by_utt

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        segments_by_utt, labels_by_utt, durations_by_utt, word_boundaries_by_utt = self._pool_oracle_segments()

        # Import runner functions from analysis / script
        try:
            from .run_representation_tests import (
                run_test1_consecutive_distance,
                run_test2_supervised_discrimination,
                run_test3_kmeans_collision,
                run_test4_cluster_phone_composition,
            )
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from run_representation_tests import (
                run_test1_consecutive_distance,
                run_test2_supervised_discrimination,
                run_test3_kmeans_collision,
                run_test4_cluster_phone_composition,
            )

        t1_dir = os.path.dirname(self.out_t1_summary_json.get_path())
        t2_dir = os.path.dirname(self.out_t2_summary_json.get_path())
        t3_dir = os.path.dirname(self.out_t3_summary_json.get_path())
        t4_dir = os.path.dirname(self.out_t4_summary_json.get_path())

        run_test1_consecutive_distance(segments_by_utt, labels_by_utt, durations_by_utt, t1_dir)
        run_test2_supervised_discrimination(segments_by_utt, labels_by_utt, t2_dir)
        run_test3_kmeans_collision(
            segments_by_utt,
            labels_by_utt,
            durations_by_utt,
            t3_dir,
            k_values=(40, 512),
            clustering_method=self.clustering_method,
            spectral_n_neighbors=self.spectral_n_neighbors,
            spectral_sample_size=self.spectral_sample_size,
        )
        run_test4_cluster_phone_composition(
            segments_by_utt,
            labels_by_utt,
            t4_dir,
            num_clusters=40,
            word_boundaries_by_utt=word_boundaries_by_utt,
            clustering_method=self.clustering_method,
            spectral_n_neighbors=self.spectral_n_neighbors,
            spectral_sample_size=self.spectral_sample_size,
        )

        logging.info("All 4 representation tests finished successfully.")


class ClusterPhoneCompositionTestJob(ConsecutiveRepresentationTestJob):
    """
    Dedicated Sisyphus Job for Test 4 (Cluster Ground-Truth Phoneme Purity & Composition +
    Word Boundary Extension) across arbitrary cluster counts (e.g., K=120 from v5.10, K=512 from Test 3),
    pooling methods (e.g., 'mean' vs 'center_weighted'), and clustering methods (K-Means vs Spectral Clustering).
    """
    __sis_version__ = 2
    __sis_hash_exclude__ = {
        "clustering_method": "kmeans",
        "spectral_n_neighbors": 30,
        "spectral_sample_size": 25000,
    }

    def __init__(
        self,
        features_npy: tk.Path,
        lengths_txt: tk.Path,
        seq_tags_txt: tk.Path,
        segmentation_hdfs: List[tk.Path],
        model_name: str = "hubert",
        pca_components_npy: Optional[tk.Path] = None,
        pca_mean_npy: Optional[tk.Path] = None,
        normalize_segments: str = "unit_sphere",
        num_utts: int = 1000,
        pooling_method: str = "mean",
        num_clusters: int = 120,
        clustering_method: str = "kmeans",
        spectral_n_neighbors: int = 30,
        spectral_sample_size: int = 25000,
    ):
        Job.__init__(self)
        self.features_npy = features_npy
        self.lengths_txt = lengths_txt
        self.seq_tags_txt = seq_tags_txt
        self.segmentation_hdfs = segmentation_hdfs
        self.model_name = model_name
        self.pca_components_npy = pca_components_npy
        self.pca_mean_npy = pca_mean_npy
        self.normalize_segments = normalize_segments
        self.num_utts = num_utts
        self.pooling_method = pooling_method
        self.num_clusters = int(num_clusters)
        self.clustering_method = clustering_method or "kmeans"
        self.spectral_n_neighbors = int(spectral_n_neighbors)
        self.spectral_sample_size = int(spectral_sample_size)

        # Test 4 outputs for K=num_clusters
        self.out_t4_summary_json = self.output_path("test4_cluster_phone_composition/cluster_phone_composition.json")
        self.out_t4_plot = self.output_path("test4_cluster_phone_composition/cluster_phone_composition.png")
        self.out_t4_wb_summary_json = self.output_path("test4_cluster_phone_composition/cluster_phone_composition_word_boundaries.json")
        self.out_t4_wb_plot = self.output_path("test4_cluster_phone_composition/cluster_phone_composition_word_boundaries.png")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", rqmt={"cpu": 4, "mem": 32, "time": 2})

    def run(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
        segments_by_utt, labels_by_utt, _, word_boundaries_by_utt = self._pool_oracle_segments()

        try:
            from .run_representation_tests import run_test4_cluster_phone_composition
        except ImportError:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from run_representation_tests import run_test4_cluster_phone_composition

        t4_dir = os.path.dirname(self.out_t4_summary_json.get_path())
        run_test4_cluster_phone_composition(
            segments_by_utt,
            labels_by_utt,
            t4_dir,
            num_clusters=self.num_clusters,
            word_boundaries_by_utt=word_boundaries_by_utt,
            clustering_method=self.clustering_method,
            spectral_n_neighbors=self.spectral_n_neighbors,
            spectral_sample_size=self.spectral_sample_size,
        )
        logging.info(
            f"Test 4 (K={self.num_clusters}, pooling='{self.pooling_method}', clustering='{self.clustering_method}') finished successfully."
        )
