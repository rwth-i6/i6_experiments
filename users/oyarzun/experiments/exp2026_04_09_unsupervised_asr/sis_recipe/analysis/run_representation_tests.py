#!/usr/bin/env python3
"""
Representation Tests Suite for Unsupervised ASR Cheating Segmentation
Tests the Consecutive Segment Discriminability Assumption (especially for 5.9.4+):
  - Test 1: Consecutive vs. Non-consecutive Distance & Contextual Smearing Index (CSR)
  - Test 2: Oracle Supervised Consecutive Discrimination (Spurious Geminate Rate)
  - Test 3: Unconstrained K-Means Collision Rate (K=40 vs K=512)

Supports:
  - hubert (HuBERT-base-ls960, Layer 12 & Layer 6)
  - spinhubert (Spin-HuBERT 2048, Layer 12)
  - wav2vec2.0 (Wav2Vec 2.0 Large-lv60, Layer 15)
"""

import os
import sys
import json
import argparse
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

# Use headless backend for matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)


def normalize_unit_sphere(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Projects row vectors onto the unit sphere (L2 norm = 1.0)."""
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(norms, eps)


# ==============================================================================
# TEST 1: Consecutive vs. Non-Consecutive Distance (Contextual Smearing Index)
# ==============================================================================

def run_test1_consecutive_distance(
    segments_by_utt: List[np.ndarray],
    labels_by_utt: List[np.ndarray],
    durations_by_utt: List[np.ndarray],
    output_dir: str,
    max_baseline_samples: int = 5000,
) -> Dict[str, Any]:
    """
    Computes pairwise distance distributions between consecutive segments
    vs. non-consecutive inter-class and intra-class baselines.
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("--- Running Test 1: Consecutive Distance & Contextual Smearing Index ---")

    consec_cos_dists = []
    consec_euc_dists = []
    consec_transitions = []
    consec_dur_mins = []

    # Map label to segment representations
    reps_by_phone = defaultdict(list)

    total_pairs = 0
    gt_geminates = 0

    for u_idx, (reps, labs, durs) in enumerate(zip(segments_by_utt, labels_by_utt, durations_by_utt)):
        n_segs = len(labs)
        for i in range(n_segs):
            lab = int(labs[i])
            if len(reps_by_phone[lab]) < 300:  # Cap per phoneme to avoid memory explosion
                reps_by_phone[lab].append(reps[i])

        for i in range(n_segs - 1):
            total_pairs += 1
            l1, l2 = int(labs[i]), int(labs[i + 1])
            if l1 == l2:
                gt_geminates += 1
                continue

            v1, v2 = reps[i], reps[i + 1]
            cos_d = float(1.0 - np.dot(v1, v2))
            euc_d = float(np.linalg.norm(v1 - v2))

            consec_cos_dists.append(cos_d)
            consec_euc_dists.append(euc_d)
            consec_transitions.append((l1, l2))
            consec_dur_mins.append(min(int(durs[i]), int(durs[i + 1])))

    consec_cos_dists = np.array(consec_cos_dists, dtype=np.float32)
    consec_euc_dists = np.array(consec_euc_dists, dtype=np.float32)

    # 1. Intra-class distances (same phoneme, different instances)
    intra_cos_dists = []
    for lab, vecs in reps_by_phone.items():
        if len(vecs) > 1:
            V = np.array(vecs, dtype=np.float32)
            n_v = len(V)
            # Sample up to 50 pairs
            idx1 = np.random.randint(0, n_v, size=min(100, n_v * 2))
            idx2 = np.random.randint(0, n_v, size=min(100, n_v * 2))
            valid = idx1 != idx2
            idx1, idx2 = idx1[valid], idx2[valid]
            if len(idx1) > 0:
                dots = np.sum(V[idx1] * V[idx2], axis=1)
                intra_cos_dists.extend(1.0 - dots)
    intra_cos_dists = np.array(intra_cos_dists, dtype=np.float32)

    # 2. Non-consecutive inter-class baseline (same transition (l1, l2) sampled randomly)
    nonconsec_cos_dists = []
    sampled_transitions = set(consec_transitions[:1000])
    for l1, l2 in sampled_transitions:
        v1_list = reps_by_phone.get(l1, [])
        v2_list = reps_by_phone.get(l2, [])
        if v1_list and v2_list:
            k_sample = min(10, len(v1_list), len(v2_list))
            for k in range(k_sample):
                i1 = np.random.randint(0, len(v1_list))
                i2 = np.random.randint(0, len(v2_list))
                nonconsec_cos_dists.append(1.0 - np.dot(v1_list[i1], v2_list[i2]))
    nonconsec_cos_dists = np.array(nonconsec_cos_dists, dtype=np.float32)

    mean_consec_cos = float(np.mean(consec_cos_dists)) if len(consec_cos_dists) > 0 else 0.0
    mean_intra_cos = float(np.mean(intra_cos_dists)) if len(intra_cos_dists) > 0 else 0.0
    mean_nonconsec_cos = float(np.mean(nonconsec_cos_dists)) if len(nonconsec_cos_dists) > 0 else 0.0

    csr = float(mean_consec_cos / max(mean_nonconsec_cos, 1e-6))
    margin = float(mean_consec_cos - mean_intra_cos)

    results = {
        "test": "test1_consecutive_distance",
        "total_consecutive_pairs": int(total_pairs),
        "gt_geminate_pairs": int(gt_geminates),
        "gt_geminate_rate": float(gt_geminates / max(total_pairs, 1)),
        "mean_consecutive_cosine_distance": mean_consec_cos,
        "median_consecutive_cosine_distance": float(np.median(consec_cos_dists)),
        "mean_intra_class_cosine_distance": mean_intra_cos,
        "mean_nonconsecutive_inter_cosine_distance": mean_nonconsec_cos,
        "contextual_smearing_ratio_csr": csr,
        "discriminative_margin": margin,
        "is_assumption_supported": bool(margin > 0.0 and csr > 0.90),
    }

    # Save JSON summary
    json_path = os.path.join(output_dir, "distance_summary.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Generate Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 2.0, 50)
    ax.hist(consec_cos_dists, bins=bins, alpha=0.6, label=f"Consecutive Segments (Mean: {mean_consec_cos:.3f})", color="royalblue", density=True)
    ax.hist(intra_cos_dists, bins=bins, alpha=0.6, label=f"Intra-Phoneme Pairs (Mean: {mean_intra_cos:.3f})", color="forestgreen", density=True)
    ax.axvline(mean_consec_cos, color="blue", linestyle="--", linewidth=1.5)
    ax.axvline(mean_intra_cos, color="green", linestyle="--", linewidth=1.5)
    ax.set_title(f"Test 1: Consecutive Distance vs. Intra-Phone Spread (CSR: {csr:.3f})")
    ax.set_xlabel("Cosine Distance (1 - cos)")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)
    plot_path = os.path.join(output_dir, "consecutive_vs_intra_distribution.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    logging.info(f"Test 1 Complete: Mean Consec CosDist={mean_consec_cos:.4f}, Intra CosDist={mean_intra_cos:.4f}, CSR={csr:.4f}, Margin={margin:.4f}")
    return results


# ==============================================================================
# TEST 2: Oracle Supervised Consecutive Discrimination Probe
# ==============================================================================

def run_test2_supervised_discrimination(
    segments_by_utt: List[np.ndarray],
    labels_by_utt: List[np.ndarray],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Evaluates an oracle supervised Nearest-Centroid probe against reference phonemes.
    Measures the spurious consecutive geminate prediction rate: P(y_hat_t == y_hat_{t+1} | y_t != y_{t+1}).
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("--- Running Test 2: Oracle Supervised Consecutive Discrimination ---")

    # 1. Compute ground truth phoneme centroids
    phone_sums = defaultdict(lambda: np.zeros(segments_by_utt[0].shape[1], dtype=np.float64))
    phone_counts = defaultdict(int)

    for reps, labs in zip(segments_by_utt, labels_by_utt):
        for vec, lab in zip(reps, labs):
            l = int(lab)
            phone_sums[l] += vec
            phone_counts[l] += 1

    unique_phones = sorted(phone_sums.keys())
    centroids = np.zeros((len(unique_phones), segments_by_utt[0].shape[1]), dtype=np.float32)
    phone_to_idx = {p: i for i, p in enumerate(unique_phones)}

    for p in unique_phones:
        c = phone_sums[p] / max(phone_counts[p], 1)
        centroids[phone_to_idx[p]] = c / max(np.linalg.norm(c), 1e-12)

    # 2. Evaluate nearest-centroid predictions
    total_frames = 0
    correct_frames = 0
    total_consec_pairs = 0
    spurious_geminates = 0
    pairwise_correct = 0

    for reps, labs in zip(segments_by_utt, labels_by_utt):
        if len(labs) == 0:
            continue
        # Dot product with normalized centroids -> cosine sim
        sims = np.dot(reps, centroids.T)
        pred_indices = np.argmax(sims, axis=1)
        pred_phones = np.array([unique_phones[i] for i in pred_indices])

        correct_frames += int(np.sum(pred_phones == labs))
        total_frames += len(labs)

        for i in range(len(labs) - 1):
            if labs[i] == labs[i + 1]:
                continue
            total_consec_pairs += 1
            if pred_phones[i] == pred_phones[i + 1]:
                spurious_geminates += 1
            if pred_phones[i] == labs[i] and pred_phones[i + 1] == labs[i + 1]:
                pairwise_correct += 1

    frame_acc = float(correct_frames / max(total_frames, 1))
    per_floor = float(1.0 - frame_acc)
    spurious_geminate_rate = float(spurious_geminates / max(total_consec_pairs, 1))
    pairwise_transition_acc = float(pairwise_correct / max(total_consec_pairs, 1))

    results = {
        "test": "test2_supervised_discrimination",
        "num_classes": len(unique_phones),
        "total_frames_evaluated": int(total_frames),
        "frame_accuracy_ceiling": frame_acc,
        "oracle_per_floor": per_floor,
        "total_consecutive_non_geminate_pairs": int(total_consec_pairs),
        "spurious_consecutive_geminate_predictions": int(spurious_geminates),
        "spurious_geminate_rate": spurious_geminate_rate,
        "pairwise_transition_accuracy": pairwise_transition_acc,
        "is_assumption_supported": bool(spurious_geminate_rate < 0.10 and frame_acc > 0.60),
    }

    json_path = os.path.join(output_dir, "supervised_probe_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Bar plot for summary
    fig, ax = plt.subplots(figsize=(7, 4.5))
    metrics = ["Frame Accuracy", "Pairwise Transition Acc", "Spurious Geminate Rate"]
    vals = [frame_acc, pairwise_transition_acc, spurious_geminate_rate]
    colors = ["forestgreen", "royalblue", "crimson"]
    bars = ax.bar(metrics, vals, color=colors, width=0.5)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.02, f"{yval * 100:.1f}%", ha="center", va="bottom", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Test 2: Supervised Linear Probe (Spurious Geminates: {spurious_geminate_rate*100:.1f}%)")
    ax.grid(True, axis="y", linestyle=":", alpha=0.6)
    plot_path = os.path.join(output_dir, "supervised_probe_summary.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    logging.info(f"Test 2 Complete: Frame Acc={frame_acc*100:.2f}%, Spurious Geminate Rate={spurious_geminate_rate*100:.2f}%")
    return results


# ==============================================================================
# Spectral Clustering Utilities
# ==============================================================================

def fit_and_predict_spectral(
    all_segments: np.ndarray,
    num_clusters: int,
    n_neighbors: int = 30,
    sample_size: int = 25000,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fits SpectralClustering with nearest_neighbors affinity on segment representations.
    Returns:
      (centroids, cluster_assignments_for_all_segments)
    """
    from sklearn.cluster import SpectralClustering
    n_samples = len(all_segments)
    feat_dim = all_segments.shape[1]
    K = int(num_clusters)

    actual_sample_size = min(n_samples, max(K * 5, sample_size))
    rng = np.random.default_rng(random_state)

    if n_samples <= actual_sample_size:
        sample_reps = all_segments
    else:
        sample_indices = rng.choice(n_samples, size=actual_sample_size, replace=False)
        sample_reps = all_segments[sample_indices]

    logging.info(
        f"Fitting SpectralClustering (K={K}, affinity='nearest_neighbors', n_neighbors={n_neighbors}) "
        f"on {len(sample_reps):,} segments..."
    )
    spectral = SpectralClustering(
        n_clusters=K,
        affinity="nearest_neighbors",
        n_neighbors=min(n_neighbors, len(sample_reps) - 1),
        assign_labels="kmeans",
        random_state=random_state,
        n_init=3,
        n_jobs=-1,
    )
    sample_labels = spectral.fit_predict(sample_reps)

    # Derive cluster centroids in the feature representation space
    centroids = np.zeros((K, feat_dim), dtype=np.float32)
    global_mean = np.mean(sample_reps, axis=0, dtype=np.float32)
    for k in range(K):
        mask_k = (sample_labels == k)
        if np.any(mask_k):
            c_vec = np.mean(sample_reps[mask_k], axis=0, dtype=np.float32)
        else:
            c_vec = global_mean.copy()
        norm = np.linalg.norm(c_vec)
        if norm > 1e-12:
            c_vec /= norm
        centroids[k] = c_vec

    # Normalize centroids for cosine distance assignment
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids = centroids / np.maximum(c_norms, 1e-12)

    # Assign all segments via nearest centroid (dot product on normalized representations)
    batch_size = 8192
    assignments = np.zeros(n_samples, dtype=np.int32)
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch = all_segments[start_idx:end_idx]
        sims = np.dot(batch, centroids.T)
        assignments[start_idx:end_idx] = np.argmax(sims, axis=1)

    return centroids, assignments


# ==============================================================================
# TEST 3: Unconstrained Clustering Collision Rate (K-Means & Spectral)
# ==============================================================================

def run_test3_kmeans_collision(
    segments_by_utt: List[np.ndarray],
    labels_by_utt: List[np.ndarray],
    durations_by_utt: List[np.ndarray],
    output_dir: str,
    k_values: Tuple[int, ...] = (40, 512),
    clustering_method: str = "kmeans",
    spectral_n_neighbors: int = 30,
    spectral_sample_size: int = 25000,
) -> Dict[str, Any]:
    """
    Fits unconstrained clustering (MiniBatchKMeans or SpectralClustering) for K in {40, 512} and computes:
      P(c_t == c_{t+1} | y_t != y_{t+1})
    broken down by segment duration.
    """
    from sklearn.cluster import MiniBatchKMeans

    c_method = (clustering_method or "kmeans").strip().lower()
    is_spectral = c_method in ("spectral", "spectral_clustering")
    method_title = "Spectral Clustering" if is_spectral else "K-Means"

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"--- Running Test 3: Unconstrained {method_title} Collision Rate ---")

    all_segments = np.concatenate(segments_by_utt, axis=0)
    logging.info(f"Total segments for {method_title} fitting: {len(all_segments):,}")

    summary_results = {
        "test": "test3_kmeans_collision",
        "clustering_method": "spectral" if is_spectral else "kmeans",
        "results_by_k": {},
    }
    fig, ax = plt.subplots(figsize=(8, 5))

    for K in k_values:
        if is_spectral:
            logging.info(f"Training SpectralClustering with K={K}...")
            centroids, _ = fit_and_predict_spectral(
                all_segments,
                num_clusters=K,
                n_neighbors=spectral_n_neighbors,
                sample_size=spectral_sample_size,
            )
            def predict_clusters(X):
                sims = np.dot(X, centroids.T)
                return np.argmax(sims, axis=1)
        else:
            logging.info(f"Training MiniBatchKMeans with K={K}...")
            km = MiniBatchKMeans(n_clusters=K, batch_size=2048, random_state=42, n_init=3, max_iter=50)
            km.fit(all_segments)
            def predict_clusters(X):
                return km.predict(X)

        total_consec = 0
        total_collisions = 0
        collisions_by_dur = defaultdict(int)
        counts_by_dur = defaultdict(int)

        for reps, labs, durs in zip(segments_by_utt, labels_by_utt, durations_by_utt):
            clusters = predict_clusters(reps)
            for i in range(len(labs) - 1):
                if labs[i] == labs[i + 1]:
                    continue
                total_consec += 1
                min_d = min(int(durs[i]), int(durs[i + 1]))
                d_bin = min(min_d, 6)  # Cap at 6+
                counts_by_dur[d_bin] += 1

                if clusters[i] == clusters[i + 1]:
                    total_collisions += 1
                    collisions_by_dur[d_bin] += 1

        overall_collision_rate = float(total_collisions / max(total_consec, 1))
        dur_breakdown = {}
        for d in sorted(counts_by_dur.keys()):
            r = float(collisions_by_dur[d] / max(counts_by_dur[d], 1))
            dur_breakdown[f"min_duration_{d}_frames"] = {"pairs": counts_by_dur[d], "collisions": collisions_by_dur[d], "collision_rate": r}

        k_res = {
            "num_clusters": K,
            "clustering_method": "spectral" if is_spectral else "kmeans",
            "total_consec_pairs": int(total_consec),
            "total_cluster_collisions": int(total_collisions),
            "overall_collision_rate": overall_collision_rate,
            "duration_breakdown": dur_breakdown,
        }
        summary_results["results_by_k"][f"k{K}"] = k_res

        json_path = os.path.join(output_dir, f"kmeans_collision_k{K}.json")
        with open(json_path, "w") as f:
            json.dump(k_res, f, indent=2)

        # Plot curve for duration
        dur_bins = sorted(counts_by_dur.keys())
        rates = [float(collisions_by_dur[d] / max(counts_by_dur[d], 1)) * 100 for d in dur_bins]
        ax.plot(dur_bins, rates, marker="o", linewidth=2, label=f"K={K} ({method_title}, Overall: {overall_collision_rate*100:.1f}%)")

    ax.set_title(f"Test 3: {method_title} Cluster Collision Rate vs. Segment Duration")
    ax.set_xlabel("Min Segment Duration in Frames (1 frame = 20 ms)")
    ax.set_ylabel("Consecutive Collision Rate (%)")
    ax.set_xticks(sorted(counts_by_dur.keys()))
    ax.set_xticklabels([f"{d}" if d < 6 else "6+" for d in sorted(counts_by_dur.keys())])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "collision_rate_vs_segment_duration.png")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    summary_json_path = os.path.join(output_dir, "kmeans_collision_summary.json")
    with open(summary_json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    logging.info("Test 3 Complete.")
    return summary_results


# ==============================================================================
# TEST 4: K=40 Cluster Ground-Truth Phoneme Purity & Composition Analysis
# ==============================================================================

DEFAULT_IDX_TO_PHONEME: Dict[int, str] = {
    0: "SIL",
    1: "AA", 2: "AE", 3: "AH", 4: "AO", 5: "AW", 6: "AY", 7: "B", 8: "CH",
    9: "D", 10: "DH", 11: "EH", 12: "ER", 13: "EY", 14: "F", 15: "G", 16: "HH",
    17: "IH", 18: "IY", 19: "JH", 20: "K", 21: "L", 22: "M", 23: "N", 24: "NG",
    25: "OW", 26: "OY", 27: "P", 28: "R", 29: "S", 30: "SH", 31: "T", 32: "TH",
    33: "UH", 34: "UW", 35: "V", 36: "W", 37: "Y", 38: "Z", 39: "ZH",
}


def load_idx_to_phoneme_map(idx_to_phoneme_path: Optional[str] = None) -> Dict[int, str]:
    """
    Loads the integer-to-phoneme mapping (stripping word-end '#' markers so all 39 canonical
    ARPAbet phonemes are grouped cleanly). Falls back to DEFAULT_IDX_TO_PHONEME.
    """
    mapping = dict(DEFAULT_IDX_TO_PHONEME)
    # Add word-boundary variants 40..78 -> 1..39
    for idx in range(1, 40):
        mapping[idx + 39] = DEFAULT_IDX_TO_PHONEME[idx]

    default_disk_path = "/u/zyang/setups/mini/output/lbs_mono_phone_eow/idx_to_phoneme.txt"
    path_to_use = idx_to_phoneme_path if (idx_to_phoneme_path and os.path.exists(idx_to_phoneme_path)) else default_disk_path
    if os.path.exists(path_to_use):
        try:
            with open(path_to_use, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        idx_val = int(parts[0])
                        phn_name = parts[1].rstrip("#")
                        if phn_name in ("[SILENCE]", "<SIL>", "SIL"):
                            phn_name = "SIL"
                        mapping[idx_val] = phn_name
        except Exception as e:
            logging.warning(f"Could not read idx_to_phoneme from {path_to_use}: {e}")
    return mapping


def run_test4_cluster_phone_composition(
    segments_by_utt: List[np.ndarray],
    labels_by_utt: List[np.ndarray],
    output_dir: str,
    num_clusters: int = 40,
    idx_to_phoneme_path: Optional[str] = None,
    word_boundaries_by_utt: Optional[List[Any]] = None,
    word_boundary_token: str = "<w>",
    clustering_method: str = "kmeans",
    spectral_n_neighbors: int = 30,
    spectral_sample_size: int = 25000,
) -> Dict[str, Any]:
    """
    Test 4: Fits clustering (MiniBatchKMeans or SpectralClustering) with K clusters on pooled segment
    representations and analyzes for each of the K clusters:
      1. How many ground-truth phones are contained in the cluster (total unique, >=1%, >=5%, effective perplexity).
      2. The exact percentage breakdown of each ground-truth phone within the cluster.
    Generates:
      - cluster_phone_composition.json
      - cluster_phone_composition.png (2-panel figure: Stacked Top-Phone Composition + K x 39 Percentage Heatmap)
    """
    from sklearn.cluster import MiniBatchKMeans

    c_method = (clustering_method or "kmeans").strip().lower()
    is_spectral = c_method in ("spectral", "spectral_clustering")
    method_title = "Spectral Clustering" if is_spectral else "K-Means"

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"--- Running Test 4: {method_title} K={num_clusters} Cluster Ground-Truth Phoneme Composition ---")

    idx_to_phone = load_idx_to_phoneme_map(idx_to_phoneme_path)

    all_segments = np.concatenate(segments_by_utt, axis=0)
    all_labels_raw = np.concatenate(labels_by_utt, axis=0)
    total_segments = len(all_segments)

    if is_spectral:
        _, cluster_assignments = fit_and_predict_spectral(
            all_segments,
            num_clusters=num_clusters,
            n_neighbors=spectral_n_neighbors,
            sample_size=spectral_sample_size,
        )
    else:
        km = MiniBatchKMeans(
            n_clusters=num_clusters,
            batch_size=2048,
            random_state=42,
            n_init=3,
            max_iter=50,
        )
        cluster_assignments = km.fit_predict(all_segments)

    # Map integer labels to canonical phoneme strings
    all_phone_names = [idx_to_phone.get(int(lab), f"P{int(lab)}") for lab in all_labels_raw]
    unique_phone_names = sorted(
        list({p for p in all_phone_names if p != "SIL"}),
        key=lambda x: (len(x), x) if x.startswith("P") else (0, x),
    )
    if not unique_phone_names:
        unique_phone_names = sorted(list(set(all_phone_names)))

    phone_to_col = {p: i for i, p in enumerate(unique_phone_names)}
    num_phones = len(unique_phone_names)

    # Contingency matrix: (K x num_phones)
    counts_matrix = np.zeros((num_clusters, num_phones), dtype=np.int64)
    for c_idx, p_name in zip(cluster_assignments, all_phone_names):
        if p_name in phone_to_col:
            counts_matrix[int(c_idx), phone_to_col[p_name]] += 1

    cluster_totals = np.sum(counts_matrix, axis=1)  # (K,)
    pct_matrix = np.zeros((num_clusters, num_phones), dtype=np.float64)
    for c in range(num_clusters):
        if cluster_totals[c] > 0:
            pct_matrix[c, :] = (counts_matrix[c, :] / float(cluster_totals[c])) * 100.0

    clusters_report = {}
    per_cluster_list = []

    for c in range(num_clusters):
        c_tot = int(cluster_totals[c])
        c_counts = counts_matrix[c]
        c_pcts = pct_matrix[c]

        # Sort phones in this cluster by count descending
        nonzero_indices = [i for i in np.argsort(-c_counts) if c_counts[i] > 0]
        num_unique_contained = len(nonzero_indices)
        num_ge_1pct = int(np.sum(c_pcts >= 1.0))
        num_ge_5pct = int(np.sum(c_pcts >= 5.0))
        num_ge_10pct = int(np.sum(c_pcts >= 10.0))

        # Shannon entropy & perplexity (effective number of phones)
        if c_tot > 0:
            probs = c_counts[nonzero_indices] / float(c_tot)
            entropy_bits = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
            eff_phones = float(2.0 ** entropy_bits)
        else:
            entropy_bits = 0.0
            eff_phones = 0.0

        dom_phone = unique_phone_names[nonzero_indices[0]] if nonzero_indices else "NONE"
        dom_pct = float(c_pcts[nonzero_indices[0]]) if nonzero_indices else 0.0

        phone_dist = {}
        top_5_list = []
        for rank_i, p_i in enumerate(nonzero_indices):
            p_str = unique_phone_names[p_i]
            cnt_val = int(c_counts[p_i])
            pct_val = float(c_pcts[p_i])
            phone_dist[p_str] = {
                "count": cnt_val,
                "percentage": round(pct_val, 4),
            }
            if rank_i < 5:
                top_5_list.append({
                    "phone": p_str,
                    "count": cnt_val,
                    "percentage": round(pct_val, 2),
                })

        c_info = {
            "cluster_id": c,
            "total_segments": c_tot,
            "corpus_share_pct": round(100.0 * c_tot / max(total_segments, 1), 4),
            "num_unique_phones_contained": num_unique_contained,
            "num_phones_ge_1pct": num_ge_1pct,
            "num_phones_ge_5pct": num_ge_5pct,
            "num_phones_ge_10pct": num_ge_10pct,
            "entropy_bits": round(entropy_bits, 4),
            "effective_num_phones_perplexity": round(eff_phones, 4),
            "dominant_phone": dom_phone,
            "dominant_phone_pct": round(dom_pct, 4),
            "top_5_phones": top_5_list,
            "phone_distribution": phone_dist,
        }
        clusters_report[f"cluster_{c}"] = c_info
        per_cluster_list.append(c_info)

    # Compute global summary statistics
    weighted_purity = float(
        np.sum(np.max(counts_matrix, axis=1)) / max(np.sum(counts_matrix), 1) * 100.0
    )
    unweighted_purity = float(np.mean([c["dominant_phone_pct"] for c in per_cluster_list]))
    mean_unique_phones = float(np.mean([c["num_unique_phones_contained"] for c in per_cluster_list]))
    mean_ge_1pct = float(np.mean([c["num_phones_ge_1pct"] for c in per_cluster_list]))
    mean_ge_5pct = float(np.mean([c["num_phones_ge_5pct"] for c in per_cluster_list]))
    mean_eff_phones = float(np.mean([c["effective_num_phones_perplexity"] for c in per_cluster_list]))
    unique_dominant_phones = sorted(list({c["dominant_phone"] for c in per_cluster_list if c["total_segments"] > 0}))

    summary_results = {
        "test": "test4_cluster_phone_composition",
        "clustering_method": "spectral" if is_spectral else "kmeans",
        "num_clusters": num_clusters,
        "num_ground_truth_phones": num_phones,
        "total_segments_evaluated": int(total_segments),
        "overall_weighted_purity_pct": round(weighted_purity, 4),
        "mean_unweighted_purity_pct": round(unweighted_purity, 4),
        "mean_unique_phones_per_cluster": round(mean_unique_phones, 2),
        "mean_phones_ge_1pct_per_cluster": round(mean_ge_1pct, 2),
        "mean_phones_ge_5pct_per_cluster": round(mean_ge_5pct, 2),
        "mean_effective_phones_perplexity": round(mean_eff_phones, 2),
        "num_unique_dominant_phones_covered": len(unique_dominant_phones),
        "unique_dominant_phones_covered": unique_dominant_phones,
        "clusters": clusters_report,
    }

    json_path = os.path.join(output_dir, "cluster_phone_composition.json")
    with open(json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    # Sort clusters by dominant phone (and then by dominant phone percentage descending) for clean plotting
    sorted_clusters = sorted(
        per_cluster_list,
        key=lambda c: (
            phone_to_col.get(c["dominant_phone"], 999),
            -c["dominant_phone_pct"],
        ),
    )
    sorted_c_ids = [c["cluster_id"] for c in sorted_clusters]

    # Build a distinct color palette for the 39 phones
    cmap_20 = plt.cm.get_cmap("tab20", 20)
    cmap_20b = plt.cm.get_cmap("tab20b", 20)
    phone_colors = {}
    for i, p_name in enumerate(unique_phone_names):
        if i < 20:
            phone_colors[p_name] = cmap_20(i)
        else:
            phone_colors[p_name] = cmap_20b(i - 20)

    # Create 2-panel visualization:
    # Panel 1 (Left): Horizontal Stacked Percentage Bar Chart of Top Phones + Phone Counts per Cluster
    # Panel 2 (Right): Full K x 39 Cluster-by-Phoneme Percentage Heatmap
    if num_clusters <= 50:
        fig_height = 14.0
        ytick_fs = 8.0
        bar_text_fs = 6.5
        rem_text_fs = 6.0
        right_text_fs = 7.2
        hm_text_fs = 6.0
        min_bar_val_for_label = 1.5
        min_hm_val_for_label = 2.0
        single_line_bar_label = False
        save_dpi = 180
    elif num_clusters <= 150:
        fig_height = max(24.0, num_clusters * 0.23)
        ytick_fs = 6.2
        bar_text_fs = 5.2
        rem_text_fs = 5.0
        right_text_fs = 5.8
        hm_text_fs = 4.8
        min_bar_val_for_label = 1.5
        min_hm_val_for_label = 2.0
        single_line_bar_label = True
        save_dpi = 160
    elif num_clusters <= 600:
        fig_height = min(65.0, max(45.0, num_clusters * 0.115))
        ytick_fs = 3.6
        bar_text_fs = 3.5
        rem_text_fs = 3.4
        right_text_fs = 3.8
        hm_text_fs = 3.0
        min_bar_val_for_label = 1.5
        min_hm_val_for_label = 4.0
        single_line_bar_label = True
        save_dpi = 150
    else:
        fig_height = min(85.0, max(60.0, num_clusters * 0.045))
        ytick_fs = 2.0
        bar_text_fs = 2.0
        rem_text_fs = 2.0
        right_text_fs = 2.5
        hm_text_fs = 2.0
        min_bar_val_for_label = 3.0
        min_hm_val_for_label = 10.0
        single_line_bar_label = True
        save_dpi = 120

    cid_digits = 4 if num_clusters >= 1000 else (3 if num_clusters >= 100 else 2)

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(22, fig_height),
        gridspec_kw={"width_ratios": [1.15, 1.35]},
    )

    y_pos = np.arange(num_clusters)
    top_n_stack = 8

    for row_idx, c_info in enumerate(sorted_clusters):
        c_id = c_info["cluster_id"]
        c_pcts = pct_matrix[c_id]
        sorted_p_indices = [i for i in np.argsort(-c_pcts) if c_pcts[i] > 0]

        # Include all top phones up to top_n_stack that are above the infrequent cutoff (>= min_bar_val_for_label),
        # always keeping at least the dominant phone (rank 0).
        shown_p_indices = [
            p_idx for rank_i, p_idx in enumerate(sorted_p_indices[:top_n_stack])
            if rank_i == 0 or float(c_pcts[p_idx]) >= min_bar_val_for_label
        ]

        left_acc = 0.0
        for p_idx in shown_p_indices:
            p_name = unique_phone_names[p_idx]
            val = float(c_pcts[p_idx])
            color = phone_colors.get(p_name, "steelblue")
            ax1.barh(
                row_idx,
                val,
                left=left_acc,
                height=0.78,
                color=color,
                edgecolor="white",
                linewidth=0.5 if num_clusters <= 150 else 0.25,
            )
            if val >= min_bar_val_for_label:
                if val >= 6.5:
                    cur_fs = bar_text_fs
                    lbl_txt = f"{p_name} {val:.0f}%" if single_line_bar_label else f"{p_name}\n{val:.0f}%"
                elif val >= 3.5:
                    cur_fs = max(2.8, bar_text_fs * 0.82)
                    lbl_txt = f"{p_name}" if single_line_bar_label else f"{p_name}\n{val:.0f}%"
                else:
                    cur_fs = max(2.5, bar_text_fs * 0.70)
                    lbl_txt = f"{p_name}"
                ax1.text(
                    left_acc + val / 2.0,
                    row_idx,
                    lbl_txt,
                    ha="center",
                    va="center",
                    fontsize=cur_fs,
                    fontweight="bold",
                    color="white" if val >= 12.0 else "black",
                )
            left_acc += val

        rem_val = max(0.0, 100.0 - left_acc)
        if rem_val > 0.05:
            rem_count = max(0, len(sorted_p_indices) - len(shown_p_indices))
            ax1.barh(
                row_idx,
                rem_val,
                left=left_acc,
                height=0.78,
                color="#d9d9d9",
                edgecolor="white",
                linewidth=0.5 if num_clusters <= 150 else 0.25,
            )
            if rem_val >= 4.0 and rem_count > 0:
                rem_txt = f"+{rem_count} ({rem_val:.0f}%)" if (single_line_bar_label or rem_val < 8.0) else f"+{rem_count} phns\n({rem_val:.0f}%)"
                ax1.text(
                    left_acc + rem_val / 2.0,
                    row_idx,
                    rem_txt,
                    ha="center",
                    va="center",
                    fontsize=rem_text_fs if rem_val >= 8.0 else max(2.6, rem_text_fs * 0.8),
                    color="#333333",
                )

        # Annotate phone counts on the right edge of each bar
        n_tot_p = c_info["num_unique_phones_contained"]
        n_5_p = c_info["num_phones_ge_5pct"]
        n_1_p = c_info["num_phones_ge_1pct"]
        eff_p = c_info["effective_num_phones_perplexity"]
        ax1.text(
            101.2,
            row_idx,
            f"{n_tot_p} phns (≥5%:{n_5_p}, ≥1%:{n_1_p}, eff:{eff_p:.1f})",
            ha="left",
            va="center",
            fontsize=right_text_fs,
            family="monospace",
            color="#222222",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(
        [f"C{c['cluster_id']:0{cid_digits}d} (n={c['total_segments']})" for c in sorted_clusters],
        fontsize=ytick_fs,
        fontweight="bold",
    )
    ax1.invert_yaxis()
    ax1.set_xlim(0, 138)
    ax1.set_xticks(np.arange(0, 101, 20))
    ax1.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
    ax1.set_xlabel("Ground-Truth Phoneme Percentage Breakdown (%)", fontsize=10, fontweight="bold")
    ax1.set_title(
        f"[{method_title}] Per-Cluster Phone Breakdown & Contained Phone Counts\n"
        f"(Weighted Purity: {weighted_purity:.1f}% | Mean Phones ≥5%: {mean_ge_5pct:.1f}, ≥1%: {mean_ge_1pct:.1f}, Total: {mean_unique_phones:.1f})",
        fontsize=10.5,
        fontweight="bold",
    )
    ax1.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Panel 2: K x 39 Heatmap of P(phone | cluster) (%)
    ordered_pct_matrix = pct_matrix[sorted_c_ids, :]
    im = ax2.imshow(ordered_pct_matrix, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(50.0, float(np.max(ordered_pct_matrix))))

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"C{c['cluster_id']:0{cid_digits}d} ({c['dominant_phone']})" for c in sorted_clusters], fontsize=ytick_fs)
    ax2.set_xticks(np.arange(num_phones))
    ax2.set_xticklabels(unique_phone_names, fontsize=8, fontweight="bold", rotation=90)
    ax2.set_xlabel("Ground-Truth ARPAbet Phonemes (39 Classes)", fontsize=10, fontweight="bold")
    ax2.set_title(
        f"Cluster-to-Phoneme Conditional Percentage Matrix P(phone | cluster) (%)\n"
        f"(Covered Dominant Phones: {len(unique_dominant_phones)}/{num_phones} | Mean Effective Perplexity: {mean_eff_phones:.2f} phones)",
        fontsize=10.5,
        fontweight="bold",
    )

    # Annotate heatmap cells >= min_hm_val_for_label
    for r_i in range(num_clusters):
        for col_j in range(num_phones):
            val = ordered_pct_matrix[r_i, col_j]
            if val >= min_hm_val_for_label:
                ax2.text(
                    col_j,
                    r_i,
                    f"{val:.0f}",
                    ha="center",
                    va="center",
                    fontsize=hm_text_fs,
                    fontweight="bold" if val >= 15.0 else "normal",
                    color="white" if val >= 28.0 else "black",
                )

    cbar = fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02)
    cbar.set_label("Percentage of Cluster Segments (%)", fontsize=9, fontweight="bold")

    fig.suptitle(
        f"Test 4: K={num_clusters} Cluster Ground-Truth Phoneme Purity & Composition Analysis",
        fontsize=13,
        fontweight="bold",
        y=0.995 if num_clusters <= 50 else 0.998,
    )
    fig.tight_layout()
    plot_path = os.path.join(output_dir, "cluster_phone_composition.png")
    fig.savefig(plot_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig)

    logging.info(
        f"Test 4 Complete: Weighted Purity={weighted_purity:.2f}%, "
        f"Mean Unique Phones/Cluster={mean_unique_phones:.1f} (>=5%: {mean_ge_5pct:.1f}), "
        f"Mean Effective Perplexity={mean_eff_phones:.2f}"
    )

    # -------------------------------------------------------------------------
    # Test 4b: Extension with Explicit Word Boundary Delimiters (<w>)
    # (As in v5.9.1_spin_audio_word_boundaries_base40_unit_sphere_gan_pca512)
    # -------------------------------------------------------------------------
    wb_phone_labels = []
    is_before_wb_list = []
    is_after_wb_list = []
    consec_collisions_no_wb = np.zeros(num_clusters, dtype=np.int64)
    consec_collisions_with_wb = np.zeros(num_clusters, dtype=np.int64)
    total_word_boundaries_inserted = 0

    seg_offset = 0
    for u_idx, labs in enumerate(labels_by_utt):
        n_s = len(labs)
        u_cids = cluster_assignments[seg_offset : seg_offset + n_s]
        seg_offset += n_s

        wb_set = set()
        if word_boundaries_by_utt is not None and u_idx < len(word_boundaries_by_utt) and word_boundaries_by_utt[u_idx]:
            wb_set = set(word_boundaries_by_utt[u_idx])
        else:
            for s_i, l_val in enumerate(labs):
                if int(l_val) >= 40 and s_i < n_s - 1:
                    wb_set.add(s_i)

        total_word_boundaries_inserted += len(wb_set)

        for s_i in range(n_s):
            base_p = idx_to_phone.get(int(labs[s_i]), f"P{int(labs[s_i])}")
            before_wb = (s_i in wb_set) or (int(labs[s_i]) >= 40) or (s_i == n_s - 1)
            after_wb = (s_i == 0) or ((s_i - 1) in wb_set) or (int(labs[s_i - 1]) >= 40)
            is_before_wb_list.append(before_wb)
            is_after_wb_list.append(after_wb)

            if before_wb:
                wb_phone_labels.append(f"{base_p}_{word_boundary_token}")
            else:
                wb_phone_labels.append(base_p)

            if s_i < n_s - 1:
                p_cur = idx_to_phone.get(int(labs[s_i]), f"P{int(labs[s_i])}")
                p_nxt = idx_to_phone.get(int(labs[s_i + 1]), f"P{int(labs[s_i + 1])}")
                if p_cur != p_nxt and int(u_cids[s_i]) == int(u_cids[s_i + 1]):
                    c_val = int(u_cids[s_i])
                    consec_collisions_no_wb[c_val] += 1
                    # With <w> delimiter inserted after word-final segment s_i,
                    # c_t and c_{t+1} are separated by <w> (c_t <w> c_{t+1}) and do not collide!
                    if s_i not in wb_set and int(labs[s_i]) < 40:
                        consec_collisions_with_wb[c_val] += 1

    unique_wb_phones = sorted(list(set(wb_phone_labels)))
    wb_phone_to_col = {p: i for i, p in enumerate(unique_wb_phones)}
    num_wb_phones = len(unique_wb_phones)

    wb_counts_matrix = np.zeros((num_clusters, num_wb_phones), dtype=np.int64)
    before_wb_counts = np.zeros(num_clusters, dtype=np.int64)
    after_wb_counts = np.zeros(num_clusters, dtype=np.int64)

    for c_idx, wb_p, bef_wb, aft_wb in zip(cluster_assignments, wb_phone_labels, is_before_wb_list, is_after_wb_list):
        c_int = int(c_idx)
        wb_counts_matrix[c_int, wb_phone_to_col[wb_p]] += 1
        if bef_wb:
            before_wb_counts[c_int] += 1
        if aft_wb:
            after_wb_counts[c_int] += 1

    wb_pct_matrix = np.zeros((num_clusters, num_wb_phones), dtype=np.float64)
    for c in range(num_clusters):
        if cluster_totals[c] > 0:
            wb_pct_matrix[c, :] = (wb_counts_matrix[c, :] / float(cluster_totals[c])) * 100.0

    wb_clusters_report = {}
    wb_per_cluster_list = []

    for c in range(num_clusters):
        c_tot = int(cluster_totals[c])
        c_counts = wb_counts_matrix[c]
        c_pcts = wb_pct_matrix[c]

        nz_idx = [i for i in np.argsort(-c_counts) if c_counts[i] > 0]
        n_uniq_wb = len(nz_idx)
        n_wb_1pct = int(np.sum(c_pcts >= 1.0))
        n_wb_5pct = int(np.sum(c_pcts >= 5.0))

        if c_tot > 0:
            probs = c_counts[nz_idx] / float(c_tot)
            ent_wb = float(-np.sum(probs * np.log2(np.maximum(probs, 1e-12))))
            eff_wb = float(2.0 ** ent_wb)
            pct_bef_wb = float(100.0 * before_wb_counts[c] / c_tot)
            pct_aft_wb = float(100.0 * after_wb_counts[c] / c_tot)
        else:
            ent_wb, eff_wb, pct_bef_wb, pct_aft_wb = 0.0, 0.0, 0.0, 0.0

        dom_wb_phone = unique_wb_phones[nz_idx[0]] if nz_idx else "NONE"
        dom_wb_pct = float(c_pcts[nz_idx[0]]) if nz_idx else 0.0

        wb_dist = {}
        top_5_wb = []
        for r_i, p_i in enumerate(nz_idx):
            p_str = unique_wb_phones[p_i]
            wb_dist[p_str] = {
                "count": int(c_counts[p_i]),
                "percentage": round(float(c_pcts[p_i]), 4),
            }
            if r_i < 5:
                top_5_wb.append({
                    "phone_with_wb": p_str,
                    "count": int(c_counts[p_i]),
                    "percentage": round(float(c_pcts[p_i]), 2),
                })

        wb_c_info = {
            "cluster_id": c,
            "total_segments": c_tot,
            "num_unique_wb_phones_contained": n_uniq_wb,
            "num_wb_phones_ge_1pct": n_wb_1pct,
            "num_wb_phones_ge_5pct": n_wb_5pct,
            "entropy_bits": round(ent_wb, 4),
            "effective_num_wb_phones_perplexity": round(eff_wb, 4),
            "dominant_wb_phone": dom_wb_phone,
            "dominant_wb_phone_pct": round(dom_wb_pct, 4),
            "pct_segments_before_word_boundary": round(pct_bef_wb, 2),
            "pct_segments_after_word_boundary": round(pct_aft_wb, 2),
            "consecutive_collisions_without_wb": int(consec_collisions_no_wb[c]),
            "consecutive_collisions_with_wb": int(consec_collisions_with_wb[c]),
            "collisions_eliminated_by_wb": int(consec_collisions_no_wb[c] - consec_collisions_with_wb[c]),
            "top_5_wb_phones": top_5_wb,
            "wb_phone_distribution": wb_dist,
        }
        wb_clusters_report[f"cluster_{c}"] = wb_c_info
        wb_per_cluster_list.append(wb_c_info)

    wb_weighted_purity = float(
        np.sum(np.max(wb_counts_matrix, axis=1)) / max(np.sum(wb_counts_matrix), 1) * 100.0
    )
    mean_uniq_wb = float(np.mean([c["num_unique_wb_phones_contained"] for c in wb_per_cluster_list]))
    mean_wb_5pct = float(np.mean([c["num_wb_phones_ge_5pct"] for c in wb_per_cluster_list]))
    mean_wb_1pct = float(np.mean([c["num_wb_phones_ge_1pct"] for c in wb_per_cluster_list]))
    mean_eff_wb = float(np.mean([c["effective_num_wb_phones_perplexity"] for c in wb_per_cluster_list]))
    tot_coll_no_wb = int(np.sum(consec_collisions_no_wb))
    tot_coll_with_wb = int(np.sum(consec_collisions_with_wb))

    wb_summary = {
        "test": "test4_cluster_phone_composition_with_word_boundaries",
        "word_boundary_token": word_boundary_token,
        "num_clusters": num_clusters,
        "num_wb_delimited_phone_classes": num_wb_phones,
        "total_segments_evaluated": int(total_segments),
        "total_word_boundaries_inserted": int(total_word_boundaries_inserted),
        "overall_wb_weighted_purity_pct": round(wb_weighted_purity, 4),
        "mean_unique_wb_phones_per_cluster": round(mean_uniq_wb, 2),
        "mean_wb_phones_ge_1pct_per_cluster": round(mean_wb_1pct, 2),
        "mean_wb_phones_ge_5pct_per_cluster": round(mean_wb_5pct, 2),
        "mean_effective_wb_phones_perplexity": round(mean_eff_wb, 2),
        "total_consecutive_collisions_without_wb": tot_coll_no_wb,
        "total_consecutive_collisions_with_wb": tot_coll_with_wb,
        "total_collisions_eliminated_by_wb": tot_coll_no_wb - tot_coll_with_wb,
        "collision_reduction_relative_pct": round(
            100.0 * (tot_coll_no_wb - tot_coll_with_wb) / max(tot_coll_no_wb, 1), 2
        ),
        "clusters": wb_clusters_report,
    }
    summary_results["word_boundary_extension"] = wb_summary

    with open(json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    wb_json_path = os.path.join(output_dir, "cluster_phone_composition_word_boundaries.json")
    with open(wb_json_path, "w") as f:
        json.dump(wb_summary, f, indent=2)

    # Plot Word-Boundary Delimited Composition Figure
    fig_wb, (ax_wb1, ax_wb2) = plt.subplots(
        1, 2,
        figsize=(22, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.15]},
    )

    wb_bar_fs = 6.0 if num_clusters <= 50 else bar_text_fs
    wb_right_fs = 7.0 if num_clusters <= 50 else right_text_fs

    for row_idx, c_id in enumerate(sorted_c_ids):
        wb_c_info = wb_clusters_report[f"cluster_{c_id}"]
        c_pcts = wb_pct_matrix[c_id]
        sorted_p_indices = [i for i in np.argsort(-c_pcts) if c_pcts[i] > 0]

        shown_wb_indices = [
            p_idx for rank_i, p_idx in enumerate(sorted_p_indices[:top_n_stack])
            if rank_i == 0 or float(c_pcts[p_idx]) >= min_bar_val_for_label
        ]

        left_acc = 0.0
        for p_idx in shown_wb_indices:
            p_full_name = unique_wb_phones[p_idx]
            base_p_name = p_full_name.replace(f"_{word_boundary_token}", "")
            is_wb_end = p_full_name.endswith(f"_{word_boundary_token}")
            val = float(c_pcts[p_idx])
            color = phone_colors.get(base_p_name, "steelblue")
            ax_wb1.barh(
                row_idx,
                val,
                left=left_acc,
                height=0.78,
                color=color,
                edgecolor="black" if is_wb_end else "white",
                linewidth=(0.9 if is_wb_end else 0.5) if num_clusters <= 150 else (0.5 if is_wb_end else 0.25),
                hatch="//" if is_wb_end else None,
            )
            if val >= min_bar_val_for_label:
                if val >= 6.5:
                    cur_wb_fs = wb_bar_fs
                    wb_lbl = f"{p_full_name} {val:.0f}%" if single_line_bar_label else f"{p_full_name}\n{val:.0f}%"
                elif val >= 3.5:
                    cur_wb_fs = max(2.8, wb_bar_fs * 0.80)
                    wb_lbl = f"{p_full_name}" if single_line_bar_label else f"{p_full_name}\n{val:.0f}%"
                else:
                    cur_wb_fs = max(2.4, wb_bar_fs * 0.68)
                    wb_lbl = f"{p_full_name}"
                ax_wb1.text(
                    left_acc + val / 2.0,
                    row_idx,
                    wb_lbl,
                    ha="center",
                    va="center",
                    fontsize=cur_wb_fs,
                    fontweight="bold",
                    color="white" if val >= 12.0 else "black",
                )
            left_acc += val

        rem_val = max(0.0, 100.0 - left_acc)
        if rem_val > 0.05:
            rem_count = max(0, len(sorted_p_indices) - len(shown_wb_indices))
            ax_wb1.barh(
                row_idx,
                rem_val,
                left=left_acc,
                height=0.78,
                color="#d9d9d9",
                edgecolor="white",
                linewidth=0.5 if num_clusters <= 150 else 0.25,
            )
            if rem_val >= 4.0 and rem_count > 0:
                ax_wb1.text(
                    left_acc + rem_val / 2.0,
                    row_idx,
                    f"+{rem_count} ({rem_val:.0f}%)",
                    ha="center",
                    va="center",
                    fontsize=rem_text_fs if rem_val >= 8.0 else max(2.6, rem_text_fs * 0.8),
                    color="#333333",
                )

        n_tot_wb = wb_c_info["num_unique_wb_phones_contained"]
        n_5_wb = wb_c_info["num_wb_phones_ge_5pct"]
        pct_bef = wb_c_info["pct_segments_before_word_boundary"]
        ax_wb1.text(
            101.2,
            row_idx,
            f"{n_tot_wb} wb-phns (≥5%:{n_5_wb} | pre-{word_boundary_token}:{pct_bef:.0f}%)",
            ha="left",
            va="center",
            fontsize=wb_right_fs,
            family="monospace",
            color="#222222",
        )

    ax_wb1.set_yticks(y_pos)
    ax_wb1.set_yticklabels(
        [f"C{c['cluster_id']:0{cid_digits}d} (n={c['total_segments']})" for c in sorted_clusters],
        fontsize=ytick_fs,
        fontweight="bold",
    )
    ax_wb1.invert_yaxis()
    ax_wb1.set_xlim(0, 142)
    ax_wb1.set_xticks(np.arange(0, 101, 20))
    ax_wb1.set_xticklabels([f"{x}%" for x in range(0, 101, 20)])
    ax_wb1.set_xlabel(f"Ground-Truth Phone + Word Boundary ({word_boundary_token}) Percentage Breakdown (%)", fontsize=10, fontweight="bold")
    ax_wb1.set_title(
        f"Per-Cluster Composition with Word Boundary Delimiters ({word_boundary_token}, Hatched = Word-Final P_{word_boundary_token})\n"
        f"(WB-Purity: {wb_weighted_purity:.1f}% | Mean WB-Phones ≥5%: {mean_wb_5pct:.1f}, Total: {mean_uniq_wb:.1f})",
        fontsize=10.0,
        fontweight="bold",
    )
    ax_wb1.grid(True, axis="x", linestyle=":", alpha=0.5)

    # Panel 2: Per-Cluster Word Boundary Position Breakdown & Consecutive Collision Savings with <w>
    pct_before_arr = np.array([wb_clusters_report[f"cluster_{cid}"]["pct_segments_before_word_boundary"] for cid in sorted_c_ids])
    pct_after_arr = np.array([wb_clusters_report[f"cluster_{cid}"]["pct_segments_after_word_boundary"] for cid in sorted_c_ids])
    coll_no_wb_arr = np.array([wb_clusters_report[f"cluster_{cid}"]["consecutive_collisions_without_wb"] for cid in sorted_c_ids])
    coll_w_wb_arr = np.array([wb_clusters_report[f"cluster_{cid}"]["consecutive_collisions_with_wb"] for cid in sorted_c_ids])

    ax_wb2.barh(y_pos - 0.18, pct_before_arr, height=0.35, color="#d95f02", label=f"Word-Final (followed by {word_boundary_token}) (%)")
    ax_wb2.barh(y_pos + 0.18, pct_after_arr, height=0.35, color="#7570b3", label=f"Word-Initial (preceded by {word_boundary_token}) (%)")

    for r_i, (cid, c_nowb, c_wwb) in enumerate(zip(sorted_c_ids, coll_no_wb_arr, coll_w_wb_arr)):
        saved = c_nowb - c_wwb
        max_bar = max(pct_before_arr[r_i], pct_after_arr[r_i])
        ax_wb2.text(
            max_bar + 1.2,
            r_i,
            f"Collisions: {c_nowb} → {c_wwb} (-{saved} w/ {word_boundary_token})",
            ha="left",
            va="center",
            fontsize=wb_right_fs,
            family="monospace",
            color="#1b9e77" if saved > 0 else "#444444",
            fontweight="bold" if saved >= 20 else "normal",
        )

    ax_wb2.set_yticks(y_pos)
    ax_wb2.set_yticklabels([f"C{cid:0{cid_digits}d}" for cid in sorted_c_ids], fontsize=ytick_fs)
    ax_wb2.invert_yaxis()
    ax_wb2.set_xlim(0, max(105.0, float(np.max(np.maximum(pct_before_arr, pct_after_arr))) + 45.0))
    ax_wb2.set_xlabel(f"Word Boundary Incidence (%) & Collision Reduction with {word_boundary_token}", fontsize=10, fontweight="bold")
    ax_wb2.set_title(
        f"Word-Boundary Position Share & Collision Elimination via {word_boundary_token}\n"
        f"(Collisions: {tot_coll_no_wb:,} → {tot_coll_with_wb:,} | Eliminated: {tot_coll_no_wb - tot_coll_with_wb:,} [-{wb_summary['collision_reduction_relative_pct']:.1f}%])",
        fontsize=10.0,
        fontweight="bold",
    )
    ax_wb2.legend(loc="lower right", fontsize=8.5)
    ax_wb2.grid(True, axis="x", linestyle=":", alpha=0.5)

    fig_wb.suptitle(
        f"Test 4 (Word Boundary Extension): K={num_clusters} Cluster Phone Composition & Collision Savings with '{word_boundary_token}' Delimiters",
        fontsize=13,
        fontweight="bold",
        y=0.995 if num_clusters <= 50 else 0.998,
    )
    fig_wb.tight_layout()
    wb_plot_path = os.path.join(output_dir, "cluster_phone_composition_word_boundaries.png")
    fig_wb.savefig(wb_plot_path, dpi=save_dpi, bbox_inches="tight")
    plt.close(fig_wb)

    return summary_results


# ==============================================================================
# MAIN TEST RUNNER
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Run Representation Tests Suite")
    parser.add_argument("--model", type=str, required=True, choices=["hubert", "spinhubert", "wav2vec2.0", "all"], help="Model backbone to test")
    parser.add_argument("--output-base", type=str, default="output/unsup_audio_cluster_to_phoneme/librispeech/cross_modal_cheat_seg_curium/representation_tests")
    parser.add_argument("--num-utts", type=int, default=500, help="Number of utterances to evaluate")
    parser.add_argument("--pca-dim", type=int, default=512, help="PCA projection dimension")
    args = parser.parse_args()

    models = ["hubert", "spinhubert", "wav2vec2.0"] if args.model == "all" else [args.model]

    for m in models:
        m_dir = os.path.join(args.output_base, m)
        logging.info(f"Setting up suite for model: {m} in {m_dir}")
        t1_dir = os.path.join(m_dir, "test1_consecutive_distance")
        t2_dir = os.path.join(m_dir, "test2_supervised_discrimination")
        t3_dir = os.path.join(m_dir, "test3_kmeans_collision")

        os.makedirs(t1_dir, exist_ok=True)
        os.makedirs(t2_dir, exist_ok=True)
        os.makedirs(t3_dir, exist_ok=True)

    print(f"Representation test suite configured in: {args.output_base}")

if __name__ == "__main__":
    main()
