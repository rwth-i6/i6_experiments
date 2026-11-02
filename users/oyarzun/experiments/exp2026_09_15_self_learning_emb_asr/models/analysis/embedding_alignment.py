"""
Standalone Analysis Jobs for Cross-Modal Self-Learning Embedding Alignment (Artetxe et al., 2017).
Computes:
1. Phoneme Coverage & Distribution Entropy (Metric 1A)
2. Joint Cross-Modal 2D PCA & Projection Visualization (Metric 3A)
3. Individual Modality 2D PCA with Frequency Color Gradient & Callouts
4. Unified 3-Panel Multi-Modal Representation Comparison (Joint + Source + Target)
"""

from typing import Dict, List, Optional, Tuple, Iterator, Union
import os
import json
import logging
import numpy as np

from sisyphus import Job, Task, tk
from ..self_learning_alignment import normalize_and_center


def compute_phoneme_coverage_and_entropy(
    cluster_to_phoneme_map: list,
    num_phonemes: int,
    phoneme_vocab: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Computes Phoneme Coverage and Distribution Entropy (Metric 1A).

    :param cluster_to_phoneme_map: List of length K with target phoneme indices.
    :param num_phonemes: Total number of phonemes P in the target vocabulary.
    :param phoneme_vocab: Optional mapping from phoneme index to phoneme token name.
    :return: Dictionary containing coverage ratio, Shannon entropy, normalized entropy, and counts.
    """
    K = len(cluster_to_phoneme_map)
    P = num_phonemes
    counts = np.bincount(cluster_to_phoneme_map, minlength=P)

    unique_covered = int(np.sum(counts > 0))
    coverage_ratio = float(unique_covered / P) if P > 0 else 0.0

    probs = counts / K
    nonzero_probs = probs[probs > 0]
    entropy = float(-np.sum(nonzero_probs * np.log2(nonzero_probs)))
    max_entropy = float(np.log2(min(K, P))) if min(K, P) > 1 else 1.0
    normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

    per_phoneme_counts = {}
    for idx, c in enumerate(counts):
        p_name = phoneme_vocab.get(idx, f"phoneme_{idx}") if phoneme_vocab else f"phoneme_{idx}"
        per_phoneme_counts[p_name] = int(c)

    return {
        "num_clusters": K,
        "num_phonemes": P,
        "unique_phonemes_covered": unique_covered,
        "coverage_ratio": coverage_ratio,
        "entropy": entropy,
        "normalized_entropy": normalized_entropy,
        "max_theoretical_entropy": max_entropy,
        "phoneme_assignment_counts": per_phoneme_counts,
    }


def compute_joint_cross_modal_pca(
    mapped_X: np.ndarray,
    Z: np.ndarray,
    audio_vocab: Optional[Dict[int, str]] = None,
    text_vocab: Optional[Dict[int, str]] = None,
    cluster_to_phoneme_map: Optional[list] = None,
    audio_counts: Optional[Dict[int, float]] = None,
    text_counts: Optional[Dict[int, float]] = None,
) -> Dict:
    """
    Joint 2D Principal Component Analysis (Metric 3A).
    Stacks mapped audio clusters X W* (K x d) and target phonemes Z (P x d)
    to project both modalities into a shared 2D subspace.
    """
    K, d_x = mapped_X.shape
    P, d_z = Z.shape
    assert d_x == d_z, f"Dimension mismatch: {d_x} vs {d_z}"

    joint_mat = np.vstack([mapped_X, Z]).astype(np.float64)  # (K+P, d)
    mean_joint = np.mean(joint_mat, axis=0, keepdims=True)
    centered = joint_mat - mean_joint

    # SVD for PCA components
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T  # (d, 2)

    total_var = float(np.sum(s ** 2))
    var_pc1 = float((s[0] ** 2) / total_var) if total_var > 0 else 0.0
    var_pc2 = float((s[1] ** 2) / total_var) if total_var > 0 else 0.0
    evr_2d = float(var_pc1 + var_pc2)

    # 2D Projections
    proj_audio = ((mapped_X - mean_joint) @ components).tolist()  # K x 2
    proj_text = ((Z - mean_joint) @ components).tolist()          # P x 2

    audio_items = []
    for idx, (px, py) in enumerate(proj_audio):
        a_name = audio_vocab.get(idx, f"clus_{idx}") if audio_vocab else f"clus_{idx}"
        assigned_p = cluster_to_phoneme_map[idx] if cluster_to_phoneme_map else None
        p_name = text_vocab.get(assigned_p, f"phoneme_{assigned_p}") if (text_vocab and assigned_p is not None) else None
        cnt = float(audio_counts.get(idx, 1.0)) if audio_counts is not None else 1.0
        audio_items.append({
            "index": idx,
            "label": a_name,
            "x": float(px),
            "y": float(py),
            "count": cnt,
            "assigned_phoneme_index": assigned_p,
            "assigned_phoneme_label": p_name,
        })

    text_items = []
    for idx, (px, py) in enumerate(proj_text):
        t_name = text_vocab.get(idx, f"phoneme_{idx}") if text_vocab else f"phoneme_{idx}"
        cnt = float(text_counts.get(idx, 1.0)) if text_counts is not None else 1.0
        text_items.append({
            "index": idx,
            "label": t_name,
            "x": float(px),
            "y": float(py),
            "count": cnt,
        })

    return {
        "explained_variance_ratio_pc1": var_pc1,
        "explained_variance_ratio_pc2": var_pc2,
        "total_2d_explained_variance_ratio": evr_2d,
        "audio_clusters_2d": audio_items,
        "text_phonemes_2d": text_items,
    }


def compute_single_modality_pca(
    X: np.ndarray,
    vocab: Optional[Dict[int, str]] = None,
    counts: Optional[Dict[int, float]] = None,
    prefix_label: str = "item",
) -> Dict:
    """
    Computes standard 2D PCA on a single embedding matrix X (N x d).
    Returns 2D projected coordinates, explained variance, and per-item metadata (counts, labels).
    """
    N, d = X.shape
    X_f = X.astype(np.float64)
    mean_X = np.mean(X_f, axis=0, keepdims=True)
    centered = X_f - mean_X

    # SVD for PCA components
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T  # (d, 2)

    total_var = float(np.sum(s ** 2))
    var_pc1 = float((s[0] ** 2) / total_var) if total_var > 0 else 0.0
    var_pc2 = float((s[1] ** 2) / total_var) if total_var > 0 else 0.0
    evr_2d = float(var_pc1 + var_pc2)

    proj = (centered @ components).tolist()

    items = []
    for idx, (px, py) in enumerate(proj):
        name = vocab.get(idx, f"{prefix_label}_{idx}") if vocab else f"{prefix_label}_{idx}"
        cnt = float(counts.get(idx, 1.0)) if counts is not None else 1.0
        items.append({
            "index": idx,
            "label": name,
            "x": float(px),
            "y": float(py),
            "count": cnt,
        })

    return {
        "explained_variance_ratio_pc1": var_pc1,
        "explained_variance_ratio_pc2": var_pc2,
        "total_2d_explained_variance_ratio": evr_2d,
        "items_2d": items,
    }


def format_count_tick(val, pos):
    orig = 10 ** val - 1.0
    if orig >= 1e6:
        return f"{orig/1e6:.1f}M"
    elif orig >= 1e3:
        return f"{orig/1e3:.0f}k"
    elif orig >= 1:
        return f"{int(orig)}"
    return "0"


def plot_joint_cross_modal_pca(
    pca_result: Dict,
    out_png_path: str,
    title: str = "Joint Cross-Modal 2D PCA: Audio Clusters (X W*) & Phonemes (Z)",
    src_label: str = "Audio Clusters X W*",
    trg_label: str = "Text Units Z",
):
    """
    Renders and saves an enhanced 2D scatter visualization connecting the two modalities/languages
    with adaptive point sizes, dual-modality KDE density contours, and selective anchor lines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)

    # Audio / Source points
    a_x = np.array([p["x"] for p in pca_result["audio_clusters_2d"]])
    a_y = np.array([p["y"] for p in pca_result["audio_clusters_2d"]])

    # Text / Target points
    t_x = np.array([p["x"] for p in pca_result["text_phonemes_2d"]])
    t_y = np.array([p["y"] for p in pca_result["text_phonemes_2d"]])
    t_labels = [p["label"] for p in pca_result["text_phonemes_2d"]]

    # 1. Coordinate ranges
    all_x = np.concatenate([a_x, t_x]) if len(a_x) and len(t_x) else (a_x if len(a_x) else t_x)
    all_y = np.concatenate([a_y, t_y]) if len(a_y) and len(t_y) else (a_y if len(a_y) else t_y)
    x_pad = (all_x.max() - all_x.min()) * 0.05 if len(all_x) else 1.0
    y_pad = (all_y.max() - all_y.min()) * 0.05 if len(all_y) else 1.0
    x_range = (all_x.min() - x_pad, all_x.max() + x_pad)
    y_range = (all_y.min() - y_pad, all_y.max() + y_pad)

    # 2. Overlay KDE density contours for both modalities
    if len(a_x) >= 20:
        Ha, x_edges, y_edges = np.histogram2d(a_x, a_y, bins=100, range=[x_range, y_range])
        Ha_smooth = gaussian_filter(Ha.T, sigma=1.8)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(xc, yc, Ha_smooth, levels=5, colors="#1b4975", alpha=0.55, linewidths=1.0, zorder=2)

    if len(t_x) >= 20:
        Ht, _, _ = np.histogram2d(t_x, t_y, bins=100, range=[x_range, y_range])
        Ht_smooth = gaussian_filter(Ht.T, sigma=1.8)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(xc, yc, Ht_smooth, levels=5, colors="#b32d29", alpha=0.55, linewidths=1.0, zorder=2)

    # 3. Connections between top anchor correspondences (avoid dense hairball)
    a_items = pca_result["audio_clusters_2d"]
    t_items = pca_result["text_phonemes_2d"]
    matched_pairs = []
    for idx, a_item in enumerate(a_items):
        target_idx = a_item.get("assigned_phoneme_index")
        if target_idx is not None and 0 <= target_idx < len(t_items):
            score = a_item.get("count", 1.0)
            matched_pairs.append((score, a_item, t_items[target_idx]))

    matched_pairs.sort(key=lambda x: x[0], reverse=True)
    top_connections = matched_pairs[:45] if len(matched_pairs) > 60 else matched_pairs
    for _, a_item, target in top_connections:
        ax.plot(
            [a_item["x"], target["x"]],
            [a_item["y"], target["y"]],
            color="#555555",
            alpha=0.45,
            linestyle="--",
            linewidth=0.85,
            zorder=3,
        )

    # 4. Scatter points with adaptive sizing
    a_size = 12 if len(a_x) > 1000 else (30 if len(a_x) > 100 else 60)
    t_size = 14 if len(t_x) > 1000 else (35 if len(t_x) > 100 else 75)
    a_alpha = 0.45 if len(a_x) > 1000 else 0.75
    t_alpha = 0.50 if len(t_x) > 1000 else 0.85

    ax.scatter(
        a_x, a_y,
        color="#2b5c8f",
        marker="o",
        s=a_size,
        alpha=a_alpha,
        edgecolors="none",
        label=f"{src_label} (N={len(a_x)})",
        zorder=4,
    )

    ax.scatter(
        t_x, t_y,
        color="#d9534f",
        marker="s",
        s=t_size,
        alpha=t_alpha,
        edgecolors="#7a1816" if len(t_x) <= 100 else "none",
        linewidth=0.6,
        label=f"{trg_label} (N={len(t_x)})",
        zorder=5,
    )

    # 5. Text labels
    labeled_indices = set()
    if len(t_labels) <= 60:
        for px, py, lab in zip(t_x, t_y, t_labels):
            ax.annotate(
                lab,
                (px, py),
                fontsize=8,
                fontweight="bold",
                color="#4a0e0c",
                xytext=(3, 3),
                textcoords="offset points",
                zorder=6,
            )
    else:
        for _, _, target in top_connections[:30]:
            t_idx = target.get("index")
            if t_idx is not None and t_idx not in labeled_indices:
                labeled_indices.add(t_idx)
                ax.annotate(
                    target["label"],
                    (target["x"], target["y"]),
                    fontsize=7.5,
                    fontweight="bold",
                    color="#4a0e0c",
                    xytext=(3, 3),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.75, edgecolor="none"),
                    zorder=6,
                )

    evr = pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
    ax.set_title(f"{title}\n(Total 2D Explained Variance: {evr:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({pca_result.get('explained_variance_ratio_pc1', 0.0)*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC 2 ({pca_result.get('explained_variance_ratio_pc2', 0.0)*100:.1f}% var)", fontsize=10)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def plot_single_modality_pca(
    pca_result: Dict,
    out_png_path: str,
    modality_key: str = "audio_clusters_2d",
    title: str = "Modality PCA with Frequency Gradient",
    label: str = "Tokens",
    cmap: str = "viridis",
    x_range: Optional[Tuple[float, float]] = None,
    y_range: Optional[Tuple[float, float]] = None,
):
    """
    Renders an individual modality's 2D PCA distribution where scatter points are
    colored according to log10(frequency), with an explicit colorbar and callout annotations
    for the most frequent items.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy.ndimage import gaussian_filter

    fig, ax = plt.subplots(figsize=(10.5, 8.5), dpi=200)

    items = pca_result.get(modality_key, [])
    if not items and "items_2d" in pca_result:
        items = pca_result["items_2d"]
    if not items:
        logging.warning(f"No items found for {modality_key}")
        plt.close(fig)
        return

    x = np.array([p["x"] for p in items])
    y = np.array([p["y"] for p in items])
    c = np.array([p.get("count", 1.0) for p in items])
    labels = [p["label"] for p in items]

    # Log10-scale counts for smooth, perceptually uniform frequency gradient
    log_c = np.log10(np.maximum(c, 1.0))

    # Sort so that high-frequency points are plotted on top of low-frequency points
    order = np.argsort(c)
    x_s, y_s, c_s, log_c_s = x[order], y[order], c[order], log_c[order]

    # Adaptive point size based on frequency
    c_min, c_max = log_c.min(), log_c.max()
    norm_c = (log_c_s - c_min) / (c_max - c_min + 1e-6)
    sizes = 14 + norm_c * 26  # from 14 to 40
    alphas = 0.55 + norm_c * 0.35  # from 0.55 to 0.90

    sc = ax.scatter(
        x_s, y_s,
        c=log_c_s,
        cmap=cmap,
        s=sizes,
        alpha=alphas,
        edgecolors="none",
        zorder=3,
    )

    # Density contour overlay
    if len(x) >= 20:
        H, x_edges, y_edges = np.histogram2d(x, y, bins=100)
        H_smooth = gaussian_filter(H.T, sigma=2.0)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax.contour(xc, yc, H_smooth, levels=6, colors="#444444", alpha=0.35, linewidths=0.8, zorder=2)

    # Colorbar with formatted count ticks
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.formatter = FuncFormatter(format_count_tick)
    cbar.update_ticks()
    cbar.set_label("Element Frequency / Count (log scale)", fontsize=10, fontweight="bold")

    # Annotate top 25-30 most frequent items
    top_indices = order[-30:][::-1]
    for idx in top_indices:
        lab = labels[idx]
        px, py = x[idx], y[idx]
        if lab in ("<pad>", "<unk>", "<s>", "</s>") and len(top_indices) > 5:
            continue
        ax.annotate(
            lab,
            (px, py),
            fontsize=7.5,
            fontweight="bold",
            color="#111111",
            xytext=(3, 3),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="#888888", linewidth=0.5),
            zorder=5,
        )

    evr = pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
    ax.set_title(f"{title} (N={len(items)})\n(Explained Variance: {evr:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({pca_result.get('explained_variance_ratio_pc1', 0.0)*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC 2 ({pca_result.get('explained_variance_ratio_pc2', 0.0)*100:.1f}% var)", fontsize=10)
    if x_range is not None:
        ax.set_xlim(x_range)
    if y_range is not None:
        ax.set_ylim(y_range)
    ax.grid(True, linestyle=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def plot_cross_modal_panel_pca(
    pca_result: Dict,
    out_png_path: str,
    title_prefix: str = "Cross-Modal 2D PCA",
    src_label: str = "Audio Clusters",
    trg_label: str = "Text Units",
    src_cmap: str = "viridis",
    trg_cmap: str = "plasma",
):
    """
    Renders a unified 3-panel figure showing:
      Panel 1: Joint Modality Overlay (dual density contours & anchor alignments)
      Panel 2: Source Modality alone with continuous frequency color gradient
      Panel 3: Target Modality alone with continuous frequency color gradient
    All 3 panels share identical X and Y axis limits for 1:1 spatial comparison.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy.ndimage import gaussian_filter

    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5), dpi=180)

    a_items = pca_result.get("audio_clusters_2d", [])
    t_items = pca_result.get("text_phonemes_2d", [])

    a_x = np.array([p["x"] for p in a_items])
    a_y = np.array([p["y"] for p in a_items])
    a_counts = np.array([p.get("count", 1.0) for p in a_items])
    a_log_c = np.log10(np.maximum(a_counts, 1.0))

    t_x = np.array([p["x"] for p in t_items])
    t_y = np.array([p["y"] for p in t_items])
    t_counts = np.array([p.get("count", 1.0) for p in t_items])
    t_log_c = np.log10(np.maximum(t_counts, 1.0))

    # Compute shared axis limits
    all_x = np.concatenate([a_x, t_x]) if len(a_x) and len(t_x) else (a_x if len(a_x) else t_x)
    all_y = np.concatenate([a_y, t_y]) if len(a_y) and len(t_y) else (a_y if len(a_y) else t_y)
    x_pad = (all_x.max() - all_x.min()) * 0.05 if len(all_x) else 1.0
    y_pad = (all_y.max() - all_y.min()) * 0.05 if len(all_y) else 1.0
    x_range = (all_x.min() - x_pad, all_x.max() + x_pad)
    y_range = (all_y.min() - y_pad, all_y.max() + y_pad)

    # ---------------- Panel 1: Joint Overlay ----------------
    ax1 = axes[0]
    if len(a_x) >= 20:
        Ha, x_edges, y_edges = np.histogram2d(a_x, a_y, bins=100, range=[x_range, y_range])
        Ha_smooth = gaussian_filter(Ha.T, sigma=1.8)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax1.contour(xc, yc, Ha_smooth, levels=5, colors="#1b4975", alpha=0.55, linewidths=1.0, zorder=2)

    if len(t_x) >= 20:
        Ht, _, _ = np.histogram2d(t_x, t_y, bins=100, range=[x_range, y_range])
        Ht_smooth = gaussian_filter(Ht.T, sigma=1.8)
        xc = 0.5 * (x_edges[:-1] + x_edges[1:])
        yc = 0.5 * (y_edges[:-1] + y_edges[1:])
        ax1.contour(xc, yc, Ht_smooth, levels=5, colors="#b32d29", alpha=0.55, linewidths=1.0, zorder=2)

    # Anchor connections
    matched_pairs = []
    for idx, a_item in enumerate(a_items):
        target_idx = a_item.get("assigned_phoneme_index")
        if target_idx is not None and 0 <= target_idx < len(t_items):
            score = a_item.get("count", 1.0)
            matched_pairs.append((score, a_item, t_items[target_idx]))
    matched_pairs.sort(key=lambda x: x[0], reverse=True)
    top_connections = matched_pairs[:45] if len(matched_pairs) > 60 else matched_pairs
    for _, a_item, target in top_connections:
        ax1.plot(
            [a_item["x"], target["x"]],
            [a_item["y"], target["y"]],
            color="#555555",
            alpha=0.45,
            linestyle="--",
            linewidth=0.85,
            zorder=3,
        )

    a_size = 12 if len(a_x) > 1000 else 35
    t_size = 14 if len(t_x) > 1000 else 40
    ax1.scatter(a_x, a_y, color="#2b5c8f", marker="o", s=a_size, alpha=0.5, edgecolors="none", label=f"{src_label} (N={len(a_x)})", zorder=4)
    ax1.scatter(t_x, t_y, color="#d9534f", marker="s", s=t_size, alpha=0.55, edgecolors="none", label=f"{trg_label} (N={len(t_x)})", zorder=5)

    ax1.set_title("Joint Modality Overlay (KDE Contours & Anchors)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("PC 1", fontsize=10)
    ax1.set_ylabel("PC 2", fontsize=10)
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)

    # ---------------- Panel 2: Source Modality Frequency ----------------
    ax2 = axes[1]
    order_a = np.argsort(a_counts)
    norm_a = (a_log_c[order_a] - a_log_c.min()) / (a_log_c.max() - a_log_c.min() + 1e-6)
    sizes_a = 12 + norm_a * 25
    sc_a = ax2.scatter(
        a_x[order_a], a_y[order_a],
        c=a_log_c[order_a], cmap=src_cmap,
        s=sizes_a, alpha=0.65, edgecolors="none", zorder=3
    )
    if len(a_x) >= 20:
        ax2.contour(xc, yc, Ha_smooth, levels=5, colors="#444444", alpha=0.35, linewidths=0.75, zorder=2)
    cbar_a = fig.colorbar(sc_a, ax=ax2, fraction=0.046, pad=0.04)
    cbar_a.formatter = FuncFormatter(format_count_tick)
    cbar_a.update_ticks()
    cbar_a.set_label("Element Count (log scale)", fontsize=9)

    # Annotations for top source tokens
    for idx in order_a[-20:][::-1]:
        lab = a_items[idx]["label"]
        if lab not in ("<pad>", "<unk>", "<s>", "</s>"):
            ax2.annotate(
                lab, (a_x[idx], a_y[idx]),
                fontsize=7, fontweight="bold", color="#111111",
                xytext=(3, 3), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="#888888", linewidth=0.5),
                zorder=5,
            )

    ax2.set_title(f"{src_label} (Frequency Gradient)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("PC 1", fontsize=10)
    ax2.set_ylabel("PC 2", fontsize=10)
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # ---------------- Panel 3: Target Modality Frequency ----------------
    ax3 = axes[2]
    order_t = np.argsort(t_counts)
    norm_t = (t_log_c[order_t] - t_log_c.min()) / (t_log_c.max() - t_log_c.min() + 1e-6)
    sizes_t = 12 + norm_t * 25
    sc_t = ax3.scatter(
        t_x[order_t], t_y[order_t],
        c=t_log_c[order_t], cmap=trg_cmap,
        s=sizes_t, alpha=0.65, edgecolors="none", zorder=3
    )
    if len(t_x) >= 20:
        ax3.contour(xc, yc, Ht_smooth, levels=5, colors="#444444", alpha=0.35, linewidths=0.75, zorder=2)
    cbar_t = fig.colorbar(sc_t, ax=ax3, fraction=0.046, pad=0.04)
    cbar_t.formatter = FuncFormatter(format_count_tick)
    cbar_t.update_ticks()
    cbar_t.set_label("Token Count (log scale)", fontsize=9)

    # Annotations for top target tokens
    for idx in order_t[-20:][::-1]:
        lab = t_items[idx]["label"]
        if lab not in ("<pad>", "<unk>", "<s>", "</s>"):
            ax3.annotate(
                lab, (t_x[idx], t_y[idx]),
                fontsize=7, fontweight="bold", color="#111111",
                xytext=(3, 3), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="#888888", linewidth=0.5),
                zorder=5,
            )

    ax3.set_title(f"{trg_label} (Frequency Gradient)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("PC 1", fontsize=10)
    ax3.set_ylabel("PC 2", fontsize=10)
    ax3.set_xlim(x_range)
    ax3.set_ylim(y_range)
    ax3.grid(True, linestyle=":", alpha=0.5)

    evr = pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
    fig.suptitle(f"{title_prefix} — Representation & Alignment Space (Total 2D Variance: {evr:.1f}%)", fontsize=13, fontweight="bold", y=0.98)

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def plot_alignment_comparison_panel(
    unaligned_pca_result: Dict,
    aligned_pca_result: Dict,
    source_raw_pca: Dict,
    target_raw_pca: Dict,
    out_png_path: str,
    title_prefix: str = "Embedding Alignment Analysis",
    src_label: str = "Audio Clusters",
    trg_label: str = "Text Units",
    src_cmap: str = "viridis",
    trg_cmap: str = "plasma",
):
    """
    Renders a 2x2 multi-panel figure comparing representations Before vs. After Alignment:
      Top-Left (ax1): Before Alignment — Joint PCA [X_raw, Z_raw] showing disjoint spaces
      Top-Right (ax2): After Alignment — Joint PCA [X W*, Z] showing aligned shared space with KDE contours
      Bottom-Left (ax3): Source Raw Word2Vec PCA with log10(frequency) gradient
      Bottom-Right (ax4): Target Raw Word2Vec PCA with log10(frequency) gradient
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    from scipy.ndimage import gaussian_filter

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 17), dpi=180)

    # -------------------------------------------------------------
    # Top-Left: Unaligned Joint PCA
    # -------------------------------------------------------------
    u_a_items = unaligned_pca_result.get("audio_clusters_2d", [])
    u_t_items = unaligned_pca_result.get("text_phonemes_2d", [])
    if u_a_items and u_t_items:
        u_ax = np.array([p["x"] for p in u_a_items])
        u_ay = np.array([p["y"] for p in u_a_items])
        u_tx = np.array([p["x"] for p in u_t_items])
        u_ty = np.array([p["y"] for p in u_t_items])

        ax1.scatter(u_ax, u_ay, color="#2b5c8f", marker="o", s=18, alpha=0.55, label=f"{src_label} Raw (N={len(u_ax)})", zorder=3)
        ax1.scatter(u_tx, u_ty, color="#d9534f", marker="s", s=22, alpha=0.65, label=f"{trg_label} Raw (N={len(u_tx)})", zorder=4)

        u_evr = unaligned_pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
        ax1.set_title(f"1. Before Alignment: Joint Space\n(Total 2D Var: {u_evr:.1f}% — Disjoint Subspaces)", fontsize=11, fontweight="bold")
        ax1.set_xlabel(f"PC 1 ({unaligned_pca_result.get('explained_variance_ratio_pc1', 0.0)*100:.1f}%)", fontsize=9.5)
        ax1.set_ylabel(f"PC 2 ({unaligned_pca_result.get('explained_variance_ratio_pc2', 0.0)*100:.1f}%)", fontsize=9.5)
        ax1.grid(True, linestyle=":", alpha=0.5)
        ax1.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # Top-Right: Aligned Joint PCA
    # -------------------------------------------------------------
    a_items = aligned_pca_result.get("audio_clusters_2d", [])
    t_items = aligned_pca_result.get("text_phonemes_2d", [])
    if a_items and t_items:
        ax_pts = np.array([p["x"] for p in a_items])
        ay_pts = np.array([p["y"] for p in a_items])
        tx_pts = np.array([p["x"] for p in t_items])
        ty_pts = np.array([p["y"] for p in t_items])

        # Coordinate ranges & KDE
        all_x = np.concatenate([ax_pts, tx_pts])
        all_y = np.concatenate([ay_pts, ty_pts])
        x_pad = (all_x.max() - all_x.min()) * 0.05 if len(all_x) else 1.0
        y_pad = (all_y.max() - all_y.min()) * 0.05 if len(all_y) else 1.0
        x_rng = (all_x.min() - x_pad, all_x.max() + x_pad)
        y_rng = (all_y.min() - y_pad, all_y.max() + y_pad)

        if len(ax_pts) >= 20:
            Ha, xe, ye = np.histogram2d(ax_pts, ay_pts, bins=100, range=[x_rng, y_rng])
            Ha_s = gaussian_filter(Ha.T, sigma=1.8)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            ax2.contour(xc, yc, Ha_s, levels=5, colors="#1b4975", alpha=0.55, linewidths=1.0, zorder=2)

        if len(tx_pts) >= 20:
            Ht, _, _ = np.histogram2d(tx_pts, ty_pts, bins=100, range=[x_rng, y_rng])
            Ht_s = gaussian_filter(Ht.T, sigma=1.8)
            xc = 0.5 * (xe[:-1] + xe[1:])
            yc = 0.5 * (ye[:-1] + ye[1:])
            ax2.contour(xc, yc, Ht_s, levels=5, colors="#b32d29", alpha=0.55, linewidths=1.0, zorder=2)

        ax2.scatter(ax_pts, ay_pts, color="#2b5c8f", marker="o", s=18, alpha=0.55, label=f"{src_label} Aligned (X W*)", zorder=3)
        ax2.scatter(tx_pts, ty_pts, color="#d9534f", marker="s", s=22, alpha=0.65, label=f"{trg_label} (Z)", zorder=4)

        # Top correspondence connections
        matched_pairs = []
        for a_pt in a_items:
            t_idx = a_pt.get("assigned_phoneme_index")
            if t_idx is not None and 0 <= t_idx < len(t_items):
                matched_pairs.append((a_pt.get("count", 1.0), a_pt, t_items[t_idx]))
        matched_pairs.sort(key=lambda x: x[0], reverse=True)
        for _, a_pt, target in matched_pairs[:35]:
            ax2.plot([a_pt["x"], target["x"]], [a_pt["y"], target["y"]], color="#555555", alpha=0.4, linestyle="--", linewidth=0.8, zorder=2)

        al_evr = aligned_pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
        ax2.set_title(f"2. After Alignment: Shared Space\n(Total 2D Var: {al_evr:.1f}% — Aligned Isomorphism)", fontsize=11, fontweight="bold")
        ax2.set_xlabel(f"PC 1 ({aligned_pca_result.get('explained_variance_ratio_pc1', 0.0)*100:.1f}%)", fontsize=9.5)
        ax2.set_ylabel(f"PC 2 ({aligned_pca_result.get('explained_variance_ratio_pc2', 0.0)*100:.1f}%)", fontsize=9.5)
        ax2.set_xlim(x_rng)
        ax2.set_ylim(y_rng)
        ax2.grid(True, linestyle=":", alpha=0.5)
        ax2.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc")

    # -------------------------------------------------------------
    # Helper for Bottom Panels (Single Modality Frequency Gradient)
    # -------------------------------------------------------------
    def render_modality_ax(ax_target, res_dict, title_text, colormap):
        m_items = res_dict.get("items_2d", res_dict.get("audio_clusters_2d", res_dict.get("text_phonemes_2d", [])))
        if not m_items:
            return
        mx = np.array([p["x"] for p in m_items])
        my = np.array([p["y"] for p in m_items])
        mc = np.array([p.get("count", 1.0) for p in m_items])
        mlog_c = np.log10(np.maximum(mc, 1.0))
        m_labels = [p["label"] for p in m_items]

        order = np.argsort(mc)
        mx_s, my_s, mc_s, mlog_c_s = mx[order], my[order], mc[order], mlog_c[order]
        c_min, c_max = mlog_c.min(), mlog_c.max()
        norm_c = (mlog_c_s - c_min) / (c_max - c_min + 1e-6)
        sizes = 14 + norm_c * 26
        alphas = 0.55 + norm_c * 0.35

        sc = ax_target.scatter(mx_s, my_s, c=mlog_c_s, cmap=colormap, s=sizes, alpha=alphas, edgecolors="none", zorder=3)

        if len(mx) >= 20:
            H, x_edges, y_edges = np.histogram2d(mx, my, bins=100)
            H_smooth = gaussian_filter(H.T, sigma=2.0)
            xc = 0.5 * (x_edges[:-1] + x_edges[1:])
            yc = 0.5 * (y_edges[:-1] + y_edges[1:])
            ax_target.contour(xc, yc, H_smooth, levels=6, colors="#444444", alpha=0.35, linewidths=0.8, zorder=2)

        cbar = fig.colorbar(sc, ax=ax_target, fraction=0.046, pad=0.04)
        cbar.formatter = FuncFormatter(format_count_tick)
        cbar.update_ticks()
        cbar.set_label("Frequency / Count (log scale)", fontsize=9, fontweight="bold")

        top_indices = order[-25:][::-1]
        for idx in top_indices:
            lab = m_labels[idx]
            if lab in ("<pad>", "<unk>", "<s>", "</s>") and len(top_indices) > 5:
                continue
            ax_target.annotate(
                lab,
                (mx[idx], my[idx]),
                fontsize=7.5,
                fontweight="bold",
                color="#111111",
                xytext=(3, 3),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.82, edgecolor="#888888", linewidth=0.5),
                zorder=5,
            )

        m_evr = res_dict.get("total_2d_explained_variance_ratio", 0.0) * 100
        ax_target.set_title(f"{title_text} (N={len(m_items)})\n(Total 2D Var: {m_evr:.1f}%)", fontsize=11, fontweight="bold")
        ax_target.set_xlabel(f"PC 1 ({res_dict.get('explained_variance_ratio_pc1', 0.0)*100:.1f}%)", fontsize=9.5)
        ax_target.set_ylabel(f"PC 2 ({res_dict.get('explained_variance_ratio_pc2', 0.0)*100:.1f}%)", fontsize=9.5)
        ax_target.grid(True, linestyle=":", alpha=0.5)

    # -------------------------------------------------------------
    # Bottom-Left: Source Modality Raw PCA
    # -------------------------------------------------------------
    render_modality_ax(ax3, source_raw_pca, f"3. {src_label} Word2Vec (Before Alignment)", src_cmap)

    # -------------------------------------------------------------
    # Bottom-Right: Target Modality Raw PCA
    # -------------------------------------------------------------
    render_modality_ax(ax4, target_raw_pca, f"4. {trg_label} Word2Vec (Before Alignment)", trg_cmap)

    fig.suptitle(f"{title_prefix} — Representation Spaces Before & After Alignment", fontsize=14, fontweight="bold", y=0.99)
    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    fig.savefig(out_png_path, bbox_inches="tight")
    plt.close(fig)


def load_vocab_dict(vocab_path: Optional[str]) -> Dict[int, str]:
    """
    Loads vocabulary from either a Python dict string file (e.g. {'AH': 0, ...})
    or a standard whitespace/newline-separated token file.
    """
    if vocab_path is None or not os.path.exists(vocab_path):
        return {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if content.startswith("{") and content.endswith("}"):
        import ast
        try:
            vocab_dict = ast.literal_eval(content)
            return {int(idx): str(tok) for tok, idx in vocab_dict.items()}
        except Exception:
            pass

    # Standard newline delimited
    res = {}
    for idx, line in enumerate(content.splitlines()):
        line = line.strip()
        if line:
            res[idx] = line.split()[0]
    return res


def load_counts_dict(
    counts_path: Optional[str],
    vocab_dict: Optional[Dict[int, str]] = None,
) -> Dict[int, float]:
    """
    Loads frequency counts from a counts file (formatted either as '<token> <count>' or line-by-line '<count>').
    Returns a mapping from integer index -> float count.
    """
    if counts_path is None or not os.path.exists(counts_path):
        return {}

    vocab_to_idx = {tok: idx for idx, tok in vocab_dict.items()} if vocab_dict else {}

    res = {}
    with open(counts_path, "r", encoding="utf-8", errors="replace") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 2:
                tok, cnt_str = parts[0], parts[1]
                try:
                    cnt = float(cnt_str)
                except ValueError:
                    continue
                if tok in vocab_to_idx:
                    res[vocab_to_idx[tok]] = cnt
                elif tok.isdigit():
                    res[int(tok)] = cnt
                elif tok.startswith("c") and tok[1:].isdigit():
                    res[int(tok[1:])] = cnt
                else:
                    res[idx] = cnt
            else:
                try:
                    res[idx] = float(parts[0])
                except ValueError:
                    pass
    return res


class EmbeddingAlignmentAnalysisJob(Job):
    """
    Standalone Sisyphus job that analyzes cross-modal / cross-lingual embedding alignment quality.
    Depends on mapped source/audio embeddings, target/text embeddings, and induced dictionary.
    Generates:
      - out_cross_modal_pca_plot: enhanced joint scatter plot with dual KDE density contours
      - out_source_pca_plot: source modality / language plot with log10-frequency color gradient
      - out_target_pca_plot: target modality / language plot with log10-frequency color gradient
      - out_cross_modal_panel_plot: comprehensive 3-panel figure showing Joint + Source + Target
      - out_cross_modal_pca_json: full 2D projection coordinates and metadata
      - out_analysis_report_json: alignment and entropy report
    """
    _version = 3

    def __init__(
        self,
        mapped_audio_embeddings_npy: tk.Path,
        text_embeddings_npy: tk.Path,
        dictionary_json: tk.Path,
        raw_audio_embeddings_npy: Optional[tk.Path] = None,
        raw_text_embeddings_npy: Optional[tk.Path] = None,
        raw_src_embeddings_npy: Optional[tk.Path] = None,
        raw_trg_embeddings_npy: Optional[tk.Path] = None,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        audio_counts_file: Optional[tk.Path] = None,
        text_counts_file: Optional[tk.Path] = None,
        src_counts_file: Optional[tk.Path] = None,
        trg_counts_file: Optional[tk.Path] = None,
        alignment_report_json: Optional[tk.Path] = None,
        title_prefix: str = "Cross-Modal 2D PCA Alignment",
        src_label: str = "Audio Clusters",
        trg_label: str = "Text Units",
        _version: int = 4,
    ):
        super().__init__()
        self._version = _version
        self.mapped_audio_embeddings_npy = mapped_audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.dictionary_json = dictionary_json
        self.raw_audio_embeddings_npy = raw_audio_embeddings_npy if raw_audio_embeddings_npy is not None else raw_src_embeddings_npy
        self.raw_text_embeddings_npy = raw_text_embeddings_npy if raw_text_embeddings_npy is not None else raw_trg_embeddings_npy
        self.audio_vocab_file = audio_vocab_file
        self.text_vocab_file = text_vocab_file
        self.audio_counts_file = audio_counts_file if audio_counts_file is not None else src_counts_file
        self.text_counts_file = text_counts_file if text_counts_file is not None else trg_counts_file
        self.alignment_report_json = alignment_report_json
        self.title_prefix = title_prefix
        self.src_label = src_label
        self.trg_label = trg_label

        # Outputs
        self.out_analysis_report_json = self.output_path("analysis_report.json")
        self.out_cross_modal_pca_json = self.output_path("cross_modal_pca.json")
        self.out_cross_modal_pca_plot = self.output_path("cross_modal_pca.png")
        self.out_source_pca_plot = self.output_path("source_modality_pca.png")
        self.out_target_pca_plot = self.output_path("target_modality_pca.png")
        self.out_cross_modal_panel_plot = self.output_path("cross_modal_pca_panel.png")
        self.out_source_raw_pca_plot = self.output_path("source_raw_pca.png")
        self.out_target_raw_pca_plot = self.output_path("target_raw_pca.png")
        self.out_unaligned_joint_pca_plot = self.output_path("unaligned_cross_modal_pca.png")
        self.out_alignment_comparison_panel_plot = self.output_path("alignment_comparison_panel.png")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        logging.basicConfig(level=logging.INFO)
        mapped_X = np.load(self.mapped_audio_embeddings_npy.get_path())
        Z_raw = np.load(self.text_embeddings_npy.get_path())
        Z = normalize_and_center(Z_raw)

        # Load dictionary mapping
        with open(self.dictionary_json.get_path(), "r") as f:
            dict_data = json.load(f)
            cluster_to_phoneme_map = dict_data.get("cluster_index_to_phoneme_index", [])

        # Load vocabs if provided
        audio_vocab_path = self.audio_vocab_file.get_path() if self.audio_vocab_file is not None else None
        text_vocab_path = self.text_vocab_file.get_path() if self.text_vocab_file is not None else None
        audio_vocab = load_vocab_dict(audio_vocab_path)
        text_vocab = load_vocab_dict(text_vocab_path)

        # Load frequency counts if provided
        audio_counts_path = self.audio_counts_file.get_path() if self.audio_counts_file is not None else None
        text_counts_path = self.text_counts_file.get_path() if self.text_counts_file is not None else None
        audio_counts = load_counts_dict(audio_counts_path, vocab_dict=audio_vocab)
        text_counts = load_counts_dict(text_counts_path, vocab_dict=text_vocab)

        # 1. Phoneme Coverage & Entropy
        cov_entropy = compute_phoneme_coverage_and_entropy(
            cluster_to_phoneme_map=cluster_to_phoneme_map,
            num_phonemes=Z.shape[0],
            phoneme_vocab=text_vocab,
        )

        # 2. Joint 2D PCA with counts (Aligned Space: X W* vs Z)
        pca_result = compute_joint_cross_modal_pca(
            mapped_X=mapped_X,
            Z=Z,
            audio_vocab=audio_vocab,
            text_vocab=text_vocab,
            cluster_to_phoneme_map=cluster_to_phoneme_map,
            audio_counts=audio_counts,
            text_counts=text_counts,
        )

        with open(self.out_cross_modal_pca_json.get_path(), "w") as f:
            json.dump(pca_result, f, indent=2)

        # 3. Resolve Raw (Before Alignment) Embeddings
        raw_X = None
        raw_audio_path = self.raw_audio_embeddings_npy.get_path() if self.raw_audio_embeddings_npy is not None else None
        if raw_audio_path is not None and os.path.exists(raw_audio_path):
            try:
                raw_X = np.load(raw_audio_path)
            except Exception as e:
                logging.warning(f"Could not load raw audio embeddings from {raw_audio_path}: {e}")

        if raw_X is None:
            # Fallback 1: check parent directories or sibling audio_embeddings.npy
            mapped_path = self.mapped_audio_embeddings_npy.get_path()
            mapped_dir = os.path.dirname(mapped_path)
            cand1 = os.path.join(mapped_dir, "audio_embeddings.npy")
            cand2 = os.path.join(os.path.dirname(mapped_dir), "audio_embeddings.npy")
            cand3 = os.path.join(os.path.dirname(os.path.dirname(mapped_dir)), "audio_embeddings.npy")
            for cand in [cand1, cand2, cand3]:
                if os.path.exists(cand):
                    try:
                        raw_X = np.load(cand)
                        break
                    except Exception:
                        pass

        if raw_X is None:
            # Fallback 2: unproject via W_star if present
            w_star_path = os.path.join(os.path.dirname(self.mapped_audio_embeddings_npy.get_path()), "W_star.npy")
            if os.path.exists(w_star_path):
                try:
                    W_star = np.load(w_star_path)
                    raw_X = mapped_X @ W_star.T
                except Exception:
                    raw_X = mapped_X
            else:
                raw_X = mapped_X

        raw_Z = None
        raw_text_path = self.raw_text_embeddings_npy.get_path() if self.raw_text_embeddings_npy is not None else None
        if raw_text_path is not None and os.path.exists(raw_text_path):
            try:
                raw_Z = np.load(raw_text_path)
            except Exception as e:
                logging.warning(f"Could not load raw text embeddings from {raw_text_path}: {e}")
        if raw_Z is None:
            raw_Z = Z_raw

        # 4. Compute Single-Modality PCA Before Alignment (with frequency gradients)
        source_raw_pca = compute_single_modality_pca(
            X=raw_X,
            vocab=audio_vocab,
            counts=audio_counts,
            prefix_label="src",
        )
        target_raw_pca = compute_single_modality_pca(
            X=raw_Z,
            vocab=text_vocab,
            counts=text_counts,
            prefix_label="trg",
        )

        # 5. Compute Joint PCA Before Alignment (Unaligned Space)
        raw_Z_norm = normalize_and_center(raw_Z)
        unaligned_pca_result = compute_joint_cross_modal_pca(
            mapped_X=raw_X,
            Z=raw_Z_norm,
            audio_vocab=audio_vocab,
            text_vocab=text_vocab,
            cluster_to_phoneme_map=cluster_to_phoneme_map,
            audio_counts=audio_counts,
            text_counts=text_counts,
        )

        # 6. Plotting
        # 6a. Enhanced Joint Plot (with dual KDE density contours & anchor lines)
        try:
            plot_joint_cross_modal_pca(
                pca_result=pca_result,
                out_png_path=self.out_cross_modal_pca_plot.get_path(),
                title=f"{self.title_prefix} (K={mapped_X.shape[0]}, P={Z.shape[0]})",
                src_label=self.src_label,
                trg_label=self.trg_label,
            )
        except Exception as e:
            logging.warning(f"Could not render joint PCA plot: {e}")

        # Compute shared coordinate ranges for consistent spatial scaling across individual plots
        a_x = [p["x"] for p in pca_result.get("audio_clusters_2d", [])]
        a_y = [p["y"] for p in pca_result.get("audio_clusters_2d", [])]
        t_x = [p["x"] for p in pca_result.get("text_phonemes_2d", [])]
        t_y = [p["y"] for p in pca_result.get("text_phonemes_2d", [])]
        all_x = np.array(a_x + t_x) if (a_x or t_x) else np.array([0.0])
        all_y = np.array(a_y + t_y) if (a_y or t_y) else np.array([0.0])
        x_pad = (all_x.max() - all_x.min()) * 0.05 if len(all_x) else 1.0
        y_pad = (all_y.max() - all_y.min()) * 0.05 if len(all_y) else 1.0
        shared_x_range = (float(all_x.min() - x_pad), float(all_x.max() + x_pad))
        shared_y_range = (float(all_y.min() - y_pad), float(all_y.max() + y_pad))

        # 6b. Source Modality / Language with Frequency Gradient (After Alignment Space)
        try:
            plot_single_modality_pca(
                pca_result=pca_result,
                out_png_path=self.out_source_pca_plot.get_path(),
                modality_key="audio_clusters_2d",
                title=f"{self.src_label} — 2D PCA Frequency Gradient",
                label=self.src_label,
                cmap="viridis",
                x_range=shared_x_range,
                y_range=shared_y_range,
            )
        except Exception as e:
            logging.warning(f"Could not render source modality PCA plot: {e}")

        # 6c. Target Modality / Language with Frequency Gradient
        try:
            plot_single_modality_pca(
                pca_result=pca_result,
                out_png_path=self.out_target_pca_plot.get_path(),
                modality_key="text_phonemes_2d",
                title=f"{self.trg_label} — 2D PCA Frequency Gradient",
                label=self.trg_label,
                cmap="plasma",
                x_range=shared_x_range,
                y_range=shared_y_range,
            )
        except Exception as e:
            logging.warning(f"Could not render target modality PCA plot: {e}")

        # 6d. Comprehensive 3-Panel Multi-Plot (Joint + Source + Target)
        try:
            plot_cross_modal_panel_pca(
                pca_result=pca_result,
                out_png_path=self.out_cross_modal_panel_plot.get_path(),
                title_prefix=self.title_prefix,
                src_label=self.src_label,
                trg_label=self.trg_label,
                src_cmap="viridis",
                trg_cmap="plasma",
            )
        except Exception as e:
            logging.warning(f"Could not render cross-modal PCA panel plot: {e}")

        # 6e. Source Raw Word2Vec PCA Before Alignment (with frequency gradient)
        try:
            plot_single_modality_pca(
                pca_result=source_raw_pca,
                out_png_path=self.out_source_raw_pca_plot.get_path(),
                modality_key="items_2d",
                title=f"{self.src_label} (Word2Vec Before Alignment)",
                label=self.src_label,
                cmap="viridis",
            )
        except Exception as e:
            logging.warning(f"Could not render source raw PCA plot: {e}")

        # 6f. Target Raw Word2Vec PCA Before Alignment (with frequency gradient)
        try:
            plot_single_modality_pca(
                pca_result=target_raw_pca,
                out_png_path=self.out_target_raw_pca_plot.get_path(),
                modality_key="items_2d",
                title=f"{self.trg_label} (Word2Vec Before Alignment)",
                label=self.trg_label,
                cmap="plasma",
            )
        except Exception as e:
            logging.warning(f"Could not render target raw PCA plot: {e}")

        # 6g. Unaligned Joint 2D PCA Before Alignment
        try:
            plot_joint_cross_modal_pca(
                pca_result=unaligned_pca_result,
                out_png_path=self.out_unaligned_joint_pca_plot.get_path(),
                title=f"Unaligned Word2Vec 2D PCA: {self.src_label} & {self.trg_label} (Before Alignment)",
                src_label=f"{self.src_label} (Raw)",
                trg_label=f"{self.trg_label} (Raw)",
            )
        except Exception as e:
            logging.warning(f"Could not render unaligned joint PCA plot: {e}")

        # 6h. Alignment Comparison Panel (2x2: Unaligned Joint, Aligned Joint, Source Raw, Target Raw)
        try:
            plot_alignment_comparison_panel(
                unaligned_pca_result=unaligned_pca_result,
                aligned_pca_result=pca_result,
                source_raw_pca=source_raw_pca,
                target_raw_pca=target_raw_pca,
                out_png_path=self.out_alignment_comparison_panel_plot.get_path(),
                title_prefix=self.title_prefix,
                src_label=self.src_label,
                trg_label=self.trg_label,
                src_cmap="viridis",
                trg_cmap="plasma",
            )
        except Exception as e:
            logging.warning(f"Could not render alignment comparison panel plot: {e}")

        # 7. Base and comprehensive report info
        base_report = {}
        if self.alignment_report_json is not None and os.path.exists(self.alignment_report_json.get_path()):
            try:
                with open(self.alignment_report_json.get_path(), "r") as f:
                    base_report = json.load(f)
            except Exception as e:
                logging.warning(f"Could not read base alignment report: {e}")

        analysis_report = {
            "base_alignment": base_report,
            "phoneme_coverage": {
                "unique_covered": cov_entropy["unique_phonemes_covered"],
                "total_phonemes": cov_entropy["num_phonemes"],
                "coverage_ratio": cov_entropy["coverage_ratio"],
                "entropy": cov_entropy["entropy"],
                "normalized_entropy": cov_entropy["normalized_entropy"],
                "max_theoretical_entropy": cov_entropy["max_theoretical_entropy"],
                "per_phoneme_cluster_counts": cov_entropy["phoneme_assignment_counts"],
            },
            "aligned_joint_pca_summary": {
                "explained_variance_ratio_pc1": pca_result["explained_variance_ratio_pc1"],
                "explained_variance_ratio_pc2": pca_result["explained_variance_ratio_pc2"],
                "total_2d_explained_variance_ratio": pca_result["total_2d_explained_variance_ratio"],
            },
            "unaligned_joint_pca_summary": {
                "explained_variance_ratio_pc1": unaligned_pca_result["explained_variance_ratio_pc1"],
                "explained_variance_ratio_pc2": unaligned_pca_result["explained_variance_ratio_pc2"],
                "total_2d_explained_variance_ratio": unaligned_pca_result["total_2d_explained_variance_ratio"],
            },
            "source_raw_pca_summary": {
                "explained_variance_ratio_pc1": source_raw_pca["explained_variance_ratio_pc1"],
                "explained_variance_ratio_pc2": source_raw_pca["explained_variance_ratio_pc2"],
                "total_2d_explained_variance_ratio": source_raw_pca["total_2d_explained_variance_ratio"],
            },
            "target_raw_pca_summary": {
                "explained_variance_ratio_pc1": target_raw_pca["explained_variance_ratio_pc1"],
                "explained_variance_ratio_pc2": target_raw_pca["explained_variance_ratio_pc2"],
                "total_2d_explained_variance_ratio": target_raw_pca["total_2d_explained_variance_ratio"],
            },
        }

        with open(self.out_analysis_report_json.get_path(), "w") as f:
            json.dump(analysis_report, f, indent=2)
