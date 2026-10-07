"""
Standalone Analysis Jobs for Cross-Modal Self-Learning Embedding Alignment (Artetxe et al., 2017).
Computes:
1. Phoneme Coverage & Distribution Entropy (Metric 1A)
2. Joint Cross-Modal 2D PCA & Projection Visualization (Metric 3A)
"""

from typing import Dict, List, Optional, Iterator
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
        audio_items.append({
            "index": idx,
            "label": a_name,
            "x": float(px),
            "y": float(py),
            "assigned_phoneme_index": assigned_p,
            "assigned_phoneme_label": p_name,
        })

    text_items = []
    for idx, (px, py) in enumerate(proj_text):
        t_name = text_vocab.get(idx, f"phoneme_{idx}") if text_vocab else f"phoneme_{idx}"
        text_items.append({
            "index": idx,
            "label": t_name,
            "x": float(px),
            "y": float(py),
        })

    return {
        "explained_variance_ratio_pc1": var_pc1,
        "explained_variance_ratio_pc2": var_pc2,
        "total_2d_explained_variance_ratio": evr_2d,
        "audio_clusters_2d": audio_items,
        "text_phonemes_2d": text_items,
    }


def plot_joint_cross_modal_pca(
    pca_result: Dict,
    out_png_path: str,
    title: str = "Joint Cross-Modal 2D PCA: Audio Clusters (X W*) & Phonemes (Z)",
):
    """
    Renders and saves a 2D scatter visualization connecting audio clusters to phonemes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=200)

    # Audio clusters
    a_x = [p["x"] for p in pca_result["audio_clusters_2d"]]
    a_y = [p["y"] for p in pca_result["audio_clusters_2d"]]

    # Text phonemes
    t_x = [p["x"] for p in pca_result["text_phonemes_2d"]]
    t_y = [p["y"] for p in pca_result["text_phonemes_2d"]]
    t_labels = [p["label"] for p in pca_result["text_phonemes_2d"]]

    # Connections between audio cluster and assigned phoneme
    for a_item in pca_result["audio_clusters_2d"]:
        target_idx = a_item.get("assigned_phoneme_index")
        if target_idx is not None and 0 <= target_idx < len(pca_result["text_phonemes_2d"]):
            target = pca_result["text_phonemes_2d"][target_idx]
            ax.plot(
                [a_item["x"], target["x"]],
                [a_item["y"], target["y"]],
                color="gray",
                alpha=0.3,
                linestyle="--",
                linewidth=0.75,
                zorder=1,
            )

    # Scatter audio clusters (blue circles)
    ax.scatter(
        a_x, a_y,
        color="#2b5c8f",
        marker="o",
        s=45,
        alpha=0.75,
        edgecolors="none",
        label=f"Audio Clusters X W* (K={len(a_x)})",
        zorder=2,
    )

    # Scatter text units (red squares)
    pt_size = 85 if len(t_x) <= 100 else 25
    pt_alpha = 0.95 if len(t_x) <= 100 else 0.55
    ax.scatter(
        t_x, t_y,
        color="#d9534f",
        marker="s",
        s=pt_size,
        alpha=pt_alpha,
        edgecolors="#7a1816" if len(t_x) <= 100 else "none",
        linewidth=0.8,
        label=f"Text Units Z (P={len(t_x)})",
        zorder=3,
    )

    # Text labels (annotate all if small vocabulary; annotate top matched if large to avoid clutter)
    if len(t_labels) <= 80:
        for px, py, lab in zip(t_x, t_y, t_labels):
            ax.annotate(
                lab,
                (px, py),
                fontsize=8,
                fontweight="bold",
                color="#4a0e0c",
                xytext=(4, 4),
                textcoords="offset points",
                zorder=4,
            )
    else:
        matched_indices = set()
        for a_item in pca_result["audio_clusters_2d"]:
            target_idx = a_item.get("assigned_phoneme_index")
            if target_idx is not None and 0 <= target_idx < len(pca_result["text_phonemes_2d"]):
                matched_indices.add(target_idx)
        for idx in sorted(list(matched_indices))[:50]:
            px, py, lab = t_x[idx], t_y[idx], t_labels[idx]
            ax.annotate(
                lab,
                (px, py),
                fontsize=7,
                fontweight="bold",
                color="#4a0e0c",
                xytext=(3, 3),
                textcoords="offset points",
                zorder=4,
            )

    evr = pca_result.get("total_2d_explained_variance_ratio", 0.0) * 100
    ax.set_title(f"{title}\n(Total 2D Explained Variance: {evr:.1f}%)", fontsize=12, fontweight="bold")
    ax.set_xlabel(f"PC 1 ({pca_result.get('explained_variance_ratio_pc1', 0.0)*100:.1f}% var)", fontsize=10)
    ax.set_ylabel(f"PC 2 ({pca_result.get('explained_variance_ratio_pc2', 0.0)*100:.1f}% var)", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="#cccccc")

    fig.tight_layout()
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


class EmbeddingAlignmentAnalysisJob(Job):
    """
    Standalone Sisyphus job that analyzes cross-modal embedding alignment quality.
    Depends on mapped audio embeddings, text phoneme embeddings, and induced dictionary.
    """
    _version = 2

    def __init__(
        self,
        mapped_audio_embeddings_npy: tk.Path,
        text_embeddings_npy: tk.Path,
        dictionary_json: tk.Path,
        audio_vocab_file: Optional[tk.Path] = None,
        text_vocab_file: Optional[tk.Path] = None,
        alignment_report_json: Optional[tk.Path] = None,
        title_prefix: str = "Cross-Modal 2D PCA Alignment",
        _version: int = 2,
    ):
        super().__init__()
        self._version = _version
        self.mapped_audio_embeddings_npy = mapped_audio_embeddings_npy
        self.text_embeddings_npy = text_embeddings_npy
        self.dictionary_json = dictionary_json
        self.audio_vocab_file = audio_vocab_file
        self.text_vocab_file = text_vocab_file
        self.alignment_report_json = alignment_report_json
        self.title_prefix = title_prefix

        # Outputs
        self.out_analysis_report_json = self.output_path("analysis_report.json")
        self.out_cross_modal_pca_json = self.output_path("cross_modal_pca.json")
        self.out_cross_modal_pca_plot = self.output_path("cross_modal_pca.png")

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

        # 1. Phoneme Coverage & Entropy
        cov_entropy = compute_phoneme_coverage_and_entropy(
            cluster_to_phoneme_map=cluster_to_phoneme_map,
            num_phonemes=Z.shape[0],
            phoneme_vocab=text_vocab,
        )

        # 2. Joint 2D PCA
        pca_result = compute_joint_cross_modal_pca(
            mapped_X=mapped_X,
            Z=Z,
            audio_vocab=audio_vocab,
            text_vocab=text_vocab,
            cluster_to_phoneme_map=cluster_to_phoneme_map,
        )

        with open(self.out_cross_modal_pca_json.get_path(), "w") as f:
            json.dump(pca_result, f, indent=2)

        # 3. Plotting
        try:
            plot_joint_cross_modal_pca(
                pca_result=pca_result,
                out_png_path=self.out_cross_modal_pca_plot.get_path(),
                title=f"{self.title_prefix} (K={mapped_X.shape[0]}, P={Z.shape[0]})",
            )
        except Exception as e:
            logging.warning(f"Could not render PCA plot: {e}")

        # 4. Optional base report info
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
            "joint_pca_summary": {
                "explained_variance_ratio_pc1": pca_result["explained_variance_ratio_pc1"],
                "explained_variance_ratio_pc2": pca_result["explained_variance_ratio_pc2"],
                "total_2d_explained_variance_ratio": pca_result["total_2d_explained_variance_ratio"],
            },
        }

        with open(self.out_analysis_report_json.get_path(), "w") as f:
            json.dump(analysis_report, f, indent=2)
