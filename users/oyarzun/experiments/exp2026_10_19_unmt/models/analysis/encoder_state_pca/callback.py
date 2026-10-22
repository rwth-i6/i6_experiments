__all__ = ["EncoderStatePcaCallback"]

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


def _fit_pca_2d(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a 2D PCA, returning (mean [F], components [F, 2])."""
    samples = samples.astype(np.float32)
    mean = samples.mean(axis=0, keepdims=True)
    centered = samples - mean
    # economy SVD; the right singular vectors are the principal axes
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].T  # [F, 2]
    return mean[0], components


def _project(samples: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (samples.astype(np.float32) - mean[None]) @ components


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def _avg_pairwise_cosine_similarity(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Average pairwise cosine similarity in the *original* feature space.

    Operates on the raw (pre-PCA) encoder states, so it is not affected by the 2D PCA collapse.

    :param x: ``[N, F]`` states.
    :param y: if given, ``[M, F]`` states from the other language -> average over all cross pairs
        (x_i, y_j). If ``None`` -> average over all distinct within-set pairs (i < j) of ``x``.
    :return: mean cosine similarity, or ``nan`` if there are no valid pairs.
    """
    x_norm = _l2_normalize(x)
    if y is None:
        n = x_norm.shape[0]
        if n < 2:
            return float("nan")
        sims = x_norm @ x_norm.T
        off_diag_sum = float(sims.sum() - np.trace(sims))
        mean_sim = off_diag_sum / (n * (n - 1))
        return mean_sim
    y_norm = _l2_normalize(y)
    if x_norm.shape[0] == 0 or y_norm.shape[0] == 0:
        return float("nan")
    mean_sim = float((x_norm @ y_norm.T).mean())
    return mean_sim


def _avg_pairwise_l2_distance(x: np.ndarray, y: Optional[np.ndarray] = None) -> float:
    """
    Average pairwise L2 distance in the *original* feature space.
    """
    x = x.astype(np.float64)
    if y is None:
        n = x.shape[0]
        if n < 2:
            return float("nan")
        x2 = np.sum(x**2, axis=1)
        dists = np.sqrt(np.clip(x2[:, None] + x2[None, :] - 2 * (x @ x.T), 0, None))
        off_diag_sum = float(dists.sum())
        mean_dist = off_diag_sum / (n * (n - 1))
        return mean_dist
    y = y.astype(np.float64)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    x2 = np.sum(x**2, axis=1)
    y2 = np.sum(y**2, axis=1)
    dists = np.sqrt(np.clip(x2[:, None] + y2[None, :] - 2 * (x @ y.T), 0, None))
    mean_dist = float(dists.mean())
    return mean_dist


class EncoderStatePcaCallback(ForwardCallbackIface):
    """
    Forward callback for the cross-lingual shared-encoder state PCA analysis.

    Collects the per-token shared-encoder states of both languages (e.g. ``de`` vs. ``en``),
    fits a single 2D PCA over the pooled states of *both* languages, and writes:
      1. A global cross-lingual PCA scatter plot (``global_language_pca.png``) comparing the
         pooled token representations of both languages.
      2. For every sequence seen in both languages, a per-sentence scatter + trajectory plot
         comparing source language (e.g. ``de``) vs. target language (e.g. ``en``) encoder states,
         plus projected coordinates (``*.npz``) and original-space pairwise cosine similarity
         and L2 distance summaries (``cosine_similarities.txt`` and ``summary.txt``).
    """

    def __init__(
        self,
        *,
        out_dir: str = "encoder_pca",
        max_points_per_modality: int = 50_000,
        plot_seq_tags: Optional[List[str]] = None,
        max_plotted_seqs: int = 20,
        cosine_similarity_summary: bool = False,
        src_label: str = "de",
        trg_label: str = "en",
        audio_output: str = "audio_states",
        text_output: str = "text_states",
        vocab=None,  # injected by serialize_forward; unused here (we plot states, not labels)
    ):
        self.out_dir = out_dir
        self.max_points_per_modality = max_points_per_modality
        self.plot_seq_tags = set(plot_seq_tags) if plot_seq_tags is not None else None
        self.max_plotted_seqs = max_plotted_seqs
        self.cosine_similarity_summary = cosine_similarity_summary
        self.src_label = src_label
        self.trg_label = trg_label
        self.audio_output = audio_output
        self.text_output = text_output

        self._plot_selected: set = set()

        # Capped pools used for fitting the shared cross-lingual PCA
        self._src_pool: List[np.ndarray] = []
        self._trg_pool: List[np.ndarray] = []
        self._src_pool_size = 0
        self._trg_pool_size = 0

        # Per-seq states kept for plotting
        self._src_seqs: Dict[str, np.ndarray] = {}
        self._trg_seqs: Dict[str, np.ndarray] = {}

    def init(self, *args, **kwargs):
        os.makedirs(self.out_dir, exist_ok=True)

    def _should_plot(self, seq_tag: str) -> bool:
        if self.plot_seq_tags is not None:
            return seq_tag in self.plot_seq_tags
        if seq_tag in self._plot_selected:
            return True
        if len(self._plot_selected) < self.max_plotted_seqs:
            self._plot_selected.add(seq_tag)
            return True
        return False

    def _add_to_pool(self, states: np.ndarray, *, is_src: bool) -> None:
        pool = self._src_pool if is_src else self._trg_pool
        size = self._src_pool_size if is_src else self._trg_pool_size
        remaining = self.max_points_per_modality - size
        if remaining <= 0:
            return
        if states.shape[0] > remaining:
            states = states[:remaining]
        pool.append(states)
        if is_src:
            self._src_pool_size += states.shape[0]
        else:
            self._trg_pool_size += states.shape[0]

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        if self.audio_output in outputs.data:
            src_arr = np.asarray(outputs[self.audio_output].raw_tensor, dtype=np.float32)
            if src_arr.shape[0] > 0:
                self._add_to_pool(src_arr, is_src=True)
                if self._should_plot(seq_tag):
                    self._src_seqs[seq_tag] = src_arr
        if self.text_output in outputs.data:
            trg_arr = np.asarray(outputs[self.text_output].raw_tensor, dtype=np.float32)
            if trg_arr.shape[0] > 0:
                self._add_to_pool(trg_arr, is_src=False)
                if self._should_plot(seq_tag):
                    self._trg_seqs[seq_tag] = trg_arr

    def finish(self, **kwargs):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._src_pool or not self._trg_pool:
            raise ValueError(
                f"Need encoder states from both languages ({self.src_label} and {self.trg_label}) to fit the shared PCA "
                f"({self.src_label}_points={self._src_pool_size}, {self.trg_label}_points={self._trg_pool_size})."
            )

        all_src = np.concatenate(self._src_pool, axis=0)
        all_trg = np.concatenate(self._trg_pool, axis=0)
        mean, components = _fit_pca_2d(np.concatenate([all_src, all_trg], axis=0))

        # Plot global cross-lingual PCA cloud (subsampled to at most 5000 points per language for clarity)
        rng = np.random.RandomState(42)
        src_sample = all_src[rng.choice(len(all_src), size=min(5000, len(all_src)), replace=False)]
        trg_sample = all_trg[rng.choice(len(all_trg), size=min(5000, len(all_trg)), replace=False)]
        src_global_coords = _project(src_sample, mean, components)
        trg_global_coords = _project(trg_sample, mean, components)

        plt.figure(figsize=(8, 8))
        plt.scatter(
            src_global_coords[:, 0], src_global_coords[:, 1],
            s=10, alpha=0.4, label=f"lang: {self.src_label}", c="#1f77b4",
        )
        plt.scatter(
            trg_global_coords[:, 0], trg_global_coords[:, 1],
            s=10, alpha=0.4, label=f"lang: {self.trg_label}", c="#d62728",
        )
        plt.title(f"Global Shared Encoder PCA ({self.src_label} vs. {self.trg_label})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_dir, "global_language_pca.png"), dpi=200)
        plt.close()

        paired_seq_tags = sorted(set(self._src_seqs) & set(self._trg_seqs))
        cosine_similarities: Dict[str, Dict[str, float]] = {}

        with open(os.path.join(self.out_dir, "summary.txt"), "w") as summary:
            summary.write(f"src_lang={self.src_label}\n")
            summary.write(f"trg_lang={self.trg_label}\n")
            summary.write(f"pooled_{self.src_label}_points={self._src_pool_size}\n")
            summary.write(f"pooled_{self.trg_label}_points={self._trg_pool_size}\n")
            summary.write(f"num_paired_seqs={len(paired_seq_tags)}\n")

            for seq_tag in paired_seq_tags:
                src_states = self._src_seqs[seq_tag]
                trg_states = self._trg_seqs[seq_tag]
                src_coords = _project(src_states, mean, components)
                trg_coords = _project(trg_states, mean, components)

                npz_data = dict(
                    seq_tag=seq_tag,
                    src_lang=self.src_label,
                    trg_lang=self.trg_label,
                    src_coords=src_coords,
                    trg_coords=trg_coords,
                    # Keep backward-compatible keys as well
                    audio_coords=src_coords,
                    text_coords=trg_coords,
                    src_time_idx=np.arange(src_coords.shape[0]),
                    trg_time_idx=np.arange(trg_coords.shape[0]),
                )

                if self.cosine_similarity_summary:
                    cos = {
                        "src_src": _avg_pairwise_cosine_similarity(src_states),
                        "trg_trg": _avg_pairwise_cosine_similarity(trg_states),
                        "src_trg": _avg_pairwise_cosine_similarity(src_states, trg_states),
                    }
                    cosine_similarities[seq_tag] = cos
                    npz_data.update(
                        src_src_cos_sim=cos["src_src"],
                        trg_trg_cos_sim=cos["trg_trg"],
                        src_trg_cos_sim=cos["src_trg"],
                    )

                safe_seq_tag = seq_tag.replace("/", "_")
                np.savez(os.path.join(self.out_dir, f"{safe_seq_tag}.npz"), **npz_data)

                plt.figure(figsize=(8, 8))
                plt.scatter(src_coords[:, 0], src_coords[:, 1], s=18, alpha=0.7, label=self.src_label, c="#1f77b4")
                plt.scatter(trg_coords[:, 0], trg_coords[:, 1], s=18, alpha=0.7, label=self.trg_label, c="#d62728")
                if len(src_coords) > 1:
                    plt.plot(src_coords[:, 0], src_coords[:, 1], alpha=0.35, c="#1f77b4")
                if len(trg_coords) > 1:
                    plt.plot(trg_coords[:, 0], trg_coords[:, 1], alpha=0.35, c="#d62728")
                self._mark_start_end(plt, src_coords, label_prefix=self.src_label)
                self._mark_start_end(plt, trg_coords, label_prefix=self.trg_label)
                title = f"Shared encoder PCA ({self.src_label} vs. {self.trg_label}) for {seq_tag}"
                if self.cosine_similarity_summary:
                    cos = cosine_similarities[seq_tag]
                    dist = {
                        "src_src": _avg_pairwise_l2_distance(src_states),
                        "trg_trg": _avg_pairwise_l2_distance(trg_states),
                        "src_trg": _avg_pairwise_l2_distance(src_states, trg_states),
                    }
                    title += (
                        f"\navg cos-sim (orig. space)  {self.src_label}-{self.src_label} {cos['src_src']:.3f}"
                        f"  {self.trg_label}-{self.trg_label} {cos['trg_trg']:.3f}"
                        f"  {self.src_label}-{self.trg_label} {cos['src_trg']:.3f}"
                        f"\navg L2-dist (orig. space)  {self.src_label}-{self.src_label} {dist['src_src']:.3f}"
                        f"  {self.trg_label}-{self.trg_label} {dist['trg_trg']:.3f}"
                        f"  {self.src_label}-{self.trg_label} {dist['src_trg']:.3f}"
                    )
                plt.title(title)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"{safe_seq_tag}.png"), dpi=200)
                plt.close()

                summary.write(
                    f"{seq_tag} {self.src_label}_points={src_coords.shape[0]} {self.trg_label}_points={trg_coords.shape[0]}\n"
                )

        if self.cosine_similarity_summary:
            self._write_cosine_summary(cosine_similarities)

    @staticmethod
    def _mark_start_end(plt, coords: np.ndarray, *, label_prefix: str) -> None:
        if len(coords) == 0:
            return
        plt.scatter(
            coords[0, 0], coords[0, 1], s=120, marker="*", c="#2ca02c",
            edgecolors="black", linewidths=0.8, zorder=5, label=f"{label_prefix} start",
        )
        if len(coords) > 1:
            plt.scatter(
                coords[-1, 0], coords[-1, 1], s=120, marker="X", c="black",
                edgecolors="white", linewidths=0.8, zorder=5, label=f"{label_prefix} end",
            )

    def _write_cosine_summary(self, cosine_similarities: Dict[str, Dict[str, float]]) -> None:
        keys = ("src_src", "trg_trg", "src_trg")
        s_s = f"{self.src_label}-{self.src_label}"
        t_t = f"{self.trg_label}-{self.trg_label}"
        s_t = f"{self.src_label}-{self.trg_label}"
        with open(os.path.join(self.out_dir, "cosine_similarities.txt"), "w") as f:
            f.write("# average pairwise cosine similarity in the original encoder feature space\n")
            f.write(
                f"# (NOT the 2D PCA space); {s_s} = within {self.src_label}, "
                f"{t_t} = within {self.trg_label}, {s_t} = {self.src_label}<->{self.trg_label}\n"
            )
            f.write(f"{'seq_tag':<60} {s_s:>8} {t_t:>8} {s_t:>8}\n")
            for seq_tag in sorted(cosine_similarities):
                cos = cosine_similarities[seq_tag]
                f.write(f"{seq_tag:<60} {cos['src_src']:>8.4f} {cos['trg_trg']:>8.4f} {cos['src_trg']:>8.4f}\n")
            f.write("-" * 88 + "\n")
            means = {
                k: float(np.nanmean([cos[k] for cos in cosine_similarities.values()])) if cosine_similarities else float("nan")
                for k in keys
            }
            f.write(f"{'MEAN over seqs':<60} {means['src_src']:>8.4f} {means['trg_trg']:>8.4f} {means['src_trg']:>8.4f}\n")
