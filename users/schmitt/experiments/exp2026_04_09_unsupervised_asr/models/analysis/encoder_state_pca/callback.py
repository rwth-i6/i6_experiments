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
    :param y: if given, ``[M, F]`` states from the other modality -> average over all cross pairs
        (x_i, y_j). If ``None`` -> average over all distinct within-set pairs (i < j) of ``x``.
    :return: mean cosine similarity, or ``nan`` if there are no valid pairs.
    """
    x_norm = _l2_normalize(x)
    if y is None:
        n = x_norm.shape[0]
        if n < 2:
            return float("nan")
        sims = x_norm @ x_norm.T
        # sum over off-diagonal entries = sum over ordered pairs i!=j; divide by their count
        off_diag_sum = float(sims.sum() - np.trace(sims))
        mean_sim = off_diag_sum / (n * (n - 1))
        return mean_sim
    y_norm = _l2_normalize(y)
    if x_norm.shape[0] == 0 or y_norm.shape[0] == 0:
        return float("nan")
    mean_sim = float((x_norm @ y_norm.T).mean())
    return mean_sim


class EncoderStatePcaCallback(ForwardCallbackIface):
    """
    Forward callback for the shared-encoder state PCA analysis.

    Collects the per-frame shared-encoder states of both modalities (the ``audio_states`` /
    ``text_states`` outputs produced by the analysis forward_step), fits a single 2D PCA over the
    pooled states of *both* modalities, and writes, for every sequence seen in both modalities,
    a scatter plot (audio vs. text) plus the projected coordinates. This mirrors
    ``notebooks/visualize_embeds.py`` but lets RETURNN handle all data loading.

    Modalities are accumulated independently per ``seq_tag``, so this works both for a paired
    ``MetaDataset`` (both modalities in the same batch) and for a ``CombinedDataset`` where the
    audio and text views of the same ``seq_tag`` arrive in different (audio-only / text-only)
    batches.
    """

    def __init__(
        self,
        *,
        out_dir: str = "encoder_pca",
        max_points_per_modality: int = 50_000,
        plot_seq_tags: Optional[List[str]] = None,
        max_plotted_seqs: int = 20,
        cosine_similarity_summary: bool = False,
        audio_output: str = "audio_states",
        text_output: str = "text_states",
        vocab=None,  # injected by serialize_forward; unused here (we plot states, not labels)
    ):
        self.out_dir = out_dir
        self.max_points_per_modality = max_points_per_modality
        self.plot_seq_tags = set(plot_seq_tags) if plot_seq_tags is not None else None
        self.max_plotted_seqs = max_plotted_seqs
        # if True, also summarize avg pairwise cosine similarities (within/between modalities) in
        # the original feature space, which -- unlike the 2D PCA -- is not distorted by dim reduction.
        self.cosine_similarity_summary = cosine_similarity_summary
        self.audio_output = audio_output
        self.text_output = text_output

        # which seq_tags we keep states for / plot. With plot_seq_tags=None we lazily select the
        # first `max_plotted_seqs` distinct seq_tags (bounded memory on large datasets).
        self._plot_selected: set = set()

        # capped pools used for fitting the shared PCA
        self._audio_pool: List[np.ndarray] = []
        self._text_pool: List[np.ndarray] = []
        self._audio_pool_size = 0
        self._text_pool_size = 0

        # per-seq states kept for plotting (only for seq_tags we actually plot)
        self._audio_seqs: Dict[str, np.ndarray] = {}
        self._text_seqs: Dict[str, np.ndarray] = {}

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

    def _add_to_pool(self, states: np.ndarray, *, audio: bool) -> None:
        pool = self._audio_pool if audio else self._text_pool
        size = self._audio_pool_size if audio else self._text_pool_size
        remaining = self.max_points_per_modality - size
        if remaining <= 0:
            return
        if states.shape[0] > remaining:
            states = states[:remaining]
        pool.append(states)
        if audio:
            self._audio_pool_size += states.shape[0]
        else:
            self._text_pool_size += states.shape[0]

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        # the forward_step only marks an output for a modality that is present in the batch,
        # so a given seq may carry audio_states, text_states, or both.
        if self.audio_output in outputs.data:
            audio = np.asarray(outputs[self.audio_output].raw_tensor, dtype=np.float32)
            if audio.shape[0] > 0:
                self._add_to_pool(audio, audio=True)
                if self._should_plot(seq_tag):
                    self._audio_seqs[seq_tag] = audio
        if self.text_output in outputs.data:
            text = np.asarray(outputs[self.text_output].raw_tensor, dtype=np.float32)
            if text.shape[0] > 0:
                self._add_to_pool(text, audio=False)
                if self._should_plot(seq_tag):
                    self._text_seqs[seq_tag] = text

    def finish(self, **kwargs):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self._audio_pool or not self._text_pool:
            raise ValueError(
                "Need encoder states from both audio and text to fit the shared PCA"
                f" (audio_points={self._audio_pool_size}, text_points={self._text_pool_size})."
            )

        mean, components = _fit_pca_2d(np.concatenate(self._audio_pool + self._text_pool, axis=0))

        paired_seq_tags = sorted(set(self._audio_seqs) & set(self._text_seqs))

        # seq_tag -> {"audio_audio", "text_text", "audio_text"} avg pairwise cosine similarities
        cosine_similarities: Dict[str, Dict[str, float]] = {}

        with open(os.path.join(self.out_dir, "summary.txt"), "w") as summary:
            summary.write(f"pooled_audio_points={self._audio_pool_size}\n")
            summary.write(f"pooled_text_points={self._text_pool_size}\n")
            summary.write(f"num_paired_seqs={len(paired_seq_tags)}\n")

            for seq_tag in paired_seq_tags:
                audio_states = self._audio_seqs[seq_tag]
                text_states = self._text_seqs[seq_tag]
                audio_coords = _project(audio_states, mean, components)
                text_coords = _project(text_states, mean, components)

                npz_data = dict(
                    seq_tag=seq_tag,
                    audio_coords=audio_coords,
                    text_coords=text_coords,
                    audio_time_idx=np.arange(audio_coords.shape[0]),
                    text_time_idx=np.arange(text_coords.shape[0]),
                )

                if self.cosine_similarity_summary:
                    # computed on the raw encoder states (original feature space), not on the
                    # 2D PCA coordinates.
                    cos = {
                        "audio_audio": _avg_pairwise_cosine_similarity(audio_states),
                        "text_text": _avg_pairwise_cosine_similarity(text_states),
                        "audio_text": _avg_pairwise_cosine_similarity(audio_states, text_states),
                    }
                    cosine_similarities[seq_tag] = cos
                    npz_data.update(
                        audio_audio_cos_sim=cos["audio_audio"],
                        text_text_cos_sim=cos["text_text"],
                        audio_text_cos_sim=cos["audio_text"],
                    )

                safe_seq_tag = seq_tag.replace("/", "_")
                np.savez(os.path.join(self.out_dir, f"{safe_seq_tag}.npz"), **npz_data)

                plt.figure(figsize=(8, 8))
                plt.scatter(audio_coords[:, 0], audio_coords[:, 1], s=18, alpha=0.7, label="audio", c="#1f77b4")
                plt.scatter(text_coords[:, 0], text_coords[:, 1], s=18, alpha=0.7, label="text", c="#d62728")
                if len(audio_coords) > 1:
                    plt.plot(audio_coords[:, 0], audio_coords[:, 1], alpha=0.35, c="#1f77b4")
                if len(text_coords) > 1:
                    plt.plot(text_coords[:, 0], text_coords[:, 1], alpha=0.35, c="#d62728")
                title = f"Shared encoder PCA for {seq_tag}"
                if self.cosine_similarity_summary:
                    cos = cosine_similarities[seq_tag]
                    title += (
                        f"\navg cos-sim (orig. space)  a-a {cos['audio_audio']:.3f}"
                        f"  t-t {cos['text_text']:.3f}  a-t {cos['audio_text']:.3f}"
                    )
                plt.title(title)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.out_dir, f"{safe_seq_tag}.png"), dpi=200)
                plt.close()

                summary.write(
                    f"{seq_tag} audio_points={audio_coords.shape[0]} text_points={text_coords.shape[0]}\n"
                )

        if self.cosine_similarity_summary:
            self._write_cosine_summary(cosine_similarities)

    def _write_cosine_summary(self, cosine_similarities: Dict[str, Dict[str, float]]) -> None:
        """Write a human-readable table of avg pairwise cosine similarities + overall means."""
        keys = ("audio_audio", "text_text", "audio_text")
        with open(os.path.join(self.out_dir, "cosine_similarities.txt"), "w") as f:
            f.write("# average pairwise cosine similarity in the original feature space\n")
            f.write("# (NOT the 2D PCA space); a-a = within audio, t-t = within text, a-t = audio<->text\n")
            f.write(f"{'seq_tag':<60} {'a-a':>8} {'t-t':>8} {'a-t':>8}\n")
            for seq_tag in sorted(cosine_similarities):
                cos = cosine_similarities[seq_tag]
                f.write(f"{seq_tag:<60} {cos['audio_audio']:>8.4f} {cos['text_text']:>8.4f} {cos['audio_text']:>8.4f}\n")
            f.write("-" * 88 + "\n")
            means = {
                k: float(np.nanmean([cos[k] for cos in cosine_similarities.values()])) if cosine_similarities else float("nan")
                for k in keys
            }
            f.write(f"{'MEAN over seqs':<60} {means['audio_audio']:>8.4f} {means['text_text']:>8.4f} {means['audio_text']:>8.4f}\n")
