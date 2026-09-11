__all__ = ["CrossAttentionWeightsCallback"]

import os
from typing import Dict, List, Optional, Union

import numpy as np

from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict


class CrossAttentionWeightsCallback(ForwardCallbackIface):
    """
    Forward callback for the decoder cross-attention analysis.

    Consumes the ``att_weights`` (``[layer, head, query, key]`` per seq) and ``labels``
    (``[query]``) outputs of the cross-attention forward step and writes, per sequence:

    - ``<seq_tag>_layer<l>.png``: one ``matshow`` per attention head of decoder layer ``l``,
    - ``<seq_tag>_head-avg.png``: the head-averaged attention of every decoder layer,
    - ``<seq_tag>.npz``: the raw weights + the (decoded) query labels, for custom re-plotting.

    Only the first ``max_plotted_seqs`` sequences (or exactly ``plot_seq_tags``) are written. With
    the default lazy selection this is the same set of sequences the encoder-state PCA analysis
    plots, since both take the first sequences of the same dataset in dataset order.
    """

    def __init__(
        self,
        *,
        out_dir: str = "cross_att",
        plot_seq_tags: Optional[List[str]] = None,
        max_plotted_seqs: int = 20,
        plot_layers: Optional[List[int]] = None,
        plot_head_average: bool = True,
        save_npz: bool = True,
        vocab: Optional[Union[str, Dict]] = None,  # injected by serialize_forward
    ):
        self.out_dir = out_dir
        self.plot_seq_tags = set(plot_seq_tags) if plot_seq_tags is not None else None
        self.max_plotted_seqs = max_plotted_seqs
        # which decoder layers to plot per-head; None -> all
        self.plot_layers = plot_layers
        self.plot_head_average = plot_head_average
        self.save_npz = save_npz
        self.vocab_opts = vocab

        self.vocab: Optional[Vocabulary] = None
        self._plot_selected: set = set()
        self._summary = None

    def init(self, *args, **kwargs):
        os.makedirs(self.out_dir, exist_ok=True)
        if isinstance(self.vocab_opts, str):
            self.vocab = Vocabulary.create_vocab(vocab_file=self.vocab_opts, unknown_label=None)
        elif isinstance(self.vocab_opts, dict):
            self.vocab = Vocabulary.create_vocab(**self.vocab_opts)
        self._summary = open(os.path.join(self.out_dir, "summary.txt"), "w")

    def _should_plot(self, seq_tag: str) -> bool:
        if self.plot_seq_tags is not None:
            return seq_tag in self.plot_seq_tags
        if seq_tag in self._plot_selected:
            return True
        if len(self._plot_selected) < self.max_plotted_seqs:
            self._plot_selected.add(seq_tag)
            return True
        return False

    def _label(self, idx: int) -> str:
        """Human-readable label for a decoder input token (the vocab's own labels + bos)."""
        if self.vocab is None:
            return f"<{idx}>"
        num_labels = self.vocab.num_labels
        if 0 <= idx < num_labels:
            return self.vocab.id_to_label(idx)
        # the model appends [mask, bos, eos] above the vocab (its `+3` convention)
        return {num_labels: "<mask>", num_labels + 1: "<bos>", num_labels + 2: "<eos>"}.get(idx, f"<{idx}>")

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        # the forward step stops marking outputs once its seq limit is reached -> nothing to do.
        if "att_weights" not in outputs.data:
            return
        if not self._should_plot(seq_tag):
            return

        att_weights = np.asarray(outputs["att_weights"].raw_tensor, dtype=np.float32)  # [L, H, q, kv]
        assert att_weights.ndim == 4, f"unexpected att_weights shape {att_weights.shape}"
        labels = None
        if "labels" in outputs.data:
            labels = [self._label(int(i)) for i in np.asarray(outputs["labels"].raw_tensor)]

        safe_seq_tag = seq_tag.replace("/", "_")
        if self.save_npz:
            np.savez_compressed(
                os.path.join(self.out_dir, f"{safe_seq_tag}.npz"),
                seq_tag=seq_tag,
                att_weights=att_weights,
                labels=np.array(labels if labels is not None else [], dtype=object),
            )

        num_layers, num_heads, num_queries, num_keys = att_weights.shape
        layers = self.plot_layers if self.plot_layers is not None else list(range(num_layers))
        for layer in layers:
            self._plot_grid(
                [att_weights[layer, head] for head in range(num_heads)],
                titles=[f"head {head}" for head in range(num_heads)],
                labels=labels,
                out_file=os.path.join(self.out_dir, f"{safe_seq_tag}_layer{layer}.png"),
                suptitle=f"cross attention, layer {layer}\n{seq_tag}",
            )
        if self.plot_head_average:
            self._plot_grid(
                [att_weights[layer].mean(axis=0) for layer in range(num_layers)],
                titles=[f"layer {layer}" for layer in range(num_layers)],
                labels=labels,
                out_file=os.path.join(self.out_dir, f"{safe_seq_tag}_head-avg.png"),
                suptitle=f"cross attention (head average)\n{seq_tag}",
            )

        self._summary.write(
            f"{seq_tag} layers={num_layers} heads={num_heads} queries={num_queries} keys={num_keys}\n"
        )
        self._summary.flush()

    def _plot_grid(self, mats: List[np.ndarray], *, titles: List[str], labels, out_file: str, suptitle: str):
        """Plot several [query, key] attention matrices as a grid of ``matshow`` panels."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        num_cols = min(4, len(mats))
        num_rows = (len(mats) + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(6 * num_cols, 4 * num_rows), squeeze=False, constrained_layout=True
        )
        for ax_idx, ax in enumerate(axes.flat):
            if ax_idx >= len(mats):
                ax.axis("off")
                continue
            att = mats[ax_idx]
            # matshow (not imshow) as in the notebook; aspect="auto" since the key axis (encoder
            # frames) is usually much longer than the query axis (labels).
            im = ax.matshow(att, cmap="Blues", aspect="auto", vmin=0.0)
            ax.set_title(titles[ax_idx])
            ax.xaxis.set_ticks_position("bottom")  # matshow puts them on top, next to the title
            ax.set_xlabel("encoder frame")
            ax.set_ylabel("label position")
            # only annotate the query axis when it stays readable
            if labels is not None and len(labels) <= 60:
                ax.set_yticks(range(len(labels)))
                ax.set_yticklabels(labels, fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.03)
        fig.suptitle(suptitle)
        fig.savefig(out_file, dpi=150)
        plt.close(fig)

    def finish(self, **kwargs):
        if self._summary is not None:
            self._summary.close()
            self._summary = None
