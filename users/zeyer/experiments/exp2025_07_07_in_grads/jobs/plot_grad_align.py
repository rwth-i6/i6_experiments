"""Grad-score saliency + alignment figure for one example utterance, wired into the Sis graph.

Like :class:`WriteLatexTableJob`, this is a graph node: it depends on a per-token grad-score HDF
(:class:`ExtractInGradsPerTokenJob`) and the dataset, reproduces the production DP alignment (same
``align_opts`` / energy-silence settings as :class:`WordAlignFromPerTokenGradsJob`), and renders the
**actual emission the DP consumes** -- per-token softmax-over-time of the energy-weighted grad, with
the separate silence/blank emission strip on top (this is exactly what the ``sil`` lever changes) --
plus the grad-aligned and gold word boundaries. Y-axis = the real tokens (chars/subwords), not words.
Outputs PNG (preview) + PDF (``\\includegraphics``). Auto-updates when the grads change.
"""

from typing import Optional, Any, Dict, List
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class PlotGradAlignJob(Job):
    """Render the per-token grad emission (as the DP sees it) + word alignment for one seq.

    :param grad_score_hdf: from :class:`ExtractInGradsPerTokenJob` (streams ``num_input_frames``,
        ``num_tokens``, ``num_tokens_per_word``, ``data``); single chunk per seq.
    :param align_opts / audio_energy_pow / blank_silence_energy_scale: must match the production
        :class:`WordAlignFromPerTokenGradsJob` so the drawn emission + boundaries equal the reported ones.
    :param seq_idx: which dataset seq to plot.
    """

    # v2: show the actual DP emission (softmax-over-time) + silence strip + per-token (char) y-labels.
    __sis_version__ = 2
    __sis_hash_exclude__ = {"tokenizer_dir": None}

    def __init__(
        self,
        *,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
        seq_idx: int,
        audio_energy_pow: float = 0.0,
        blank_silence_energy_scale: float = 0.0,
        boundary_source: str = "word_detail",
        title: Optional[str] = None,
        tokenizer_dir: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        super().__init__()
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts
        self.seq_idx = int(seq_idx)
        self.audio_energy_pow = float(audio_energy_pow)
        self.blank_silence_energy_scale = float(blank_silence_energy_scale)
        self.boundary_source = boundary_source
        self.title = title
        self.tokenizer_dir = tokenizer_dir
        self.returnn_root = returnn_root
        self.out_png = self.output_path("plot.png")
        self.out_pdf = self.output_path("plot.pdf")

    def tasks(self):
        yield Task("run", mini_task=True)

    def _token_labels(self, words: List[str], tokens_per_word, n_tok: int) -> Optional[List[str]]:
        """Reconstruct per-token labels from the reference words.

        The grad HDF stores only token COUNTS, not token strings. For char-level targets the tokens
        ARE the word characters (optionally with an inter-word separator token); reconstruct under the
        two common layouts and verify the count. Return ``None`` (caller falls back to word labels) if
        it doesn't match -- e.g. subword/BPE tokens, which we can't reconstruct without the tokenizer.
        """
        for sep in (None, "␣"):  # no separator, or a visible-space separator token between words
            cand: List[str] = []
            for wi, w in enumerate(words):
                if sep is not None and wi > 0:
                    cand.append(sep)
                cand.extend(list(w))
            if len(cand) == n_tok:
                return cand
        return None

    def run(self):
        import os
        import sys
        import numpy as np

        set_hf_offline_mode()

        import i6_experiments

        recipe_dir = os.path.dirname(os.path.dirname(i6_experiments.__file__))
        sys.path.insert(0, recipe_dir)

        import i6_core.util as util

        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.hdf import HDFDataset
        from i6_experiments.users.zeyer.experiments.exp2025_05_05_align import Aligner

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        si = self.seq_idx
        hdf = HDFDataset([self.grad_score_hdf.get_path()])
        hdf.initialize()
        hdf.init_seq_order(epoch=1)
        hdf.load_seqs(0, si + 1)  # in-order, gap-free (HDFDataset requirement)

        from datasets import load_dataset

        data = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))[self.dataset_key][si]
        samplerate = data["audio"]["sampling_rate"]
        audio = np.asarray(data["audio"]["array"], dtype=np.float64)
        audio_len_secs = len(audio) / samplerate

        n_tf = int(hdf.get_data(si, "num_input_frames")[0, 0])
        n_tok = int(hdf.get_data(si, "num_tokens")[0, 0])
        assert n_tok > 0, f"seq {si} has no tokens (empty recog) -- pick another seq_idx"
        tokens_per_word = hdf.get_data(si, "num_tokens_per_word").reshape(-1).astype(int)
        grad_mat = hdf.get_data(si, self.grad_score_key).reshape(n_tok, n_tf).astype(np.float64)
        secs_per_tf = audio_len_secs / n_tf

        if n_tok > n_tf:  # char tokens on a coarse grid -> upsample time, same as the align job
            up = -(-n_tok // n_tf)
            grad_mat = np.repeat(grad_mat, up, axis=1)
            n_tf *= up
            secs_per_tf = audio_len_secs / n_tf

        # Energy envelope on the grad frame grid (silence-aware blank + energy weighting), same as align.
        win = max(int(0.025 * samplerate), 1)
        hann = np.hanning(win)
        hann = hann / hann.sum()
        env = np.sqrt(np.convolve(audio * audio, hann, mode="same") + 1e-9)
        centers = ((np.arange(n_tf) + 0.5) / n_tf * (len(env) - 1)).astype(int)
        e = env[centers]
        e = e / (e.max() + 1e-9)

        # Blank/silence emission row (whisper-timestamped-style VAD analog), from the ORIGINAL (pre-energy)
        # grads -- exactly as WordAlignFromPerTokenGradsJob computes blank_override.
        _s0 = np.log(grad_mat + 1e-12)
        _s0 = _s0 - max(float(_s0.max()), 0.0)
        _m0 = _s0.max(axis=1, keepdims=True)
        _ls0 = _s0 - _m0 - np.log(np.sum(np.exp(_s0 - _m0), axis=1, keepdims=True))
        _mean_t, _std_t = _ls0.mean(0), _ls0.std(0)
        blank_score = float(self.align_opts.get("blank_score", -5))
        if self.blank_silence_energy_scale:
            _ze = (e - e.mean()) / (e.std() + 1e-9)
            blank_row = _mean_t - self.blank_silence_energy_scale * _ze * _std_t
        else:
            blank_row = np.full(n_tf, blank_score, dtype=np.float64)

        # Token emission as the DP sees it: energy-weighted grad, then log-softmax over TIME per token.
        grad_for_align = grad_mat * (e[None, :] ** self.audio_energy_pow) if self.audio_energy_pow else grad_mat
        _e = np.log(grad_for_align + 1e-12)
        _e = _e - _e.max(axis=1, keepdims=True)
        emis = np.exp(_e - np.log(np.sum(np.exp(_e), axis=1, keepdims=True) + 1e-12))  # [n_tok, n_tf], rows ~sum 1

        # The boundaries the production DP produces (faithful: same Aligner, same blank_override).
        aligner = Aligner(**self.align_opts)
        tok_se = aligner.align(grad_for_align, blank_override=(blank_row if self.blank_silence_energy_scale else None))
        assert len(tok_se) == n_tok
        pred, cur = [], 0
        for k in tokens_per_word:
            pred.append((tok_se[cur][0] * secs_per_tf, tok_se[cur + k - 1][1] * secs_per_tf))
            cur += k

        words: List[str] = list(data[self.boundary_source]["utterance"])
        scale = self.dataset_offset_factors / samplerate
        gold = [
            (s * scale, en * scale)
            for s, en in zip(data[self.boundary_source]["start"], data[self.boundary_source]["stop"])
        ]
        assert len(words) == len(tokens_per_word) == len(gold)

        tok_labels = self._token_labels(words, tokens_per_word, n_tok)
        if tok_labels is None and self.tokenizer_dir is not None:
            # Subword/BPE: recover the actual pieces by re-tokenizing the SAME string the extract used
            # (the word-level path: " " + " ".join(words)) and decoding each token. \u2423 = space.
            from transformers import AutoTokenizer

            _tk = AutoTokenizer.from_pretrained(get_content_dir_from_hub_cache_dir(self.tokenizer_dir))
            _ids = _tk(" " + " ".join(words), add_special_tokens=False).input_ids
            if len(_ids) == n_tok:
                tok_labels = [(_tk.decode([i]).replace(" ", "\u2423") or "\u00b7") for i in _ids]

        # silence strip (softmax-over-time of the blank row, for display) above the token emission.
        _b = blank_row - blank_row.max()
        sil_disp = np.exp(_b - np.log(np.sum(np.exp(_b)) + 1e-12))[None, :]

        fig_h = max(2.8, min(11.0, 0.20 * n_tok + 0.8))
        fig = plt.figure(figsize=(12, fig_h))
        gs = fig.add_gridspec(2, 1, height_ratios=[0.10, 1.0], hspace=0.06)
        ax_sil = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1], sharex=ax_sil)

        ax_sil.imshow(sil_disp, aspect="auto", origin="lower", extent=[0.0, audio_len_secs, 0.0, 1.0], cmap="magma")
        ax_sil.set_yticks([0.5])
        ax_sil.set_yticklabels(["∅/sil"], fontsize=7)
        ax_sil.tick_params(labelbottom=False)
        if self.title:
            ax_sil.set_title(self.title, fontsize=10)

        ax.imshow(emis, aspect="auto", origin="lower", extent=[0.0, audio_len_secs, 0.0, n_tok], cmap="magma")
        # grad-aligned (lime) + gold (cyan) span per word, drawn over the word's token rows.
        cur = 0
        for wi, k in enumerate(tokens_per_word):
            yc = cur + k / 2.0
            ax.plot(pred[wi], [yc + 0.16 * k, yc + 0.16 * k], color="#39ff14", lw=2.0, solid_capstyle="butt")
            ax.plot(gold[wi], [yc - 0.16 * k, yc - 0.16 * k], color="#00e5ff", lw=2.0, ls=(0, (2, 1)))
            cur += k
        if tok_labels is not None:
            ax.set_yticks(np.arange(n_tok) + 0.5)
            ax.set_yticklabels(tok_labels, fontsize=6)
            ax.set_ylabel("target tokens")
        else:  # subword/BPE: no token strings available -> label words at group centers
            yt, cur = [], 0
            for k in tokens_per_word:
                yt.append(cur + k / 2.0)
                cur += k
            ax.set_yticks(yt)
            ax.set_yticklabels(words, fontsize=7)
            ax.set_ylabel("target tokens (by word; subword strings n/a)")
        ax.set_xlim(0.0, audio_len_secs)
        ax.set_ylim(0.0, n_tok)
        ax.set_xlabel("time [s]")
        ax.legend(
            handles=[
                Line2D([0], [0], color="#39ff14", lw=2.0, label="grad-align"),
                Line2D([0], [0], color="#00e5ff", lw=2.0, ls=(0, (2, 1)), label="gold"),
            ],
            loc="upper right",
            fontsize=8,
            framealpha=0.6,
        )
        fig.savefig(self.out_png.get_path(), dpi=200, bbox_inches="tight")
        fig.savefig(self.out_pdf.get_path(), bbox_inches="tight")
        print(f"seq {si}: {n_tok} tokens, {len(words)} words, {audio_len_secs:.2f}s -> {self.out_png.get_path()}")
