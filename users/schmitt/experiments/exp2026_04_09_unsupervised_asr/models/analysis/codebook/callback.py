__all__ = ["CodebookUsageCallback"]

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from returnn.datasets.util.vocabulary import Vocabulary
from returnn.forward_iface import ForwardCallbackIface
from returnn.tensor import TensorDict

_MODALITIES = ("audio", "text")


def _entropy_bits(p: np.ndarray) -> float:
    """Shannon entropy in bits of a (normalized) distribution, ignoring zero entries."""
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _perplexity(p: np.ndarray) -> float:
    """exp(H) in nats, i.e. the "effective number of entries" of a distribution."""
    return float(np.exp(_entropy_bits(p) * np.log(2.0)))


def _js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence in bits; 0 = identical, 1 = disjoint support."""
    m = 0.5 * (p + q)
    return 0.5 * (_kl_bits(p, m) + _kl_bits(q, m))


def _kl_bits(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float((p[mask] * np.log2(p[mask] / np.clip(q[mask], 1e-300, None))).sum())


def _mass_overlap(p: np.ndarray, q: np.ndarray) -> float:
    """sum_v min(p_v, q_v): the fraction of probability mass the two distributions share.

    Unlike a binary support count this does not saturate: with millions of frames and only a few
    hundred entries virtually every entry gets used at least once by both modalities, so support
    overlap is ~1 regardless of how differently the two modalities actually use the codebook.
    """
    return float(np.minimum(p, q).sum())


def _mutual_info_bits(joint_counts: np.ndarray) -> float:
    """Mutual information in bits of a 2D contingency count matrix."""
    total = joint_counts.sum()
    if total <= 0:
        return 0.0
    p = joint_counts.astype(np.float64) / total
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    mask = p > 0
    return float((p[mask] * np.log2(p[mask] / (px @ py)[mask])).sum())


def _support(counts: np.ndarray, threshold: int) -> np.ndarray:
    return counts >= threshold


def _mass_covering_support(p: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask of the smallest set of entries covering ``fraction`` of the mass."""
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order])
    keep = order[: int(np.searchsorted(cum, fraction) + 1)]
    mask = np.zeros_like(p, dtype=bool)
    mask[keep] = True
    return mask


def _distinct_at_sample_size(p: np.ndarray, num_samples: int) -> float:
    """
    Expected number of distinct outcomes when drawing ``num_samples`` frames from ``p``.

    Needed because "how many joint codes does this modality visit" grows with the number of frames,
    and audio has many more frames than text (sub-phone rate vs. one symbol per phoneme). Comparing
    the raw counts would attribute a purely combinatorial difference to the model.
    """
    if num_samples <= 0:
        return 0.0
    # E[#distinct] = sum_v (1 - (1-p_v)^n), exact under multinomial sampling.
    with np.errstate(under="ignore"):
        return float((1.0 - np.exp(num_samples * np.log1p(-np.clip(p, 0.0, 1.0 - 1e-15)))).sum())


class _ModalityStats:
    """Accumulators for one modality. Everything is counts, so memory is independent of the corpus."""

    def __init__(self, num_groups: int, num_vars: int, vocab_size: int, max_run_length: int):
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.vocab_size = vocab_size
        self.num_frames = 0
        self.num_seqs = 0
        # [G, V] per-group usage counts
        self.usage = np.zeros((num_groups, num_vars), dtype=np.int64)
        # [V, V] joint (group0, group1) counts -- only meaningful for G == 2
        self.joint = np.zeros((num_vars, num_vars), dtype=np.int64) if num_groups == 2 else None
        # [G, V, V] code bigram counts (consecutive frames within a sequence)
        self.transitions = np.zeros((num_groups, num_vars, num_vars), dtype=np.int64)
        # [G, V, L] code x encoder-input-symbol counts
        self.code_label = np.zeros((num_groups, num_vars, vocab_size), dtype=np.int64)
        # summed softmax over frames -> the *soft* usage distribution the diversity loss optimizes
        self.prob_sum = np.zeros((num_groups, num_vars), dtype=np.float64)
        self.conf_sum = np.zeros(num_groups, dtype=np.float64)
        self.cos_sum = 0.0
        self.run_hist = np.zeros(max_run_length + 2, dtype=np.int64)

    def add_seq(self, codes: np.ndarray, labels: np.ndarray, conf: np.ndarray, cos: np.ndarray, prob_sum) -> None:
        num_frames, num_groups = codes.shape
        self.num_frames += num_frames
        self.num_seqs += 1
        for g in range(num_groups):
            cg = codes[:, g]
            self.usage[g] += np.bincount(cg, minlength=self.num_vars)
            self.code_label[g] += np.bincount(
                cg * self.vocab_size + labels, minlength=self.num_vars * self.vocab_size
            ).reshape(self.num_vars, self.vocab_size)
            if num_frames > 1:
                self.transitions[g] += np.bincount(
                    cg[:-1] * self.num_vars + cg[1:], minlength=self.num_vars * self.num_vars
                ).reshape(self.num_vars, self.num_vars)
        if self.joint is not None:
            self.joint += np.bincount(
                codes[:, 0] * self.num_vars + codes[:, 1], minlength=self.num_vars * self.num_vars
            ).reshape(self.num_vars, self.num_vars)
        self.conf_sum += conf.sum(axis=0)
        self.cos_sum += float(cos.sum())
        if prob_sum is not None:
            self.prob_sum += prob_sum
        # run lengths of the (joint, if available) code identity
        ids = codes[:, 0] * self.num_vars + codes[:, 1] if self.num_groups == 2 else codes[:, 0]
        if ids.size:
            boundaries = np.flatnonzero(np.diff(ids)) + 1
            run_lengths = np.diff(np.concatenate(([0], boundaries, [ids.size])))
            np.add.at(self.run_hist, np.clip(run_lengths, 0, self.run_hist.size - 1), 1)

    @property
    def marginals(self) -> np.ndarray:
        """[G, V] normalized hard usage distributions."""
        return self.usage / max(self.num_frames, 1)

    @property
    def joint_probs(self) -> Optional[np.ndarray]:
        if self.joint is None:
            return None
        return self.joint / max(self.num_frames, 1)

    @property
    def soft_marginals(self) -> np.ndarray:
        """[G, V] normalized *soft* usage (mean softmax), i.e. what the diversity loss maximizes."""
        return self.prob_sum / max(self.num_frames, 1)


class CodebookUsageCallback(ForwardCallbackIface):
    """
    Forward callback for the codebook (GumbelVectorQuantizer) analysis.

    Consumes the per-frame outputs of the codebook ``forward_step`` and accumulates *counts* only
    (usage histograms, joint code distributions, code<->input-symbol contingency tables, code
    bigrams, run lengths), so memory is O(G*V + V^2 + V*L) and independent of the corpus size --
    the whole test set can be processed.

    Written to ``out_dir``:

    - ``summary.txt``          all scalar metrics, grouped by question,
    - ``codebook_stats.npz``   every count matrix, for ad-hoc follow-up analysis,
    - ``shared_codes.txt``     per group, the entries with the most *shared* mass, together with the
      phonemes (text side) and cluster ids (audio side) that use them,
    - plots (usage curves, joint code maps, code x label heatmaps, run lengths).

    The metrics are ordered as a ladder of increasingly strong evidence for a genuinely shared
    discrete space (see ``CODEBOOK.md``):

    1. marginal support overlap -- weakest; the per-modality diversity loss directly optimizes each
       modality toward uniform usage of all entries, so near-total overlap is expected and says
       nothing about alignment,
    2. marginal mass overlap -- same caveat,
    3. joint (group-tuple) overlap -- the first level that is not directly optimized: both modalities
       can cover every entry of every group while occupying disjoint regions of the product space,
    4. code<->symbol agreement -- the only level that tests whether a shared code *means* the same
       thing in both modalities.
    """

    def __init__(
        self,
        *,
        out_dir: str = "codebook",
        max_run_length: int = 64,
        top_shared_codes: int = 40,
        top_labels_per_code: int = 4,
        save_plots: bool = True,
        vocab: Optional[Union[str, Dict]] = None,  # injected by serialize_forward (the text vocab)
    ):
        self.out_dir = out_dir
        self.max_run_length = max_run_length
        self.top_shared_codes = top_shared_codes
        self.top_labels_per_code = top_labels_per_code
        self.save_plots = save_plots
        self.vocab_opts = vocab
        self.vocab: Optional[Vocabulary] = None

        self._stats: Dict[str, _ModalityStats] = {}

    def init(self, *args, **kwargs):
        os.makedirs(self.out_dir, exist_ok=True)
        if isinstance(self.vocab_opts, str):
            self.vocab = Vocabulary.create_vocab(vocab_file=self.vocab_opts, unknown_label=None)
        elif isinstance(self.vocab_opts, dict):
            self.vocab = Vocabulary.create_vocab(**self.vocab_opts)

    def _label_name(self, modality: str, idx: int, vocab_size: int) -> str:
        """Name of an encoder *input* symbol. The model reserves the top 3 ids for [mask, bos, eos]."""
        num_labels = vocab_size - 3
        if idx >= num_labels:
            return {num_labels: "<mask>", num_labels + 1: "<bos>", num_labels + 2: "<eos>"}.get(idx, f"<{idx}>")
        if modality == "text" and self.vocab is not None and idx < self.vocab.num_labels:
            return self.vocab.id_to_label(idx)
        return str(idx)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict, **kwargs):
        for modality in _MODALITIES:
            code_key = f"{modality}_codes"
            if code_key not in outputs.data:
                continue  # modality absent from this batch (CombinedDataset)
            codes_tensor = outputs[code_key]
            codes = np.asarray(codes_tensor.raw_tensor, dtype=np.int64)  # [T, G]
            if codes.ndim != 2 or codes.shape[0] == 0:
                continue
            labels_tensor = outputs[f"{modality}_labels"]
            labels = np.asarray(labels_tensor.raw_tensor, dtype=np.int64)  # [T]
            conf = np.asarray(outputs[f"{modality}_conf"].raw_tensor, dtype=np.float64)  # [T, G]
            cos = np.asarray(outputs[f"{modality}_cos"].raw_tensor, dtype=np.float64)  # [T]
            prob_sum_key = f"{modality}_prob_sum"
            prob_sum = (
                np.asarray(outputs[prob_sum_key].raw_tensor, dtype=np.float64)
                if prob_sum_key in outputs.data
                else None
            )

            stats = self._stats.get(modality)
            if stats is None:
                num_groups = codes.shape[1]
                num_vars = codes_tensor.sparse_dim.dimension
                vocab_size = labels_tensor.sparse_dim.dimension
                stats = _ModalityStats(num_groups, num_vars, vocab_size, self.max_run_length)
                self._stats[modality] = stats
            stats.add_seq(codes, labels, conf, cos, prob_sum)

    # -- reporting -------------------------------------------------------------------------------

    def finish(self, **kwargs):
        npz: Dict[str, np.ndarray] = {}
        for modality, stats in self._stats.items():
            npz[f"{modality}_usage"] = stats.usage
            npz[f"{modality}_code_label"] = stats.code_label
            npz[f"{modality}_transitions"] = stats.transitions
            npz[f"{modality}_run_hist"] = stats.run_hist
            npz[f"{modality}_prob_sum"] = stats.prob_sum
            if stats.joint is not None:
                npz[f"{modality}_joint"] = stats.joint
        np.savez_compressed(os.path.join(self.out_dir, "codebook_stats.npz"), **npz)

        with open(os.path.join(self.out_dir, "summary.txt"), "w") as out:
            self._write_summary(out)
        self._write_shared_codes()
        if self.save_plots:
            self._write_plots()

    def _write_summary(self, out) -> None:
        missing = [m for m in _MODALITIES if m not in self._stats]
        if missing:
            out.write(
                f"WARNING: no frames seen for modality/modalities {missing}. All cross-modal metrics"
                " below are skipped. The codebook analysis needs a dataset providing both the audio"
                " and the text key (e.g. the paired MetaDataset test set).\n\n"
            )
            print(f"WARNING: codebook analysis saw no frames for {missing}; cross-modal metrics skipped")

        out.write("=== per modality ===\n")
        for modality, stats in sorted(self._stats.items()):
            marg = stats.marginals
            soft = stats.soft_marginals
            out.write(f"[{modality}]\n")
            out.write(f"  num_seqs={stats.num_seqs} num_frames={stats.num_frames}\n")
            out.write(f"  num_groups={stats.num_groups} num_vars={stats.num_vars}\n")
            hard_ppl = [_perplexity(marg[g]) for g in range(stats.num_groups)]
            out.write(f"  hard_perplexity_per_group={_fmt(hard_ppl)} sum={sum(hard_ppl):.2f}\n")
            if stats.prob_sum.any():
                soft_ppl = [_perplexity(soft[g]) for g in range(stats.num_groups)]
                out.write(f"  soft_perplexity_per_group={_fmt(soft_ppl)} sum={sum(soft_ppl):.2f}\n")
                out.write(
                    "  # soft = what the diversity loss maximizes; hard = what the model actually uses.\n"
                    "  # soft >> hard means the loss is satisfied by spread-out soft probabilities while\n"
                    "  # the argmax has collapsed. NB not comparable to the training-log codebook_prob_ppl:\n"
                    "  # that is computed per batch and averaged over batches (and includes padding),\n"
                    "  # whereas this pools over the whole corpus, which is systematically higher.\n"
                )
            used = [int((marg[g] > 0).sum()) for g in range(stats.num_groups)]
            out.write(f"  used_entries_per_group={used}")
            out.write(f" of {stats.num_vars}\n")
            out.write(f"  mean_selection_confidence={_fmt(stats.conf_sum / max(stats.num_frames, 1))}\n")
            out.write(f"  mean_cosine(state, its code)={stats.cos_sum / max(stats.num_frames, 1):.4f}\n")
            out.write(
                "  # cosine ~1 -> the codebook is nearly an identity (the bottleneck does little);\n"
                "  # low cosine -> quantized frames carry a lot of replacement noise.\n"
            )
            if stats.joint is not None:
                jp = stats.joint_probs
                out.write(f"  joint_perplexity={_perplexity(jp.ravel()):.1f} of {stats.num_vars ** 2}\n")
                out.write(f"  distinct_joint_codes={int((stats.joint > 0).sum())}\n")
                mi = _mutual_info_bits(stats.joint)
                out.write(f"  mutual_info_between_groups={mi:.4f} bits\n")
                out.write(
                    "  # I(g0;g1) ~ 0 -> the groups are independent, so the joint is the product of the\n"
                    "  # marginals and joint disjointness is unreachable: the joint question below then\n"
                    "  # reduces to the marginal one. Substantial I -> the joint carries the signal.\n"
                )
            mean_run = _mean_run_length(stats.run_hist)
            out.write(f"  mean_code_run_length={mean_run:.2f} frames\n")
            out.write("\n")

        if len(self._stats) < 2:
            return

        audio, text = self._stats["audio"], self._stats["text"]

        out.write("=== (1,2) marginal sharing -- weak evidence, directly optimized by the diversity loss ===\n")
        assert audio.num_groups == text.num_groups and audio.num_vars == text.num_vars
        for g in range(audio.num_groups):
            pa, pt = audio.marginals[g], text.marginals[g]
            out.write(f"[group {g}]\n")
            for threshold in (1, 10, 100):
                sa, st = _support(audio.usage[g], threshold), _support(text.usage[g], threshold)
                out.write(
                    f"  support(count>={threshold}): both={int((sa & st).sum())}"
                    f" audio_only={int((sa & ~st).sum())} text_only={int((st & ~sa).sum())}"
                    f" unused={int((~sa & ~st).sum())}\n"
                )
            ma, mt = _mass_covering_support(pa, 0.95), _mass_covering_support(pt, 0.95)
            out.write(
                f"  core support (smallest set covering 95% of mass): audio={int(ma.sum())}"
                f" text={int(mt.sum())} shared={int((ma & mt).sum())}\n"
            )
            out.write(f"  mass_overlap=sum_v min(p_audio,p_text)={_mass_overlap(pa, pt):.4f}\n")
            out.write(f"  js_divergence={_js_divergence_bits(pa, pt):.4f} bits\n")
        out.write("\n")

        if audio.joint is not None and text.joint is not None:
            out.write("=== (3) joint (group-tuple) sharing -- first level not directly optimized ===\n")
            pa, pt = audio.joint_probs.ravel(), text.joint_probs.ravel()
            out.write(f"  mass_overlap={_mass_overlap(pa, pt):.4f}\n")
            out.write(f"  js_divergence={_js_divergence_bits(pa, pt):.4f} bits\n")
            sa, st = _support(audio.joint, 1), _support(text.joint, 1)
            out.write(
                f"  support(count>=1): both={int((sa & st).sum())} audio_only={int((sa & ~st).sum())}"
                f" text_only={int((st & ~sa).sum())} unused={int((~sa & ~st).sum())}\n"
            )
            num_matched = min(audio.num_frames, text.num_frames)
            out.write(
                f"  distinct joint codes at matched sample size (n={num_matched}):"
                f" audio={_distinct_at_sample_size(pa, num_matched):.1f}"
                f" text={_distinct_at_sample_size(pt, num_matched):.1f}\n"
            )
            out.write(
                "  # matched because the raw distinct-code count grows with the number of frames, and\n"
                f"  # audio has {audio.num_frames} frames vs. {text.num_frames} for text.\n"
            )
            # is the observed joint overlap more than what the marginals alone already force?
            indep_a = np.outer(audio.marginals[0], audio.marginals[1]).ravel()
            indep_t = np.outer(text.marginals[0], text.marginals[1]).ravel()
            out.write(f"  mass_overlap under within-modality group independence={_mass_overlap(indep_a, indep_t):.4f}\n")
            out.write(
                "  # compare with the observed joint mass_overlap above: a large drop means the two\n"
                "  # modalities couple their groups differently, i.e. disjoint regions of the product space.\n"
            )
            out.write("\n")

        out.write("=== (4) do shared codes mean the same thing? -- code <-> input symbol ===\n")
        for modality, stats in sorted(self._stats.items()):
            for g in range(stats.num_groups):
                table = stats.code_label[g]
                total = table.sum()
                if total == 0:
                    continue
                label_counts = table.sum(axis=0)
                h_label = _entropy_bits(label_counts / total)
                mi = _mutual_info_bits(table)
                purity = float(table.max(axis=1).sum()) / float(total)
                out.write(
                    f"  [{modality} group {g}] I(symbol;code)={mi:.4f} bits  H(symbol)={h_label:.4f} bits"
                    f"  NMI={mi / h_label if h_label > 0 else float('nan'):.4f}  code_purity={purity:.4f}\n"
                )
        out.write(
            "  # NMI ~1 -> a code essentially identifies the input symbol; ~0 -> codes encode context or\n"
            "  # position rather than symbol identity. See shared_codes.txt for the per-code breakdown.\n"
        )
        out.write("\n")
        out.write(
            "NOTE: even perfect joint overlap does not prove alignment -- audio and text could use the\n"
            "same code for different phones. shared_codes.txt is what shows whether a shared entry\n"
            "corresponds to the same underlying sound in both modalities.\n"
        )

    def _write_shared_codes(self) -> None:
        """Per group, the entries carrying the most *shared* mass + which symbols use them."""
        if len(self._stats) < 2:
            return
        audio, text = self._stats["audio"], self._stats["text"]
        with open(os.path.join(self.out_dir, "shared_codes.txt"), "w") as out:
            out.write(
                "Codebook entries ranked by shared mass min(p_audio, p_text), per group.\n"
                "For each entry: how much of each modality's mass it carries, and the input symbols\n"
                "(cluster ids for audio, phonemes for text) whose frames map to it.\n"
                "If the shared space is meaningful, an entry's audio clusters and text phonemes should\n"
                "correspond to the same underlying sound.\n\n"
            )
            for g in range(audio.num_groups):
                pa, pt = audio.marginals[g], text.marginals[g]
                shared = np.minimum(pa, pt)
                order = np.argsort(shared)[::-1][: self.top_shared_codes]
                out.write(f"=== group {g} (top {len(order)} of {audio.num_vars} entries) ===\n")
                out.write(f"{'entry':>6} {'p_audio':>9} {'p_text':>9} {'shared':>9}  audio clusters | text phonemes\n")
                for v in order:
                    audio_top = _top_labels(audio.code_label[g][v], self.top_labels_per_code)
                    text_top = _top_labels(text.code_label[g][v], self.top_labels_per_code)
                    audio_str = ", ".join(
                        f"{self._label_name('audio', i, audio.vocab_size)}:{f:.2f}" for i, f in audio_top
                    )
                    text_str = ", ".join(
                        f"{self._label_name('text', i, text.vocab_size)}:{f:.2f}" for i, f in text_top
                    )
                    out.write(
                        f"{int(v):>6} {pa[v]:>9.5f} {pt[v]:>9.5f} {shared[v]:>9.5f}"
                        f"  {audio_str or '-'} | {text_str or '-'}\n"
                    )
                out.write("\n")

    def _write_plots(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"audio": "tab:red", "text": "tab:blue"}

        # usage per entry + rank curves
        any_stats = next(iter(self._stats.values()))
        for g in range(any_stats.num_groups):
            fig, axes = plt.subplots(2, 1, figsize=(12, 6))
            for modality, stats in sorted(self._stats.items()):
                p = stats.marginals[g]
                axes[0].plot(p, color=colors[modality], lw=0.7, alpha=0.8, label=modality)
                axes[1].plot(np.sort(p)[::-1], color=colors[modality], lw=1.2, label=modality)
            axes[0].set_title(f"group {g}: usage per codebook entry")
            axes[0].set_xlabel("codebook entry")
            axes[0].set_ylabel("p(entry)")
            axes[0].legend()
            axes[1].set_title("usage, sorted per modality (rank curve)")
            axes[1].set_xlabel("rank")
            axes[1].set_yscale("log")
            axes[1].legend()
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, f"usage_g{g}.png"), dpi=120)
            plt.close(fig)

        # joint code maps + overlap (only meaningful for G == 2, where the joint is 2D and plottable)
        if all(s.joint is not None for s in self._stats.values()) and len(self._stats) == 2:
            audio, text = self._stats["audio"], self._stats["text"]
            for modality, stats in (("audio", audio), ("text", text)):
                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(np.log1p(stats.joint), cmap="viridis", aspect="auto")
                ax.set_title(f"{modality}: joint code usage log(1+count)")
                ax.set_xlabel("group 1 entry")
                ax.set_ylabel("group 0 entry")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(os.path.join(self.out_dir, f"joint_{modality}.png"), dpi=120)
                plt.close(fig)

            # RGB overlap: red = audio-only, blue = text-only, purple = used by both
            pa, pt = audio.joint_probs, text.joint_probs
            rgb = np.zeros(pa.shape + (3,), dtype=np.float64)
            rgb[..., 0] = _norm01(np.log1p(pa / max(pa.max(), 1e-12)))
            rgb[..., 2] = _norm01(np.log1p(pt / max(pt.max(), 1e-12)))
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.imshow(rgb, aspect="auto")
            ax.set_title("joint code usage: red=audio, blue=text, purple=shared")
            ax.set_xlabel("group 1 entry")
            ax.set_ylabel("group 0 entry")
            fig.tight_layout()
            fig.savefig(os.path.join(self.out_dir, "joint_overlap.png"), dpi=120)
            plt.close(fig)

        # code x input symbol heatmaps
        for modality, stats in sorted(self._stats.items()):
            for g in range(stats.num_groups):
                table = stats.code_label[g].astype(np.float64)
                row_sums = table.sum(axis=1, keepdims=True)
                cond = table / np.clip(row_sums, 1e-12, None)
                fig, ax = plt.subplots(figsize=(8, 10))
                im = ax.imshow(cond, cmap="magma", aspect="auto", vmin=0.0, vmax=1.0)
                ax.set_title(f"{modality} group {g}: p(input symbol | code)")
                ax.set_xlabel("input symbol")
                ax.set_ylabel("codebook entry")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                fig.savefig(os.path.join(self.out_dir, f"code_label_{modality}_g{g}.png"), dpi=120)
                plt.close(fig)

        # code run lengths
        fig, ax = plt.subplots(figsize=(8, 4))
        for modality, stats in sorted(self._stats.items()):
            hist = stats.run_hist.astype(np.float64)
            total = hist.sum()
            if total > 0:
                ax.plot(hist / total, color=colors[modality], label=modality)
        ax.set_title("code run length (consecutive frames with the same code)")
        ax.set_xlabel("run length [frames]")
        ax.set_ylabel("fraction of runs")
        ax.set_yscale("log")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, "run_length.png"), dpi=120)
        plt.close(fig)


def _norm01(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _top_labels(counts: np.ndarray, num: int) -> List[Tuple[int, float]]:
    total = counts.sum()
    if total == 0:
        return []
    order = np.argsort(counts)[::-1][:num]
    return [(int(i), float(counts[i]) / float(total)) for i in order if counts[i] > 0]


def _mean_run_length(hist: np.ndarray) -> float:
    total = hist.sum()
    if total == 0:
        return float("nan")
    return float((hist * np.arange(hist.size)).sum()) / float(total)


def _fmt(values) -> str:
    return "[" + ", ".join(f"{float(v):.3f}" for v in values) + "]"
