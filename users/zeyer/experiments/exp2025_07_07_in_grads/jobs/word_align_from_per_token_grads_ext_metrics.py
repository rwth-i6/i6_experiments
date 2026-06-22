"""Per-token alignment + extended metrics matching the table format used in
the standard forced-alignment literature.

Mirrors :class:`WordAlignFromPerTokenGradsJob` but emits, in addition to the
mean WBE in seconds, the metric family used in Rousso et al. 2024 and Huang
et al. 2024 for direct comparability:

- mean / median word-boundary error in ms,
- % of word boundaries within {10, 25, 50, 100} ms,
- F1 @ 20 ms collar.

The Aligner is run identically to :class:`WordAlignFromPerTokenGradsJob`;
boundaries are collapsed from per-token alignment via ``num_tokens_per_word``.

Forced mode only (hyp = ref). Open-recognition mode (hyp != ref) can be added
later via an optional ``hyps_text_dict`` arg + Levenshtein matching --
covered by the ``__sis_hash_exclude__`` entry so adding it later doesn't
invalidate existing forced-mode hashes.
"""

from typing import Optional, Any, Dict, List, Tuple
from sisyphus import Job, Task, tk
from i6_experiments.users.zeyer.external_models.huggingface import (
    set_hf_offline_mode,
    get_content_dir_from_hub_cache_dir,
)


class WordAlignFromPerTokenGradsExtMetricsJob(Job):
    """See module docstring."""

    __sis_version__ = 2  # word end now exclusive (Aligner max+1); was inclusive (1 frame early)
    __sis_hash_exclude__ = {"hyps_text_dict": None}

    def __init__(
        self,
        *,
        returnn_root: Optional[tk.Path] = None,
        grad_score_hdf: tk.Path,
        grad_score_key: str,
        dataset_dir: tk.Path,
        dataset_key: str,
        dataset_offset_factors: int,
        align_opts: Dict[str, Any],
        hyps_text_dict: Optional[tk.Path] = None,
    ):
        """
        :param hyps_text_dict: For future use (open-recognition mode). Currently
            must be None; will gain Levenshtein-matched WBE + identity-gated F1
            once wired. Default-None excluded from hash so existing forced-mode
            instances keep their hash when the param is added later.
        """
        super().__init__()
        self.returnn_root = returnn_root
        self.grad_score_hdf = grad_score_hdf
        self.grad_score_key = grad_score_key
        self.dataset_dir = dataset_dir
        self.dataset_key = dataset_key
        self.dataset_offset_factors = dataset_offset_factors
        self.align_opts = align_opts
        self.hyps_text_dict = hyps_text_dict
        assert hyps_text_dict is None, "open-mode not yet implemented"

        # Headline metrics, all scalars.
        self.out_wbe_s = self.output_var("wbe_s.txt")
        self.out_wbe_ms = self.output_var("wbe_ms.txt")
        self.out_median_wbe_ms = self.output_var("median_wbe_ms.txt")
        self.out_pct_within_10ms = self.output_var("pct_within_10ms.txt")
        self.out_pct_within_25ms = self.output_var("pct_within_25ms.txt")
        self.out_pct_within_50ms = self.output_var("pct_within_50ms.txt")
        self.out_pct_within_100ms = self.output_var("pct_within_100ms.txt")
        self.out_f1_20ms = self.output_var("f1_20ms.txt")

        # Per-seq word boundaries, TextDict-style.
        self.out_word_boundaries = self.output_path("word_boundaries.py.gz")
        # Full readable report.
        self.out_report = self.output_path("report.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 10, "time": 5})

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

        from datasets import load_dataset

        ds = load_dataset(get_content_dir_from_hub_cache_dir(self.dataset_dir))
        print(f"Dataset key={self.dataset_key}, num_seqs={len(ds[self.dataset_key])}")

        ds_hdf = HDFDataset([self.grad_score_hdf.get_path()])
        ds_hdf.initialize()
        ds_hdf.init_seq_order(epoch=1)

        aligner = Aligner(**self.align_opts)

        # Each boundary contributes one error value (per-boundary, not per-word).
        # forced mode: 2 boundaries per word; both predicted boundaries match
        # the corresponding ref boundary by position.
        boundary_errors_s: List[float] = []
        wbe_per_word_s: List[float] = []  # avg of start+end err per word, mean of these = our standard WBE
        word_boundaries: Dict[str, List[Tuple[str, float, float]]] = {}

        for seq_idx, data in enumerate(ds[self.dataset_key]):
            ds_hdf.load_seqs(seq_idx, seq_idx + 1)

            samplerate = data["audio"]["sampling_rate"]
            num_audio_samples = len(data["audio"]["array"])
            audio_len_secs = num_audio_samples / samplerate

            chunk_T = int(ds_hdf.get_data(seq_idx, "num_input_frames")[0, 0])
            chunk_nt = int(ds_hdf.get_data(seq_idx, "num_tokens")[0, 0])
            tpw = ds_hdf.get_data(seq_idx, "num_tokens_per_word").reshape(-1).astype(int)
            grad = ds_hdf.get_data(seq_idx, self.grad_score_key).reshape(chunk_nt, chunk_T)
            secs_per_tf = audio_len_secs / chunk_T

            token_se = aligner.align(grad)
            assert len(token_se) == chunk_nt

            # Collapse to per-word.
            align_word_se: List[Tuple[float, float]] = []
            cursor = 0
            for k in tpw:
                first = token_se[cursor]
                last = token_se[cursor + k - 1]
                align_word_se.append((first[0] * secs_per_tf, last[1] * secs_per_tf))
                cursor += k

            ref_words: List[str] = list(data["word_detail"]["utterance"])
            ref_starts = [t * self.dataset_offset_factors / samplerate for t in data["word_detail"]["start"]]
            ref_ends = [t * self.dataset_offset_factors / samplerate for t in data["word_detail"]["stop"]]
            assert len(align_word_se) == len(ref_words), f"forced: {len(align_word_se)=} vs {len(ref_words)=}"

            seq_tag = f"seq-{seq_idx}"
            word_boundaries[seq_tag] = [
                (ref_words[w], align_word_se[w][0], align_word_se[w][1]) for w in range(len(ref_words))
            ]

            for w in range(len(ref_words)):
                err_start = abs(ref_starts[w] - align_word_se[w][0])
                err_end = abs(ref_ends[w] - align_word_se[w][1])
                boundary_errors_s.append(err_start)
                boundary_errors_s.append(err_end)
                wbe_per_word_s.append((err_start + err_end) / 2)

        b_err_ms = np.array(boundary_errors_s) * 1000.0
        wbe_per_word_ms = np.array(wbe_per_word_s) * 1000.0
        # "Standard" WBE used elsewhere = mean of per-word-avg (i.e. mean of wbe_per_word).
        wbe_standard_s = float(wbe_per_word_ms.mean() / 1000.0)
        # Per-boundary mean / median.
        mean_b_err_ms = float(b_err_ms.mean())
        median_b_err_ms = float(np.median(b_err_ms))

        tolerances_ms = [10, 25, 50, 100]
        pct_within = {t: float((b_err_ms <= t).mean()) * 100.0 for t in tolerances_ms}

        # In forced mode each predicted boundary maps 1-to-1 to a ref boundary,
        # so precision = recall = TP / total. F1 @ 20 ms = % within 20 ms.
        n_b_within_20 = int((b_err_ms <= 20).sum())
        n_b_total = len(b_err_ms)
        precision_20 = n_b_within_20 / n_b_total if n_b_total else 0.0
        recall_20 = precision_20
        f1_20 = 2 * precision_20 * recall_20 / (precision_20 + recall_20) if (precision_20 + recall_20) > 0 else 0.0

        self.out_wbe_s.set(wbe_standard_s)
        self.out_wbe_ms.set(mean_b_err_ms)  # per-boundary mean (matches Rousso 2024 convention)
        self.out_median_wbe_ms.set(median_b_err_ms)
        self.out_pct_within_10ms.set(pct_within[10])
        self.out_pct_within_25ms.set(pct_within[25])
        self.out_pct_within_50ms.set(pct_within[50])
        self.out_pct_within_100ms.set(pct_within[100])
        self.out_f1_20ms.set(f1_20 * 100.0)  # in percent

        with util.uopen(self.out_word_boundaries.get_path(), "wt") as f:
            f.write("{\n")
            for tag, items in word_boundaries.items():
                f.write(f"  {tag!r}: [\n")
                for w, s, e in items:
                    f.write(f"    ({w!r}, {s:.4f}, {e:.4f}),\n")
                f.write("  ],\n")
            f.write("}\n")

        with open(self.out_report.get_path(), "w") as f:
            f.write(f"Total seqs: {len(ds[self.dataset_key])}\n")
            f.write(f"Total per-word WBE entries: {len(wbe_per_word_s)}\n")
            f.write(f"Total per-boundary entries: {n_b_total}\n")
            f.write("\n--- Per-word WBE (matches existing out_wbe in WordAlignFromPerTokenGradsJob) ---\n")
            f.write(f"  mean: {wbe_standard_s * 1000:.2f} ms ({wbe_standard_s:.6f} s)\n")
            f.write("\n--- Per-boundary error (Rousso 2024 convention) ---\n")
            f.write(f"  mean:   {mean_b_err_ms:.2f} ms\n")
            f.write(f"  median: {median_b_err_ms:.2f} ms\n")
            f.write(f"  % within  10 ms: {pct_within[10]:.2f}\n")
            f.write(f"  % within  25 ms: {pct_within[25]:.2f}\n")
            f.write(f"  % within  50 ms: {pct_within[50]:.2f}\n")
            f.write(f"  % within 100 ms: {pct_within[100]:.2f}\n")
            f.write(f"  F1 @ 20 ms collar: {f1_20 * 100:.2f}\n")
            f.write("\n--- Per-seq mean WBE distribution ---\n")
            for p in (1, 5, 25, 50, 75, 95, 99):
                f.write(f"  p{p:>2}: {np.percentile(wbe_per_word_ms, p):.2f} ms\n")
