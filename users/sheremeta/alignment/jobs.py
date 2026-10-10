"""Label-free aligner-eval jobs: perturbation drift (1) + cross-source word-onset agreement (2).

Method 1 works on align_jsonl CTC stores directly. Method 2 first normalizes each source (a CTC
align store, or an MFA word store) to a word-onset store {seq_tag, words, onsets_s}, then compares,
so heterogeneous sources (MFA vs CTC aligner) line up by seq_tag + word index. Dropped utterances
are categorized (missing/count/validity/label failures) and dumped, never silently discarded.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from typing import Dict, List, Optional, Tuple

from sisyphus import Job, Task, tk


def _subset_of(seq_tag: str, subset_regex: Optional[str]) -> str:
    if not subset_regex:
        return "all"
    m = re.search(subset_regex, seq_tag)
    return m.group(1) if m else "all"


def _norm_label(word: str) -> str:
    # casefold + strip non-alphanumerics so MFA and SPM word spellings compare fairly
    return "".join(c for c in str(word).casefold() if c.isalnum())


def _load_onsets(store_path: str, *, unit: str, blank: int, frame_rate: float, sp):
    """
    Reads the onsets of an align_jsonl CTC store, per word or per token.

    :param store_path: the store
    :param unit: "word" or "token"
    :param blank: the blank index of the alignment
    :param frame_rate: alignment frames per second
    :param sp: the SentencePiece processor of the labels
    :return: per seq_tag the labels and their onsets in seconds
    """
    import numpy as np

    from i6_experiments.users.sheremeta.alignment.onsets import path_to_token_onsets, token_onsets_to_words

    out = {}
    with gzip.open(store_path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tok_onsets = path_to_token_onsets(rec["alignment"], blank)
            if unit == "token":
                labels = [sp.id_to_piece(int(t)) for _, t in tok_onsets]
                frames = [fr for fr, _ in tok_onsets]
            else:
                labels, frames = token_onsets_to_words(tok_onsets, sp)
            out[str(rec["seq_tag"])] = (labels, np.asarray(frames, dtype="float64") / frame_rate)
    return out


def _write_word_onset_store(path: str, data: Dict[str, Tuple[List[str], List[float]]]):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for seq_tag, (words, onsets) in data.items():
            f.write(json.dumps({"seq_tag": seq_tag, "words": list(words),
                                "onsets_s": [float(o) for o in onsets]}, separators=(",", ":")) + "\n")


def _load_word_onset_store(path: str):
    import numpy as np

    out = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["seq_tag"])] = (rec["words"], np.asarray(rec["onsets_s"], dtype="float64"))
    return out


def _write_skipped(path: str, records: List[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _json_safe(obj):
    # map non-finite floats (NaN/inf from empty or degenerate stats) to null so the dumped
    # summary stays valid JSON per RFC 8259
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


# normalizers: any alignment source -> word-onset store
class CtcAlignToWordOnsetsJob(Job):
    """align_jsonl CTC path store + spm -> word-onset store {seq_tag, words, onsets_s}."""

    __sis_version__ = 1

    def __init__(self, *, ctc_store: tk.Path, spm_file: tk.Path, vocab_size: int, frame_rate: float):
        super().__init__()
        self.ctc_store = ctc_store
        self.spm_file = spm_file
        self.vocab_size = int(vocab_size)
        self.frame_rate = float(frame_rate)
        self.out_store = self.output_path("word_onsets.jsonl.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        import sentencepiece as spm

        sp = spm.SentencePieceProcessor(model_file=self.spm_file.get_path())
        onsets = _load_onsets(self.ctc_store.get_path(), unit="word", blank=self.vocab_size,
                              frame_rate=self.frame_rate, sp=sp)
        _write_word_onset_store(self.out_store.get_path(), onsets)


class MfaStoreToWordOnsetsJob(Job):
    """
    MFA word store ([(word, start_s, end_s)] per seq_tag) -> word-onset store.

    ``store_format`` is "jsonl" (gzip ``{seq_tag, words}``) or "sqlite" (an ``alignments(seq_tag,
    words)`` table with a JSON words column). start/end are the seconds build_word_chunked expects,
    so ``time_scale`` stays 1.0 unless a store uses different units.
    """

    __sis_version__ = 1

    def __init__(self, *, mfa_store: tk.Path, store_format: str = "sqlite", time_scale: float = 1.0):
        super().__init__()
        assert store_format in ("jsonl", "sqlite"), store_format
        self.mfa_store = mfa_store
        self.store_format = store_format
        self.time_scale = float(time_scale)
        self.out_store = self.output_path("word_onsets.jsonl.gz")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1})

    def run(self):
        from i6_experiments.users.sheremeta.alignment.onsets import mfa_words_to_onsets

        out = {}
        if self.store_format == "jsonl":
            with gzip.open(self.mfa_store.get_path(), "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    out[str(rec["seq_tag"])] = mfa_words_to_onsets(rec["words"], time_scale=self.time_scale)
        else:
            import sqlite3

            conn = sqlite3.connect(f"file:{self.mfa_store.get_path()}?mode=ro&immutable=1", uri=True)
            for seq_tag, words_json in conn.execute("SELECT seq_tag, words FROM alignments"):
                out[str(seq_tag)] = mfa_words_to_onsets(json.loads(words_json), time_scale=self.time_scale)
            conn.close()
        _write_word_onset_store(self.out_store.get_path(), out)


# method 2: agreement over word-onset stores (source-agnostic)
class WordOnsetAgreementJob(Job):
    """
    Cross-source word-onset disagreement given normalized word-onset stores (no gold TS).

    Reports consensus concentration (MAD), outlier visibility (max deviation + range), per-source
    leave-one-out deviation when 3+ sources are given, failure taxonomy with rates, and an
    utterance-level bootstrap CI. Skipped utterances go to skipped.jsonl with their reason.
    """

    __sis_version__ = 4

    # normalized-label agreement below this fraction marks the utterance label_mismatch
    LABEL_AGREE_MIN = 0.8

    def __init__(self, *, stores: List[tk.Path], source_names: List[str],
                 subset_regex: Optional[str] = None, top_k_flag: int = 100,
                 expected_tags: Optional[tk.Path] = None):
        super().__init__()
        assert len(stores) >= 2 and len(stores) == len(source_names)
        self.stores = stores
        self.source_names = source_names
        self.subset_regex = subset_regex
        self.top_k_flag = int(top_k_flag)
        # independent evaluation-population manifest (one seq_tag per line): missing_output is
        # counted against it, not a result store, so an aligner's own drops still count as missing
        self.expected_tags = expected_tags
        self.out_summary = self.output_path("summary.json")
        self.out_report = self.output_path("report.txt")
        self.out_flagged = self.output_path("flagged.jsonl")
        self.out_skipped = self.output_path("skipped.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 2})

    def run(self):
        from collections import defaultdict

        from i6_experiments.users.sheremeta.alignment.metrics import (FailureCounts, aggregate_agreement,
                                                                     format_stats_table, onset_failure_reason)

        per_source = [_load_word_onset_store(s.get_path()) for s in self.stores]
        present = set().union(*[set(d) for d in per_source])
        # the evaluation population: the independent manifest when given, else every tag seen
        if self.expected_tags is not None:
            with open(self.expected_tags.get_path(), encoding="utf-8") as mf:
                expected = {line.strip() for line in mf if line.strip()}
        else:
            expected = present
        attempted = len(expected)

        failures = FailureCounts()
        skipped: List[dict] = []
        unexpected = 0
        for tag in sorted(present - expected):
            # present in some source but outside the evaluation population: report, do not fail
            unexpected += 1
            skipped.append({"seq_tag": tag, "reason": "unexpected_output",
                            "present_in": [n for n, d in zip(self.source_names, per_source) if tag in d]})
        per_utt, per_utt_sub = [], defaultdict(list)
        for tag in sorted(expected):
            if not all(tag in d for d in per_source):
                failures.add("missing_output")
                skipped.append({"seq_tag": tag, "reason": "missing_output",
                                "present_in": [n for n, d in zip(self.source_names, per_source) if tag in d]})
                continue
            labels_by_src = [d[tag][0] for d in per_source]
            onsets = [d[tag][1] for d in per_source]
            counts = [len(o) for o in onsets]
            if len(set(counts)) != 1:
                failures.add("count_mismatch")
                skipped.append({"seq_tag": tag, "reason": "count_mismatch",
                                "counts": dict(zip(self.source_names, counts))})
                continue
            reason = None
            for name, o in zip(self.source_names, onsets):
                reason = onset_failure_reason(o)
                if reason:
                    skipped.append({"seq_tag": tag, "reason": reason, "source": name})
                    break
            if reason:
                failures.add(reason)
                continue
            labels_kept = labels_by_src[0]
            orig_idx = None
            if counts[0]:
                norm = [[_norm_label(w) for w in ls] for ls in labels_by_src]
                col_match = [len(set(col)) == 1 for col in zip(*norm)]
                agree = sum(col_match) / counts[0]
                if agree < self.LABEL_AGREE_MIN:
                    failures.add("label_mismatch")
                    skipped.append({"seq_tag": tag, "reason": "label_mismatch",
                                    "label_agreement": round(agree, 3),
                                    "labels_sample": [ls[:5] for ls in labels_by_src]})
                    continue
                if not all(col_match):
                    # drop positions where sources label different words so mismatched onsets do
                    # not contaminate the stats; keep original indices so flags report true positions
                    keep = [i for i, m in enumerate(col_match) if m]
                    onsets = [o[keep] for o in onsets]
                    labels_kept = [labels_by_src[0][i] for i in keep]
                    orig_idx = keep
            rec = (tag, labels_kept, onsets, orig_idx)
            per_utt.append(rec)
            per_utt_sub[_subset_of(tag, self.subset_regex)].append(rec)

        res = aggregate_agreement(per_utt, top_k_flag=self.top_k_flag,
                                  precounted_failures=failures, attempted_utts=attempted)
        by_subset = {s: aggregate_agreement(p, top_k_flag=10) for s, p in sorted(per_utt_sub.items())}

        loo_rows = None
        if res.loo_per_source is not None:
            loo_rows = dict(zip(self.source_names, res.loo_per_source))

        with open(self.out_summary.get_path(), "w") as f:
            json.dump(_json_safe({
                "sources": self.source_names,
                "attempted_utts": res.attempted_utts,
                "n_utts": res.n_utts,
                "n_boundaries": res.n_boundaries,
                "unexpected_outputs": unexpected,
                "failures": res.failures.as_dict(),
                "failure_rate": res.failure_rate,
                "mad": res.mad.as_row(),
                "max_dev": res.max_dev.as_row(),
                "range": res.span.as_row(),
                "loo_per_source": {n: s.as_row() for n, s in loo_rows.items()} if loo_rows else None,
                "ci_ms": {k: list(v) for k, v in res.ci_ms.items()},
                "per_subset": {s: r.mad.as_row() for s, r in by_subset.items()},
            }), f, indent=1)

        with open(self.out_flagged.get_path(), "w") as f:
            for fb in res.flagged:
                f.write(json.dumps({"seq_tag": fb.utt_id, "index": fb.index, "label": fb.label,
                                    "mad_ms": fb.mad_ms, "max_ms": fb.max_ms,
                                    "range_ms": fb.range_ms, "onsets_s": fb.onsets_s}) + "\n")
        _write_skipped(self.out_skipped.get_path(), skipped)

        lines = [
            f"word-onset agreement  sources={self.source_names}",
            f"attempted={res.attempted_utts}  used={res.n_utts}  boundaries={res.n_boundaries}"
            f"  unexpected(outside population)={unexpected}",
            f"failures: {res.failures.as_dict()}  rate={100 * res.failure_rate:.2f}%",
            "",
            format_stats_table({"MAD (consensus)": res.mad, "MAX deviation": res.max_dev,
                                "RANGE": res.span}, value_label="disagreement (ms)"),
            "",
            f"95% CI (utt bootstrap): mad median [{res.ci_ms['mad_median'][0]:.1f}, "
            f"{res.ci_ms['mad_median'][1]:.1f}] ms, range median "
            f"[{res.ci_ms['range_median'][0]:.1f}, {res.ci_ms['range_median'][1]:.1f}] ms",
        ]
        if loo_rows:
            lines += ["", "per-source leave-one-out deviation from the others' consensus:",
                      format_stats_table(loo_rows, value_label="source")]
        if by_subset:
            lines += ["", "per subset (MAD):",
                      format_stats_table({s: r.mad for s, r in by_subset.items()},
                                         value_label="subset")]
        lines += ["", "most-disagreed boundaries by MAX deviation (bad-transcript / hard-onset candidates):"]
        for fb in res.flagged[:20]:
            onset_str = " ".join(f"{o:.2f}" for o in fb.onsets_s)
            lines.append(f"  max={fb.max_ms:7.1f}ms range={fb.range_ms:7.1f}ms "
                         f"{fb.utt_id[:28]:28s} '{fb.label[:18]:18s}' onsets[s]: {onset_str}")
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")


# method 1: perturbation consistency (CTC align stores, one aligner)
class PerturbationConsistencyJob(Job):
    """
    Boundary drift of one aligner under known audio transforms (no gold labels).

    Reports boundary-micro and utterance-macro stats per transform, per-transform failure rates
    (a perturbation changing the onset count is itself an instability event), an equal-weight
    mean over transforms, and an utterance-level bootstrap CI of the pooled median.
    """

    __sis_version__ = 3

    def __init__(self, *, stores: Dict[str, tk.Path], spm_file: tk.Path, vocab_size: int,
                 frame_rate: float, unit: str = "word", subset_regex: Optional[str] = None):
        super().__init__()
        assert "orig" in stores, "need the unperturbed 'orig' store as reference"
        self.stores = stores
        self.spm_file = spm_file
        self.blank = int(vocab_size)
        self.frame_rate = float(frame_rate)
        self.unit = unit
        self.subset_regex = subset_regex
        self.out_summary = self.output_path("summary.json")
        self.out_report = self.output_path("report.txt")
        self.out_skipped = self.output_path("skipped.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 8, "time": 2})

    def run(self):
        from collections import defaultdict

        import sentencepiece as spm

        from i6_experiments.users.sheremeta.alignment.metrics import (FailureCounts, aggregate_perturbation,
                                                                     format_stats_table, onset_failure_reason,
                                                                     perturbation_deviations_ms)
        from i6_experiments.users.sheremeta.alignment.transforms import transform_by_name

        sp = spm.SentencePieceProcessor(model_file=self.spm_file.get_path())

        def load(p):
            return _load_onsets(p.get_path(), unit=self.unit, blank=self.blank,
                                frame_rate=self.frame_rate, sp=sp)

        base = load(self.stores["orig"])
        per_pair = []
        per_pair_sub = defaultdict(list)
        failures_by_tr: Dict[str, FailureCounts] = {}
        attempted_by_tr: Dict[str, int] = {}
        skipped: List[dict] = []

        for name, store in self.stores.items():
            if name == "orig":
                continue
            tr = transform_by_name(name)
            fc = failures_by_tr.setdefault(name, FailureCounts())
            perturbed = load(store)
            # tags the perturbed store produced but the reference did not: a missing original
            # output, counted so the pair accounting stays symmetric
            extra = [t for t in perturbed if t not in base]
            for tag in extra:
                fc.add("missing_output")
                skipped.append({"seq_tag": tag, "transform": name, "reason": "missing_orig_output"})
            attempted_by_tr[name] = len(base) + len(extra)
            for tag, (_labels, base_on) in base.items():
                if tag not in perturbed:
                    fc.add("missing_output")
                    skipped.append({"seq_tag": tag, "transform": name, "reason": "missing_output"})
                    continue
                onsets = perturbed[tag][1]
                if onsets.shape != base_on.shape:
                    fc.add("count_mismatch")
                    skipped.append({"seq_tag": tag, "transform": name, "reason": "count_mismatch",
                                    "n_orig": int(base_on.size), "n_perturbed": int(onsets.size)})
                    continue
                reason = onset_failure_reason(base_on) or onset_failure_reason(onsets)
                if reason:
                    fc.add(reason)
                    skipped.append({"seq_tag": tag, "transform": name, "reason": reason})
                    continue
                dev = perturbation_deviations_ms(base_on, tr.phi_inv(onsets))
                per_pair.append((name, tag, dev))
                per_pair_sub[_subset_of(tag, self.subset_regex)].append((name, tag, dev))

        res = aggregate_perturbation(per_pair, n_utts=len(base),
                                     failures_by_transform=failures_by_tr,
                                     attempted_by_transform=attempted_by_tr)
        by_subset = {s: aggregate_perturbation(p, 0) for s, p in sorted(per_pair_sub.items())}

        with open(self.out_summary.get_path(), "w") as f:
            json.dump(_json_safe({
                "unit": self.unit,
                "n_utts": res.n_utts,
                "attempted_pairs": res.attempted_pairs,
                "failures": res.failures.as_dict(),
                "failure_rate": res.failure_rate,
                "per_transform": {
                    k: {
                        "boundary": t.boundary.as_row(),
                        "utterance": t.utterance.as_row(),
                        "attempted": t.attempted,
                        "failures": t.failures.as_dict(),
                        "failure_rate": t.failure_rate,
                    }
                    for k, t in res.per_transform.items()
                },
                "overall_boundary": res.overall_boundary.as_row(),
                "overall_utterance": res.overall_utterance.as_row(),
                "mean_of_transforms": res.mean_of_transforms,
                "overall_ci_ms": list(res.overall_ci_ms),
                "per_subset": {s: r.overall_boundary.as_row() for s, r in by_subset.items()},
            }), f, indent=1)
        _write_skipped(self.out_skipped.get_path(), skipped)

        fail_lines = [
            f"  {k:16s} attempted={t.attempted:6d}  failures={t.failures.total:5d} "
            f"({100 * t.failure_rate:.2f}%)  {t.failures.as_dict()}"
            for k, t in res.per_transform.items()
        ]
        lines = [
            f"perturbation consistency ({self.unit} onsets)  utts={res.n_utts}  "
            f"pair failure rate={100 * res.failure_rate:.2f}%",
            "",
            "boundary-level per transform:",
            format_stats_table({k: t.boundary for k, t in res.per_transform.items()},
                               value_label="transform"),
            "",
            "utterance-level per transform (stats over per-utt medians):",
            format_stats_table({k: t.utterance for k, t in res.per_transform.items()},
                               value_label="transform"),
            "",
            "failures per transform (count changes under perturbation are instability events):",
            *fail_lines,
            "",
            "overall:",
            format_stats_table({"boundaries pooled": res.overall_boundary,
                                "utterance medians": res.overall_utterance},
                               value_label="overall"),
            "equal-weight mean over transforms: "
            + ", ".join(f"{k}={v:.1f}" for k, v in res.mean_of_transforms.items()
                        if k.endswith("_ms"))
            + ", "
            + ", ".join(f"{k}={100 * v:.1f}%" for k, v in res.mean_of_transforms.items()
                        if k.startswith("within")),
            f"95% CI (utt bootstrap) of pooled median: [{res.overall_ci_ms[0]:.1f}, "
            f"{res.overall_ci_ms[1]:.1f}] ms",
        ]
        if by_subset:
            lines += ["", "per subset (boundary-level):",
                      format_stats_table({s: r.overall_boundary for s, r in by_subset.items()},
                                         value_label="subset")]
        with open(self.out_report.get_path(), "w") as f:
            f.write("\n".join(lines) + "\n")


class DatasetSeqTagsJob(Job):
    """
    Dump every seq_tag of a RETURNN dataset dict to a manifest (one tag per line), CPU-only.

    Gives WordOnsetAgreementJob an evaluation population independent of any aligner output, so an
    aligner that drops an utterance is counted as missing instead of shrinking the denominator.
    """

    __sis_version__ = 1

    def __init__(self, *, returnn_dataset: dict, returnn_root: tk.Path, iterate: bool = False):
        super().__init__()
        self.returnn_dataset = returnn_dataset
        self.returnn_root = returnn_root
        self.iterate = iterate
        self.out_seq_tags = self.output_path("seq_tags.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        import sys

        sys.path.insert(0, self.returnn_root.get_path())
        import i6_core.util as util
        from returnn.config import Config, set_global_config
        from returnn.datasets import init_dataset
        from returnn.log import log

        cfg = Config()
        set_global_config(cfg)
        cfg.typed_dict.setdefault("log_verbosity", 4)
        log.init_by_config(cfg)

        dataset_dict = util.instanciate_delayed(self.returnn_dataset)
        ds = init_dataset(dataset_dict)
        if self.iterate:
            # some datasets can only enumerate tags by walking seq order
            ds.init_seq_order(epoch=1)
            tags, i = [], 0
            while ds.is_less_than_num_seqs(i):
                ds.load_seqs(i, i + 1)
                tags.append(ds.get_tag(i))
                i += 1
        else:
            tags = list(ds.get_all_tags())
        with open(self.out_seq_tags.get_path(), "w", encoding="utf-8") as f:
            for t in tags:
                f.write(f"{t}\n")
