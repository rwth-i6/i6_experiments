"""Compare completed CTC API runs by ID, token path, text, costs and exact WER."""

import json
import math
from sisyphus import Job, Task

from .benchmark_scoring import read_jsonl
from .hf_scoring import normalize_hf_english


def compare_runs(old, new, old_score, new_score):
    a, b = read_jsonl(old), read_jsonl(new)
    aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
    if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
        raise ValueError("CTC API comparison requires unique, identical segment IDs")
    counters = dict(raw_differences=0, normalized_differences=0, token_differences=0)
    max_am = max_lm = 0.0
    differences = []
    for key in aa:
        x, y = aa[key], bb[key]
        if "token_ids" not in x or "token_ids" not in y or not x.get("traceback") or not y.get("traceback"):
            raise ValueError("Missing token path or scored traceback")
        raw = x["text"] != y["text"]
        norm = normalize_hf_english(x["text"]) != normalize_hf_english(y["text"])
        tokens = x["token_ids"] != y["token_ids"]
        counters["raw_differences"] += raw
        counters["normalized_differences"] += norm
        counters["token_differences"] += tokens
        am_delta = float(y["traceback"][-1]["am_score"]) - float(x["traceback"][-1]["am_score"])
        lm_delta = float(y["traceback"][-1]["lm_score"]) - float(x["traceback"][-1]["lm_score"])
        if not math.isfinite(am_delta) or not math.isfinite(lm_delta):
            raise ValueError("Nonfinite cumulative scores in API comparison")
        if not tokens:
            max_am, max_lm = max(max_am, abs(am_delta)), max(max_lm, abs(lm_delta))
        if raw or tokens or abs(am_delta) > 1e-3 or abs(lm_delta) > 1e-3:
            differences.append(
                dict(
                    id=key, old=x, new=y, normalized_diff=norm, token_diff=tokens, am_delta=am_delta, lm_delta=lm_delta
                )
            )
    with open(old_score) as f:
        s0 = json.load(f)
    with open(new_score) as f:
        s1 = json.load(f)
    if s0["normalizer_revision"] != s1["normalizer_revision"]:
        raise ValueError("Normalizer mismatch in API comparison")
    score_equal = all(s0[k] == s1[k] for k in ("ins", "del", "sub", "total", "ref_len"))
    exact = not any(counters.values()) and score_equal and max(max_am, max_lm) <= 1e-3
    return (
        dict(
            num_segments=len(aa),
            **counters,
            old_score=s0,
            new_score=s1,
            equal_edit_counts=score_equal,
            max_am_delta_on_equal_paths=max_am,
            max_lm_delta_on_equal_paths=max_lm,
            matched_score_paths=len(aa) - counters["token_differences"],
            exact_parity=exact,
            verdict="parity" if exact else "review differences before pooling results",
        ),
        differences,
    )


class CompareCtcApisJob(Job):
    def __init__(self, old, new, old_score, new_score, *, code_version):
        self.old, self.new, self.old_score, self.new_score = old, new, old_score, new_score
        self.code_version = code_version
        self.out_report = self.output_path("comparison.json")
        self.out_differences = self.output_path("differences.jsonl")
        self.out_markdown = self.output_path("comparison.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=2))

    def run(self):
        report, differences = compare_runs(
            self.old.get_path(), self.new.get_path(), self.old_score.get_path(), self.new_score.get_path()
        )
        with open(self.out_report.get_path(), "w") as f:
            json.dump(report, f, indent=2)
        with open(self.out_differences.get_path(), "w") as f:
            for r in differences:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(self.out_markdown.get_path(), "w") as f:
            f.write("# Granite CTC + Speech-Qwen 1.7B: RASR API parity\n\n")
            f.write(f"Verdict: {report['verdict']}.\n\n")
            f.write(
                f"9804f5aa WER: {report['old_score']['wer_percent']:.6f}%; "
                f"d000386c WER: {report['new_score']['wer_percent']:.6f}%.\n\n"
            )
            f.write(
                f"{report['num_segments']} segments; {report['token_differences']} token-path differences; "
                f"{report['normalized_differences']} normalized-text differences.\n"
            )
