"""ID-based, reusable scoring of native and RASR benchmark hypotheses.

Uses our updated HF normalizer, compound equivalence, and kaldialign. Historical
leaderboard numbers can use the older normalizer described in hf_normalizer.
No timing heuristics, CTM reconstruction, or unconditional recording concatenation.
Only datasets explicitly carrying parent/chunk metadata are scored by session.
"""

import json
from collections import defaultdict
from importlib.metadata import version

from sisyphus import Job, Task

from .hf_scoring import normalize_compound_pair, normalize_hf_english


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def scoring_pairs(references, hypotheses):
    refs = {r["id"]: r for r in references}
    hyps = {r["id"]: r for r in hypotheses}
    if len(refs) != len(references) or len(hyps) != len(hypotheses):
        raise ValueError("Duplicate reference/hypothesis ids")
    if refs.keys() != hyps.keys():
        raise ValueError(
            f"Hypothesis coverage mismatch: missing={sorted(refs.keys() - hyps.keys())[:10]}, "
            f"extra={sorted(hyps.keys() - refs.keys())[:10]}"
        )
    groups = defaultdict(list)
    for row in references:
        groups[row["group_id"]].append(row)
    for group, chunks in sorted(groups.items()):
        chunks.sort(key=lambda x: x["chunk_index"])
        if len({c["chunk_index"] for c in chunks}) != len(chunks):
            raise ValueError(f"Duplicate chunk index in {group}")
        if len({c["reference"] for c in chunks}) != 1:
            raise ValueError(f"Inconsistent session reference in {group}")
        # Joining precedes normalization, exactly as for the official chunk scorer.
        ref = normalize_hf_english(chunks[0]["reference"])
        hyp = normalize_hf_english(" ".join(hyps[c["id"]]["text"] for c in chunks))
        ref, hyp = normalize_compound_pair(ref, hyp)
        yield dict(id=group, reference=ref, hypothesis=hyp)


class ScoreBenchmarkJob(Job):
    def __init__(self, references, hypotheses, *, code_version: str):
        self.references, self.hypotheses, self.code_version = references, hypotheses, code_version
        self.out_summary = self.output_path("summary.json")
        self.out_segments = self.output_path("segments.jsonl")
        self.out_wer = self.output_var("wer")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        from kaldialign import edit_distance

        totals = dict(ins=0, del_=0, sub=0, total=0, ref_len=0)
        n = 0
        with open(self.out_segments.get_path(), "w") as f:
            for pair in scoring_pairs(read_jsonl(self.references.get_path()), read_jsonl(self.hypotheses.get_path())):
                counts = edit_distance(pair["reference"].split(), pair["hypothesis"].split())
                for key in totals:
                    totals[key] += counts["del" if key == "del_" else key]
                f.write(json.dumps(dict(pair, counts=counts), ensure_ascii=False) + "\n")
                n += 1
        totals["del"] = totals.pop("del_")
        if not totals["ref_len"]:
            raise ValueError("No reference words to score")
        wer = 100 * totals["total"] / totals["ref_len"]
        with open(self.out_summary.get_path(), "w") as f:
            json.dump(
                dict(
                    totals,
                    wer_percent=wer,
                    num_scoring_units=n,
                    kaldialign_version=version("kaldialign"),
                    normalizer_revision="48219c6028db0517d704600d92f31edfc96e8c23",
                ),
                f,
                indent=2,
            )
        self.out_wer.set(wer)


class CompareBenchmarkHypothesesJob(Job):
    """Preserve raw/token and normalized differences; never claim parity from rounded WER."""

    def __init__(self, native, rasr, *, code_version: str):
        self.native, self.rasr, self.code_version = native, rasr, code_version
        self.out_report = self.output_path("comparison.json")
        self.out_differences = self.output_path("differences.jsonl")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        a, b = read_jsonl(self.native.get_path()), read_jsonl(self.rasr.get_path())
        aa, bb = {r["id"]: r for r in a}, {r["id"]: r for r in b}
        if len(aa) != len(a) or len(bb) != len(b) or aa.keys() != bb.keys():
            raise ValueError("Native/RASR ids differ")
        raw = normalized = tokens = 0
        with open(self.out_differences.get_path(), "w") as f:
            for key in aa:
                x, y = aa[key], bb[key]
                raw_diff = x["text"] != y["text"]
                norm_diff = normalize_hf_english(x["text"]) != normalize_hf_english(y["text"])
                token_diff = x.get("token_ids") != y.get("token_ids")
                raw += raw_diff
                normalized += norm_diff
                tokens += token_diff
                if raw_diff or token_diff:
                    f.write(
                        json.dumps(dict(id=key, native=x, rasr=y, normalized_diff=norm_diff), ensure_ascii=False) + "\n"
                    )
        with open(self.out_report.get_path(), "w") as f:
            json.dump(
                dict(
                    num_segments=len(a),
                    raw_differences=raw,
                    normalized_differences=normalized,
                    token_differences=tokens,
                ),
                f,
                indent=2,
            )


class BenchmarkSummaryJob(Job):
    def __init__(self, scores, expected_datasets, *, code_version: str):
        self.scores, self.expected_datasets, self.code_version = scores, tuple(expected_datasets), code_version
        self.out_report = self.output_path("results.json")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        result = {}
        for condition, datasets in self.scores.items():
            rows = {}
            for name, path in datasets.items():
                with open(path.get_path()) as f:
                    rows[name] = json.load(f)
            complete = set(rows) == set(self.expected_datasets)
            result[condition] = dict(
                datasets=rows,
                complete=complete,
                macro_wer_percent=sum(r["wer_percent"] for r in rows.values()) / len(rows) if complete else None,
            )
        with open(self.out_report.get_path(), "w") as f:
            json.dump(result, f, indent=2)
