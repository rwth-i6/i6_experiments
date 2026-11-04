"""Descriptive, paired error analysis; no recognition and no test-set selection.

Official score files define scoring-unit correctness and the selection oracle.
Reference-word attribution is restricted to identical normalized references:
compound-equivalence normalization may otherwise change the reference token grid.
"""

import gzip
import itertools
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path

from sisyphus import Job, Task

from . import controlled_comparison as cc
from .audio_provenance import check_audio_origins, audio_origins


NATIVE = {
    "Granite CTC": ("qwen-granite", "native"),
    "Parakeet RNN-T": ("qwen-parakeet-rnnt", "native"),
    "Speech Qwen 0.6B": ("qwen-speech-0.6b", "native"),
    "Speech Qwen 1.7B": ("qwen-speech-1.7b", "native"),
    "Cohere AED": ("cohere-aed", "native"),
}
BEAM = dict(NATIVE, **{
    "Parakeet RNN-T": ("qwen-parakeet-rnnt", "zero-beam8-thr3"),
    "Speech Qwen 0.6B": ("qwen-speech-0.6b", "native-beam4"),
    "Speech Qwen 1.7B": ("qwen-speech-1.7b", "native-beam4"),
})
TRANSFER_FUSION = (
    ("qwen-speech-0.6b", "granite-fusion"), ("qwen-speech-1.7b", "granite-fusion"),
    ("qwen-parakeet-rnnt", "speech-qwen-fast"),
    ("qwen-parakeet-rnnt-speech-1.7b", "speech-qwen-fast"),
)


def load_units(record):
    if record["normalizer"] != cc.NORMALIZER:
        raise ValueError("Complementarity requires the current common normalizer")
    if cc.sha256(record["score"]) != record["score_sha256"]:
        raise ValueError("Scoring evidence changed after snapshot")
    result = {}
    with open(record["segments"]) as src:
        for line in src:
            row = json.loads(line)
            if row["id"] in result:
                raise ValueError("Duplicate scoring-unit id")
            counts = row["counts"]
            if any(not isinstance(counts[k], int) or counts[k] < 0 for k in ("ins", "del", "sub", "total", "ref_len")):
                raise ValueError("Invalid nonnegative integer edit counts")
            if counts["total"] != counts["ins"] + counts["del"] + counts["sub"]:
                raise ValueError("Invalid edit counts")
            result[row["id"]] = row
    for field in ("ins", "del", "sub", "total", "ref_len"):
        if sum(r["counts"][field] for r in result.values()) != record["summary"][field]:
            raise ValueError("Segment/summary counts differ")
    if not result or record["summary"]["ref_len"] <= 0:
        raise ValueError("Empty scoring evidence")
    return result


def check_pair(a, b, ua, ub):
    if not a.get("reference_sha256") or a["reference_sha256"] != b.get("reference_sha256"):
        raise ValueError("Pair has different raw references")
    check_audio_origins(a, b)
    if set(ua) != set(ub):
        raise ValueError("Pair has different scoring-unit coverage")


def unit_category(a, b):
    ca, cb = a["counts"]["total"] == 0, b["counts"]["total"] == 0
    if ca and cb:
        return "both_correct"
    if ca:
        return "only_a_correct"
    if cb:
        return "only_b_correct"
    return "both_wrong_same_hypothesis" if a["hypothesis"] == b["hypothesis"] else "both_wrong_different_hypotheses"


def aligned_errors(row):
    """One deterministic minimum-edit alignment; insertion events stay separate."""
    return _aligned_errors(row["reference"], row["hypothesis"], row["counts"]["total"], row["counts"]["ref_len"])


@lru_cache(maxsize=4096)
def _aligned_errors(reference, hypothesis, expected_cost, expected_ref_len):
    from kaldialign import align
    ref, hyp = reference.split(), hypothesis.split()
    gap = "<@complementarity-gap@>"
    if gap in ref or gap in hyp:
        raise ValueError("Alignment gap token occurs in transcript")
    outcomes, insertions = [], Counter()
    for r, h in (zip(ref, hyp) if ref == hyp else align(ref, hyp, gap)):
        if r == gap:
            insertions[len(outcomes), h] += 1
        else:
            outcomes.append(("correct" if r == h else "deletion" if h == gap else "substitution", None if h == gap else h))
    cost = sum(kind != "correct" for kind, _ in outcomes) + sum(insertions.values())
    if len(outcomes) != len(ref) or len(ref) != expected_ref_len or cost != expected_cost:
        raise ValueError("Alignment disagrees with official minimum-edit count")
    return outcomes, insertions


def word_complementarity(a, b):
    if a["reference"] != b["reference"]:
        return Counter(word_units_excluded_reference_mismatch=1)
    aa, ai = aligned_errors(a)
    bb, bi = aligned_errors(b)
    result = Counter(word_units_included=1, reference_words=len(aa))
    for x, y in zip(aa, bb):
        ca, cb = x[0] == "correct", y[0] == "correct"
        key = "both_correct" if ca and cb else "only_a_correct" if ca else "only_b_correct" if cb else (
            "both_wrong_same_error" if x == y else "both_wrong_different_error")
        result["words_" + key] += 1
    result.update(shared_insertions=sum((ai & bi).values()), only_a_insertions=sum((ai - bi).values()),
                  only_b_insertions=sum((bi - ai).values()))
    return result


def error_changes(base, fusion):
    if base["reference"] != fusion["reference"]:
        return Counter(word_units_excluded_reference_mismatch=1)
    aa, ai = aligned_errors(base)
    ff, fi = aligned_errors(fusion)
    result = Counter(word_units_included=1, reference_words=len(aa))
    for a, f in zip(aa, ff):
        if a[0] != "correct" and f[0] == "correct":
            result["corrected_word_errors"] += 1
            result["corrected_" + a[0]] += 1
        elif a[0] == "correct" and f[0] != "correct":
            result["introduced_word_errors"] += 1
            result["introduced_" + f[0]] += 1
        elif a[0] != "correct" and f[0] != "correct":
            result["persistent_same_word_error" if a == f else "changed_but_still_wrong"] += 1
    result.update(removed_insertions=sum((ai - fi).values()), added_insertions=sum((fi - ai).values()))
    return result


def panel_records(records, protocol, panel):
    lookup = {(r["group"], r["condition"]): r for r in records if r["status"] == "complete"}
    controls, fusions, missing = {}, [], []
    if panel in {"native", "beam-controls"}:
        for model, key in (NATIVE if panel == "native" else BEAM).items():
            if key in lookup:
                controls[model] = lookup[key]
            else:
                missing.append(dict(model=model, group=key[0], condition=key[1], role="standalone"))
        for key in TRANSFER_FUSION:
            if key in lookup:
                fusions.append(lookup[key])
            else:
                missing.append(dict(group=key[0], condition=key[1], role="fixed-transfer fusion"))
    elif panel == "dev-selected":
        if not protocol["ready"]:
            return {}, [], [dict(role="protocol", reason="development selection still provisional")]
        for label, choice in protocol["selected"].items():
            if not label.endswith(" | fast"):
                continue
            model, auxiliary, _ = choice["family"].split(" | ")
            if auxiliary != "None" and not auxiliary.startswith("Speech Qwen"):
                continue  # Text-only LMs have no standalone ASR transcription.
            key = choice["group"], choice["condition"]
            if key not in lookup:
                missing.append(dict(family=choice["family"], group=key[0], condition=key[1], role="selected evaluation"))
            elif auxiliary == "None":
                controls[model] = lookup[key]
            else:
                fusions.append(lookup[key])
        # Cohere is an explicitly fixed native baseline, not test-selected.
        if ("cohere-aed", "native") in lookup:
            controls["Cohere AED"] = lookup["cohere-aed", "native"]
    else:
        raise ValueError("Unknown baseline panel")
    return controls, fusions, missing


def analyze_pair(a, b, ua, ub, fusions, *, unit_sink=None, example_sink=None, examples_per_category=3):
    check_pair(a, b, ua, ub)
    totals, words, oracle, examples = Counter(), Counter(), Counter(), Counter()
    fusion_stats = {label: dict(units=Counter(), words_vs_a=Counter(), words_vs_b=Counter(),
                               both_correct_words_broken=0, both_wrong_words_corrected=0,
                               joint_word_units_excluded=0) for label in fusions}
    for key in sorted(ua):
        x, y = ua[key], ub[key]
        category = unit_category(x, y)
        totals[category] += 1
        ex, ey = x["counts"]["total"], y["counts"]["total"]
        totals["a_strictly_better" if ex < ey else "b_strictly_better" if ey < ex else "equal_error_count"] += 1
        best = x if ex <= ey else y
        oracle.update({k: best["counts"][k] for k in ("ins", "del", "sub", "total", "ref_len")})
        words.update(word_complementarity(x, y))
        unit = dict(id=key, category=category, a_errors=ex, b_errors=ey, oracle_source="a" if ex <= ey else "b",
                    oracle_counts=best["counts"], fusions={})
        for label, (_, uf) in fusions.items():
            z = uf[key]
            ez = z["counts"]["total"]
            stats = fusion_stats[label]
            u = stats["units"]
            u["improved_vs_a" if ez < ex else "worsened_vs_a" if ez > ex else "tied_vs_a"] += 1
            u["improved_vs_b" if ez < ey else "worsened_vs_b" if ez > ey else "tied_vs_b"] += 1
            u["beats_selection_oracle" if ez < min(ex, ey) else "worse_than_selection_oracle" if ez > min(ex, ey)
              else "ties_selection_oracle"] += 1
            if ex == 0 and ey == 0 and ez > 0:
                u["both_correct_units_broken"] += 1
            if ex > 0 and ey > 0 and ez == 0:
                u["both_wrong_units_corrected"] += 1
            if category in {"only_a_correct", "only_b_correct"}:
                u[category + ("_recovered" if ez == 0 else "_missed")] += 1
            stats["words_vs_a"].update(error_changes(x, z))
            stats["words_vs_b"].update(error_changes(y, z))
            if x["reference"] == y["reference"] == z["reference"]:
                ax, _ = aligned_errors(x)
                by, _ = aligned_errors(y)
                fz, _ = aligned_errors(z)
                stats["both_correct_words_broken"] += sum(p[0] == q[0] == "correct" and r[0] != "correct"
                                                          for p, q, r in zip(ax, by, fz))
                stats["both_wrong_words_corrected"] += sum(p[0] != "correct" and q[0] != "correct" and r[0] == "correct"
                                                           for p, q, r in zip(ax, by, fz))
            else:
                stats["joint_word_units_excluded"] += 1
            unit["fusions"][label] = dict(errors=ez, delta_vs_a=ez - ex, delta_vs_b=ez - ey)
            example_key = label + ("/new-error-vs-a" if ex == 0 and ez else "/recovers-b-only" if ex and ey == 0 and ez == 0 else "")
            if example_key != label and example_sink and examples[example_key] < examples_per_category:
                example_sink(dict(unit, example_kind=example_key, reference=x["reference"][:1200],
                                  a=x["hypothesis"][:1200], b=y["hypothesis"][:1200], fusion=z["hypothesis"][:1200]))
                examples[example_key] += 1
        if example_sink and examples[category] < examples_per_category:
            example_sink(dict(unit, example_kind=category, reference=x["reference"][:1200],
                              a=x["hypothesis"][:1200], b=y["hypothesis"][:1200]))
            examples[category] += 1
        if unit_sink:
            unit_sink(unit)
    wa, wb = a["summary"]["wer_percent"], b["summary"]["wer_percent"]
    oracle_wer = 100 * oracle["total"] / oracle["ref_len"]
    return dict(a=a["model"], b=b["model"], a_wer=wa, b_wer=wb, n_units=len(ua), units=totals, words=words,
                selection_oracle=dict(counts=oracle, wer=oracle_wer, headroom_pp=min(wa, wb) - oracle_wer,
                    error_count_headroom=min(a["summary"]["total"], b["summary"]["total"]) - oracle["total"],
                    candidate_reference_denominators_differ=any(ua[k]["counts"]["ref_len"] != ub[k]["counts"]["ref_len"] for k in ua),
                    diagnostic_only=True, rule="Minimum edit count per scoring unit; ties choose A; selected hypothesis denominator"),
                fusions={label: dict(stats, wer=record["summary"]["wer_percent"],
                                     delta_vs_better_standalone=record["summary"]["wer_percent"] - min(wa, wb),
                                     source=record["score"], condition=record["condition"], decoding=record["decoding"])
                         for label, (record, _) in fusions.items()},
                provenance=dict(a=a["score"], b=b["score"], a_segments_sha256=cc.sha256(a["segments"]),
                                b_segments_sha256=cc.sha256(b["segments"]), reference_sha256=a["reference_sha256"],
                                a_audio_origins=audio_origins(a), b_audio_origins=audio_origins(b)))


class ErrorComplementarityJob(Job):
    def __init__(self, dataset, snapshots, protocol, *, examples_per_category=3, code_version):
        self.dataset, self.snapshots, self.protocol = dataset, snapshots, protocol
        self.examples_per_category, self.code_version = examples_per_category, code_version
        self.out_summary = self.output_path("complementarity.json")
        self.out_units = self.output_path("units.jsonl.gz")
        self.out_examples = self.output_path("examples.jsonl")
        self.out_report = self.output_path("complementarity.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=12, time=8))

    def run(self):
        protocol = cc.read_json(self.protocol.get_path())
        records = []
        for snapshot in self.snapshots:
            records += [r for r in cc.read_json(snapshot.get_path())["records"] if r["dataset"] == self.dataset]
        # Later snapshots are explicit selected-evaluation evidence; they replace
        # matching export records, never choose by test WER.
        records = list({(r["group"], r["condition"]): r for r in records}.values())
        cache, panels = {}, {}
        def units(record):
            path = record["segments"]
            if path not in cache:
                cache[path] = load_units(record)
            return cache[path]
        with gzip.open(self.out_units.get_path(), "wt") as unit_dst, open(self.out_examples.get_path(), "w") as example_dst:
            for panel in ("native", "beam-controls", "dev-selected"):
                controls, fusion_records, missing = panel_records(records, protocol, panel)
                pairs = []
                for a_name, b_name in itertools.combinations(sorted(controls), 2):
                    a, b = controls[a_name], controls[b_name]
                    ua, ub = units(a), units(b)
                    fs = {}
                    for r in fusion_records:
                        if {r["model"], r["lm"]} != {a_name, b_name}:
                            continue
                        uf = units(r)
                        check_pair(a, r, ua, uf)
                        label = r["decoding"] + "/" + r["condition"]
                        fs[label] = r, uf
                    context = dict(dataset=self.dataset, panel=panel, a_model=a_name, b_model=b_name)
                    result = analyze_pair(a, b, ua, ub, fs,
                        unit_sink=lambda r: unit_dst.write(json.dumps(dict(context, **r)) + "\n"),
                        example_sink=lambda r: example_dst.write(json.dumps(dict(context, **r), ensure_ascii=False) + "\n"),
                        examples_per_category=self.examples_per_category)
                    pairs.append(result)
                panels[panel] = dict(pairs=pairs, missing=missing, controls={k: dict(group=v["group"], condition=v["condition"],
                    source=v["score"], normalizer=v["normalizer"], lexicon=v["lexicon"]) for k, v in controls.items()})
                print(f"{self.dataset}/{panel}: {len(pairs)} pairs; {len(missing)} missing conditions", flush=True)
        from importlib.metadata import version
        report = dict(dataset=self.dataset, normalizer=cc.NORMALIZER, protocol_sha256=cc.sha256(self.protocol.get_path()),
                      alignment_kaldialign_version=version("kaldialign"),
                      exploratory=True, selection_oracle_not_deployable=True, panels=panels)
        with open(self.out_summary.get_path(), "w") as dst:
            json.dump(report, dst, indent=2)
        with open(self.out_report.get_path(), "w") as dst:
            dst.write(render_dataset(report))


def render_dataset(summary):
    lines = ["# Error complementarity: " + summary["dataset"], "", NOTES, ""]
    for panel, results in summary["panels"].items():
        lines += ["## " + panel, "", "| A | B | A WER | B WER | Selection oracle | Headroom pp | Only A correct units | Only B correct units | Both wrong units |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for p in results["pairs"]:
            n, u = p["n_units"], p["units"]
            both_wrong = u.get("both_wrong_same_hypothesis", 0) + u.get("both_wrong_different_hypotheses", 0)
            lines.append(f"| {p['a']} | {p['b']} | {p['a_wer']:.3f} | {p['b_wer']:.3f} | {p['selection_oracle']['wer']:.3f} | "
                         f"{p['selection_oracle']['headroom_pp']:.3f} | {u.get('only_a_correct', 0)}/{n} | "
                         f"{u.get('only_b_correct', 0)}/{n} | {both_wrong}/{n} |")
        for p in results["pairs"]:
            if not p["fusions"]:
                continue
            lines += ["", f"### Fusion attribution: {p['a']} / {p['b']}", "",
                      "| Fusion | WER | Δ vs better standalone | Corrected / new word errors vs A | Corrected / new word errors vs B | Both-correct units broken | Word units excluded vs A / B |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for label, f in p["fusions"].items():
                a, b = f["words_vs_a"], f["words_vs_b"]
                lines.append(f"| {label} | {f['wer']:.3f} | {f['delta_vs_better_standalone']:+.3f} | "
                    f"{a.get('corrected_word_errors',0)} / {a.get('introduced_word_errors',0)} | "
                    f"{b.get('corrected_word_errors',0)} / {b.get('introduced_word_errors',0)} | "
                    f"{f['units'].get('both_correct_units_broken',0)} | "
                    f"{a.get('word_units_excluded_reference_mismatch',0)} / {b.get('word_units_excluded_reference_mismatch',0)} |")
        lines += ["", "Missing conditions (no fallback or test-best substitution):", ""]
        lines += ["- " + json.dumps(r, sort_keys=True) for r in results["missing"]] or ["None."]
        lines += [""]
    return "\n".join(lines) + "\n"


NOTES = (
    "Descriptive exploratory analysis; no significance claim or cross-corpus pooling. "
    "Native and beam-control panels are prespecified fixed configurations, not per-test winners. "
    "Dev-selected uses exact fast-protocol conditions only; Cohere is fixed native. "
    "Text-only LMs have no standalone ASR hypothesis and are not standalone pairs here. "
    "Units are official scoring units (recordings for grouped corpora), not necessarily utterances. "
    "The reference-dependent selection oracle picks the smaller edit count per unit, ties A, "
    "and retains that hypothesis's reference denominator. It is neither deployable nor a lower bound "
    "on fusion: fusion can combine corrections within a unit. It minimizes edit counts, not a global ratio; "
    "hypothesis-dependent reference lengths can produce tiny negative WER headroom despite nonnegative error-count headroom. "
    "Word attribution uses a deterministic minimum-edit alignment only when normalized references agree; "
    "excluded counts are explicit. Insertions are reported separately as boundary/token multisets with multiplicity. "
    "Alignment ties can change attribution. Example transcripts are clipped to 1200 characters per field."
)


class ComplementarityOverviewJob(Job):
    def __init__(self, summaries, *, code_version):
        self.summaries, self.code_version = summaries, code_version
        self.out_report = self.output_path("index.md")
        self.out_pairs = self.output_path("pairs.tsv")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import csv
        with open(self.out_report.get_path(), "w") as dst, open(self.out_pairs.get_path(), "w") as table:
            dst.write("# Model error complementarity\n\n" + NOTES + "\n\n")
            writer = csv.writer(table, delimiter="\t")
            writer.writerow(["dataset", "panel", "a", "b", "a_wer", "b_wer", "selection_oracle_wer", "headroom_pp", "n_units",
                             "only_a_correct", "only_b_correct", "both_wrong_same_hypothesis", "both_wrong_different_hypotheses",
                             "word_units_excluded", "a_source", "b_source"])
            for dataset, path in sorted(self.summaries.items()):
                summary = cc.read_json(path.get_path())
                dst.write(f"## {dataset}\n\n| Panel | Pair | Better standalone WER | Selection oracle WER | Headroom pp |\n|---|---|---:|---:|---:|\n")
                for panel, data in summary["panels"].items():
                    for p in data["pairs"]:
                        dst.write(f"| {panel} | {p['a']} / {p['b']} | {min(p['a_wer'],p['b_wer']):.3f} | "
                                  f"{p['selection_oracle']['wer']:.3f} | {p['selection_oracle']['headroom_pp']:.3f} |\n")
                        writer.writerow([dataset, panel, p["a"], p["b"], p["a_wer"], p["b_wer"], p["selection_oracle"]["wer"],
                            p["selection_oracle"]["headroom_pp"], p["n_units"],
                            *[p["units"].get(k,0) for k in ("only_a_correct", "only_b_correct", "both_wrong_same_hypothesis", "both_wrong_different_hypotheses")],
                            p["words"].get("word_units_excluded_reference_mismatch",0), p["provenance"]["a"], p["provenance"]["b"]])
                dst.write("\n")
