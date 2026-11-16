"""Consolidate completed, frozen public-study evidence without constructing ASR jobs.

Run as a module with --output-root and --destination to refresh the checked-in
snapshot. The Sisyphus launcher uses the same renderer. No test-set selection or
aggregation across overlapping corpora is performed here.
"""

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def validate(comparison, protocol, protocol_sha):
    """Reject stale/partial evidence and wrong constituent baselines before publishing."""
    if not protocol.get("ready") or comparison["protocol_sha256"] != protocol_sha:
        raise ValueError("Comparison does not match the ready frozen protocol")
    expected = {v["family"] for k, v in protocol["selected"].items() if k.endswith(" | fast")}
    normalizers = set()
    for key, panel in comparison["comparisons"].items():
        if not key.startswith("fast/"):
            raise ValueError("Consolidation accepts the frozen fast protocol only")
        rows = panel["rows"]
        if len(rows) != len(expected) or {r["family"] for r in rows} != expected:
            raise ValueError(f"Incomplete or duplicate families: {key}")
        standalone = {r["model"]: r for r in rows if r["lm"] == "None"}
        for row in rows:
            chosen = protocol["selected"][row["family"] + " | fast"]
            if row["status"] != "complete" or row.get("missing_baseline"):
                raise ValueError(f"Incomplete evaluation: {key}, {row['family']}")
            if (row["group"], row["condition"]) != (chosen["group"], chosen["condition"]):
                raise ValueError(f"Evaluation condition was not frozen on dev: {row['family']}")
            if row["dataset"] != panel["dataset"]:
                raise ValueError("Mixed datasets in comparison panel")
            normalizers.add(row["normalizer"])
            counts = row["summary"]
            computed = 100 * (counts["ins"] + counts["del"] + counts["sub"]) / counts["ref_len"]
            if not math.isclose(computed, row["wer"], abs_tol=1e-8):
                raise ValueError("WER and edit counts disagree")
            if row["lm"] != "None":
                components = [standalone[row["model"]]]
                if row["lm"].startswith("Speech Qwen "):
                    components.append(standalone[row["lm"]])
                indices = row.get("baseline_indices", [])
                if {rows[i]["family"] for i in indices} != {r["family"] for r in components}:
                    raise ValueError("Wrong standalone constituents (including Qwen size)")
                delta = row["wer"] - min(r["wer"] for r in components)
                if not math.isclose(delta, row["delta"], abs_tol=1e-8):
                    raise ValueError("Delta is not against the better standalone constituent")
    if len(normalizers) != 1:
        raise ValueError("Scoring normalizers differ")


def short_family(family):
    model, lm, method = family.split(" | ")
    if lm == "None":
        return {"Granite CTC": "Granite", "Parakeet RNN-T": "Parakeet"}.get(model, model)
    return lm.replace(" Qwen ", " ") + (" / 1p" if method == "one-pass" else " / 2p")


def dataset_name(dataset):
    names = {"ami-cleaned": "AMI", "earnings22-cleaned": "Earnings22", "gigaspeech-cleaned": "GigaSpeech",
             "librispeech-test-clean": "LibriSpeech clean", "librispeech-test-other": "LibriSpeech other",
             "loquacious-test": "Loquacious", "monsoon-en-in": "Monsoon", "spgispeech": "SPGISpeech",
             "voxpopuli-cleaned": "VoxPopuli"}
    return names.get(dataset, dataset.removeprefix("loquacious-test."))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |"] +
                     ["| " + " | ".join(map(str, row)) + " |" for row in rows]) + "\n"


def render_tables(comparison):
    panels = sorted(comparison["comparisons"].values(), key=lambda p: p["dataset"])
    families = [r["family"] for r in panels[0]["rows"]]
    groups = [("Standalone controls", [f for f in families if " | None | " in f]),
              ("Granite fusion", [f for f in families if f.startswith("Granite CTC | ") and " | None | " not in f]),
              ("Parakeet fusion", [f for f in families if f.startswith("Parakeet RNN-T | ") and " | None | " not in f])]
    text = ["# Frozen public evaluation: WER (%)\n",
            "All settings were selected jointly on LibriSpeech dev-other and Loquacious dev.short. "
            "These are fixed operating points, not per-test-set winners. 1p = one-pass delayed fusion; "
            "2p = n-best rescoring. Speech/Text identify the Qwen model type; sizes are explicit.\n",
            "Granite is the selected AM-only open-tree first-pass control, not native greedy decoding. "
            "Parakeet is the selected AM-only RASR beam control. Speech-Qwen standalone uses beam 8, "
            "length penalty 0. The Granite two-pass first pass uses an extended LibriSpeech lexicon; "
            "one-pass fusion is lexicon-free. Parakeet fusion is lexicon-free.\n"]
    for title, subset in [("Main corpora", False), ("Loquacious source subsets", True)]:
        selected = [p for p in panels if p["dataset"].startswith("loquacious-test.") == subset]
        text.append(f"## {title}\n")
        if subset:
            text.append("These overlap the Loquacious total; its LibriSpeech subset also overlaps "
                        "the separate LibriSpeech evaluation. Do not pool them.\n")
        for heading, columns in groups:
            rows = []
            for p in selected:
                lookup = {r["family"]: r for r in p["rows"]}
                rows.append([dataset_name(p["dataset"])] + [f"{lookup[f]['wer']:.3f}" for f in columns])
            text.extend([f"### {heading}\n", table(["Test set"] + [short_family(f) for f in columns], rows)])
    return "\n".join(text)


def render_metrics(comparison):
    text = ["# Gains, uncertainty and runtime\n",
            "Δ is fusion WER minus the better standalone constituent's WER, in percentage points "
            "(negative is better). Speech-Qwen baselines match the auxiliary model size. Text-Qwen "
            "has only the AM baseline. Intervals are the existing 10,000-replicate paired source-cluster "
            "bootstrap intervals; unadjusted and exploratory.\n",
            "RTF is recorded recognition/rescoring time divided by audio duration. Historical hardware "
            "is not matched. Two-pass timing includes generation and auxiliary scoring, including the "
            "independent one-best parity decode where recorded; it excludes downloads and evaluation. "
            "Loquacious subset runtimes come from their parent run. The RTF <1.5 selection constraint "
            "applies to development data, not every test corpus.\n"]
    for panel in sorted(comparison["comparisons"].values(), key=lambda p: p["dataset"]):
        text.append(f"## {dataset_name(panel['dataset'])}\n")
        text.append(f"{panel['n_scoring_units']:,} scoring units; {panel['n_clusters']:,} source clusters.\n")
        if panel.get("uncertainty_warning"):
            text.append(str(panel["uncertainty_warning"]) + "\n")
        rows = []
        for r in panel["rows"]:
            label = short_family(r["family"])
            if r["lm"] != "None":
                label = ("Granite + " if r["model"] == "Granite CTC" else "Parakeet + ") + label
            ci = r.get("delta_ci")
            timing = r.get("timing") or {}
            rtf = timing.get("rtf")
            rows.append([label, f"{r['wer']:.3f}", f"{r['delta']:+.3f}" if ci else "—",
                         f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "—",
                         f"{rtf:.3f}" + (" (parent)" if r.get("runtime_is_parent") else "") if rtf is not None else "—"])
        text.append(table(["Condition", "WER", "Δ", "95% CI for Δ", "RTF"], rows))
    return "\n".join(text)


def render_diagnostics(evidence):
    text = ["# Completed development diagnostics\n",
            "These are development-set diagnostics, not held-out performance or deployable oracle systems. "
            "They describe candidate availability and final ranking, not within-word pruning.\n"]
    for dataset in sorted(evidence):
        parts = evidence[dataset]
        if "ranking" not in parts:
            continue
        text.append(f"## {dataset}\n")
        pairs = parts["complementarity"]["panels"]["dev-selected"]["pairs"]
        text.append("### Standalone complementarity\n")
        text.append(table(["Pair", "AM WER", "Speech WER", "Unit-selection oracle WER"], [
            [p["a"] + " + " + p["b"], f"{p['a_wer']:.3f}", f"{p['b_wer']:.3f}",
             f"{p['selection_oracle']['wer']:.3f}"]
            for p in pairs if p["a"] in ("Granite CTC", "Parakeet RNN-T") and p["b"] == "Speech Qwen 1.7B"]))
        text.append("The oracle selects the better standalone hypothesis per scoring unit using the reference. "
                    "It is neither executable recognition nor a lower bound on arbitrary fusion.\n")
        text.append("### Candidate availability versus ranking\n")
        rows = []
        for r in parts["ranking"]["results"]:
            if "Speech Qwen 1.7B" not in r["family"]:
                continue
            scores, counts = r["scores"], r["counters"]
            rows.append([r["family"].split(" | ")[0], f"{scores['first']['wer']:.3f}",
                         f"{scores['selected']['wer']:.3f}", f"{scores['oracle']['wer']:.3f}",
                         counts["coverage_gap_vs_qwen"], counts["ranking_gap_vs_qwen"],
                         counts["ranking_missed_correct"], f"{counts['unique_candidates']/counts['units']:.1f}"])
        text.append(table(["AM (+ Speech 1.7B)", "First", "Rescored", "Pool oracle", "Coverage gaps",
                           "Ranking gaps", "Missed correct", "Mean distinct candidates"], rows))
        text.append("Coverage gaps: the pool oracle has more edits than standalone Speech-Qwen. "
                    "Ranking gaps: the pool can match or beat Speech-Qwen, but rescoring chooses a worse output. "
                    "Missed correct: a reference-correct candidate exists but is not selected. Counts are scoring "
                    "units, not word errors. A nominal 50-best cap does not imply 50 distinct candidates. "
                    "The n-best first-pass rows are controls of their own candidate generators; they can differ "
                    "from the independently selected standalone controls above.\n")
        text.append("### Reference-word accuracy by LibriSpeech LM-text frequency\n")
        rows = []
        for r in parts["stratified"]["panels"]["dev-selected"]["systems"]:
            if r["model"] not in ("Granite CTC", "Parakeet RNN-T"):
                continue
            if r["auxiliary"] not in ("None", "Speech Qwen 1.7B"):
                continue
            u, c = r["words"]["frequency/unseen"], r["words"]["frequency/>100"]
            rows.append([r["model"] + (" control" if r["auxiliary"] == "None" else " + Speech 1.7B " + r["method"]),
                         u["reference_words"], f"{u['accuracy']:.2f}", c["reference_words"], f"{c['accuracy']:.2f}"])
        text.append(table(["Condition", "Unseen N", "Unseen acc. %", "Frequent N", "Frequent acc. %"], rows))
        panel = parts["stratified"]["panels"]["dev-selected"]
        text.append(f"Word-alignment units: {panel.get('word_units_included', 'see source')} included, "
                    f"{panel.get('word_units_excluded_reference_mismatch', 'see source')} excluded for "
                    "reference-normalization mismatch. Insertions are separate from reference-word accuracy. "
                    "Unseen means absent from the LibriSpeech LM training text, not necessarily OOV to a "
                    "decoder or unseen in pretraining; it is not a proper-name annotation.\n")
    return "\n".join(text)


def render_attribution(evidence):
    text = ["# Corrected held-out error attribution\n",
            "Selected Parakeet + Speech-Qwen 1.7B; counts come from the refreshed multi-fusion "
            "analysis. Fixed/new counts are corrected/introduced reference-word errors. Insertions "
            "are separate. Units whose normalized references disagree are excluded pairwise; therefore "
            "these counts do not reconstruct corpus WER. No totals are pooled across corpora/subsets.\n"]
    for baseline, side in [("Parakeet", "a"), ("Speech-Qwen 1.7B", "b")]:
        text.append(f"## Relative to {baseline}\n")
        rows = []
        for dataset, parts in sorted(evidence.items()):
            if "ranking" in parts:
                continue
            pairs = parts["complementarity"]["panels"]["dev-selected"]["pairs"]
            pair = next(p for p in pairs if p["a"] == "Parakeet RNN-T" and p["b"] == "Speech Qwen 1.7B")
            for label, r in sorted(pair["fusions"].items()):
                w = r["words_vs_" + side]
                rows.append([dataset_name(dataset), "1p" if label.startswith("1-pass") else "2p",
                             w.get("corrected_word_errors", 0), w.get("introduced_word_errors", 0),
                             w.get("removed_insertions", 0), w.get("added_insertions", 0),
                             w.get("word_units_included", 0), w.get("word_units_excluded_reference_mismatch", 0)])
        text.append(table(["Test set", "Method", "Fixed", "New", "Removed ins.", "Added ins.",
                           "Units used", "Excluded"], rows))
    text.append("Earnings22 has very few comparable recording-level units; do not infer stable word-error "
                "patterns from its attribution counts. Loquacious total and source rows overlap.\n")
    return "\n".join(text)


def consolidate(comparison_file, protocol_file, diagnostic_files, destination):
    comparison, protocol = read(comparison_file), read(protocol_file)
    protocol_sha = digest(protocol_file)
    validate(comparison, protocol, protocol_sha)
    evidence = {}
    sources = {"comparison": comparison_file, "protocol": protocol_file}
    for dataset, files in diagnostic_files.items():
        evidence[dataset] = {}
        for kind, path in files.items():
            item = read(path)
            if item["protocol_sha256"] != protocol_sha or item["dataset"] != dataset:
                raise ValueError(f"Diagnostic protocol/dataset mismatch: {path}")
            normalizer = next(iter(comparison["comparisons"].values()))["rows"][0]["normalizer"]
            if item["normalizer"] != normalizer:
                raise ValueError(f"Diagnostic scoring normalizer mismatch: {path}")
            evidence[dataset][kind] = item
            sources[dataset + "/" + kind] = path
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in [("tables.md", render_tables(comparison)), ("metrics.md", render_metrics(comparison)),
                          ("diagnostics.md", render_diagnostics(evidence)), ("attribution.md", render_attribution(evidence))]:
        (destination / name).write_text(content, encoding="utf-8")
    fields = ["dataset", "family", "condition", "wer", "ins", "del", "sub", "ref_len", "delta_pp",
              "delta_ci_low", "delta_ci_high", "rtf", "runtime_is_parent", "meets_eval_budget", "baselines", "score", "hypotheses"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for p in sorted(comparison["comparisons"].values(), key=lambda p: p["dataset"]):
        for r in p["rows"]:
            ci = r.get("delta_ci", ["", ""])
            writer.writerow(dict(dataset=p["dataset"], family=r["family"], condition=r["condition"], wer=r["wer"],
                                 **{k: r["summary"][k] for k in ("ins", "del", "sub", "ref_len")}, delta_pp=r.get("delta", ""),
                                 delta_ci_low=ci[0], delta_ci_high=ci[1], rtf=(r.get("timing") or {}).get("rtf", ""),
                                 runtime_is_parent=r.get("runtime_is_parent", False), meets_eval_budget=r.get("meets_eval_budget"),
                                 baselines="; ".join(p["rows"][i]["family"] for i in r.get("baseline_indices", [])),
                                 score=r["score"], hypotheses=r["hypotheses"]))
    (destination / "results.tsv").write_text(stream.getvalue())
    manifest = dict(protocol_sha256=protocol_sha, seed=comparison["seed"], bootstrap_samples=comparison["bootstrap_samples"],
                    cells=sum(len(p["rows"]) for p in comparison["comparisons"].values()),
                    renderer_sha256=digest(__file__),
                    sources={k: dict(path=str(Path(p).resolve()), sha256=digest(p)) for k, p in sources.items()},
                    outputs={name: digest(destination / name) for name in ("tables.md", "metrics.md", "diagnostics.md", "attribution.md", "results.tsv")})
    (destination / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


def input_files(output_root):
    root = Path(output_root).resolve()
    protocol = root / "controlled-comparison/selection.json"
    comparison = root / "controlled-evaluation" / digest(protocol)[:16] / "comparison.json"
    diagnostics = {dataset: {
        "complementarity": root / "error-complementarity" / dataset / "summary.json",
        "ranking": root / "candidate-ranking" / dataset / "summary.json",
        "stratified": root / "stratified-errors" / dataset / "summary.json",
    } for dataset in ("librispeech-dev-other", "loquacious-dev.short")}
    for panel in read(comparison)["comparisons"].values():
        dataset = panel["dataset"]
        diagnostics[dataset] = {"complementarity": comparison.parent / "error-complementarity" / dataset / "summary.json"}
    return comparison, protocol, diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    consolidate(*input_files(args.output_root), args.destination)
