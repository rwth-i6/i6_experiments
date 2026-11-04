"""Steps 3/4: CPU-only coverage/ranking and fixed-condition stratification."""

import gzip
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from sisyphus import Job, Task
from i6_core.util import uopen

from . import controlled_comparison as cc
from . import error_complementarity as ec
from .hf_scoring import normalize_hf_english, normalize_compound_pair


def official_counts(reference, hypothesis):
    from kaldialign import edit_distance

    ref, hyp = normalize_compound_pair(normalize_hf_english(reference), normalize_hf_english(hypothesis))
    return edit_distance(ref.split(), hyp.split()), hyp


def diagnose_pool(reference, candidates, qwen, setting):
    """An augmented oracle is diagnostic only: Qwen has no fabricated AM cost."""
    from kaldialign import edit_distance

    if (
        setting["n"] < 1
        or setting["scale"] < 0
        or not all(math.isfinite(setting[k]) for k in ("scale", "word_penalty"))
    ):
        raise ValueError("Invalid frozen selection settings")
    if not candidates:
        raise ValueError("Empty candidate pool")
    rows = []
    for i, c in enumerate(candidates[: setting["n"]]):
        if (
            not all(math.isfinite(c[k]) for k in ("first_pass_cost", "am_cost", "lm_cost"))
            or abs(c["am_cost"] + c["lm_cost"] - c["first_pass_cost"]) > 0.01
        ):
            raise ValueError("Invalid cumulative first-pass costs")
        scores = c["speech"][setting["policy"]]
        if (
            not scores["token_costs"]
            or not math.isfinite(scores["nll"])
            or not all(math.isfinite(c) for c in scores["token_costs"])
            or abs(sum(scores["token_costs"]) - scores["nll"]) > 1e-3
        ):
            raise ValueError("Invalid cached Speech-Qwen NLL")
        counts, hyp = official_counts(reference, c["text"])
        rows.append(
            dict(
                index=i,
                counts=counts,
                normalized_hypothesis=hyp,
                first_pass=c["first_pass_cost"],
                am=c["am_cost"],
                retained_lm=c["lm_cost"],
                speech_nll=scores["nll"],
                eos_cost=scores["token_costs"][-1],
                word_cost=setting["word_penalty"] * len(c["text"].split()),
                auxiliary_total=setting["scale"] * scores["nll"] + setting["word_penalty"] * len(c["text"].split()),
                total=c["first_pass_cost"]
                + setting["scale"] * scores["nll"]
                + setting["word_penalty"] * len(c["text"].split()),
            )
        )
    first = min(rows, key=lambda r: r["first_pass"])
    selected = min(rows, key=lambda r: r["total"])
    speech_only = min(rows, key=lambda r: r["speech_nll"])
    auxiliary_choice = min(rows, key=lambda r: r["auxiliary_total"])
    oracle = min(rows, key=lambda r: r["counts"]["total"])
    qerrors = qwen["counts"]["total"]
    oracle_errors = oracle["counts"]["total"]
    qtext = qwen["hypothesis"]
    exact = any(r["normalized_hypothesis"] == qtext for r in rows)
    equivalent = any(
        edit_distance(*[s.split() for s in normalize_compound_pair(qtext, r["normalized_hypothesis"])])["total"] == 0
        for r in rows
    )
    gap = (
        "coverage_gap_vs_qwen"
        if oracle_errors > qerrors
        else "ranking_gap_vs_qwen" if selected["counts"]["total"] > qerrors else "no_gap_vs_qwen"
    )
    return dict(
        first=first,
        selected=selected,
        speech_only=speech_only,
        auxiliary_choice=auxiliary_choice,
        oracle=oracle,
        qwen_errors=qerrors,
        qwen_correct=qerrors == 0,
        first_wrong=first["counts"]["total"] > 0,
        normalized_qwen_in_pool=exact,
        compound_equivalent_qwen_in_pool=equivalent,
        pool_has_correct=oracle_errors == 0,
        ranking_missed_correct=oracle_errors == 0 and selected["counts"]["total"] > 0,
        first_pass_overturns_better_speech_choice=auxiliary_choice["counts"]["total"] < selected["counts"]["total"],
        word_reward_overturns_better_speech_choice=speech_only["counts"]["total"] < auxiliary_choice["counts"]["total"],
        gap=gap,
        unique_candidates=len({c["text"] for c in candidates[: setting["n"]]}),
        augmented_oracle_errors=min(oracle_errors, qerrors),
        augmented_oracle_diagnostic_only=True,
        augmented_candidate_am_score=None,
    )


class CandidateRankingDiagnosticsJob(Job):
    def __init__(self, dataset, dev_snapshot, protocol, *, code_version, examples_per_case=5):
        if dataset not in cc.DEV:
            raise ValueError("Candidate diagnostics are development-only")
        self.dataset, self.dev_snapshot, self.protocol = dataset, dev_snapshot, protocol
        self.code_version, self.examples_per_case = code_version, examples_per_case
        self.out_summary = self.output_path("ranking.json")
        self.out_units = self.output_path("units.jsonl.gz")
        self.out_examples = self.output_path("examples.jsonl")
        self.out_report = self.output_path("ranking.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=12, time=8))

    def run(self):
        protocol = cc.read_json(self.protocol.get_path())
        if not protocol["ready"]:
            raise ValueError("Finish development selection before ranking diagnostics")
        records = [r for r in cc.read_json(self.dev_snapshot.get_path())["records"] if r["dataset"] == self.dataset]
        lookup = {(r["group"], r["condition"]): r for r in records if r["status"] == "complete"}
        results = []
        with gzip.open(self.out_units.get_path(), "wt") as dst, open(self.out_examples.get_path(), "w") as examples:
            for label, choice in protocol["selected"].items():
                model, aux, method = choice["family"].split(" | ")
                if not label.endswith(" | fast") or method != "two-pass" or not aux.startswith("Speech Qwen"):
                    continue
                qchoice = protocol["selected"][aux + " | None | standalone | fast"]
                q = lookup[qchoice["group"], qchoice["condition"]]
                selected = lookup[choice["group"], choice["condition"]]
                qunits, sunits = ec.load_units(q), ec.load_units(selected)
                ec.check_pair(q, selected, qunits, sunits)
                stage = next(s for s in selected["stages"] if s["job_type"] == "SelectSpeechNbestJob")
                params = stage["parameters"]
                setting = {k: params[k] for k in ("n", "scale", "word_penalty", "policy")}
                refs = {}
                with open(selected["references"]) as src:
                    for line in src:
                        r = json.loads(line)
                        if r["group_id"] != r["id"] or r["chunk_index"] != 0 or r["id"] in refs:
                            raise ValueError("Development pool diagnostic requires independent unique units")
                        refs[r["id"]] = r["reference"]
                counters, totals, seen, example_counts = Counter(), defaultdict(Counter), set(), Counter()
                with open(params["candidates"]) as src:
                    for line in src:
                        row = json.loads(line)
                        key = row["id"]
                        if key in seen or key not in refs or key not in qunits:
                            raise ValueError("Candidate id/coverage mismatch")
                        seen.add(key)
                        d = diagnose_pool(refs[key], row["candidates"], qunits[key], setting)
                        if (
                            d["selected"]["counts"]["total"] != sunits[key]["counts"]["total"]
                            or d["selected"]["normalized_hypothesis"] != sunits[key]["hypothesis"]
                        ):
                            raise ValueError("Cached selection does not reproduce the frozen scored output")
                        counters["units"] += 1
                        counters[d["gap"]] += 1
                        counters["unique_candidates"] += d["unique_candidates"]
                        for flag in (
                            "normalized_qwen_in_pool",
                            "compound_equivalent_qwen_in_pool",
                            "pool_has_correct",
                            "ranking_missed_correct",
                            "first_pass_overturns_better_speech_choice",
                            "word_reward_overturns_better_speech_choice",
                        ):
                            counters[flag] += int(d[flag])
                        if d["qwen_correct"] and d["first_wrong"]:
                            counters["qwen_correct_first_wrong"] += 1
                            counters["qwen_correct_first_wrong_" + d["gap"]] += 1
                        for role in ("first", "selected", "speech_only", "auxiliary_choice", "oracle"):
                            totals[role].update(
                                {k: d[role]["counts"][k] for k in ("ins", "del", "sub", "total", "ref_len")}
                            )
                        totals["qwen"].update(
                            {k: qunits[key]["counts"][k] for k in ("ins", "del", "sub", "total", "ref_len")}
                        )
                        counters["augmented_oracle_errors"] += d["augmented_oracle_errors"]
                        dst.write(json.dumps(dict(family=choice["family"], id=key, **d)) + "\n")
                        case = d["gap"]
                        if d["first_pass_overturns_better_speech_choice"]:
                            case += "/first-pass-overturn"
                        if example_counts[case] < self.examples_per_case:
                            examples.write(
                                json.dumps(
                                    dict(
                                        family=choice["family"],
                                        id=key,
                                        case=case,
                                        reference=refs[key][:1200],
                                        qwen=qunits[key]["hypothesis"][:1200],
                                        selected=row["candidates"][d["selected"]["index"]]["text"][:1200],
                                        pool_oracle=row["candidates"][d["oracle"]["index"]]["text"][:1200],
                                        score_components={
                                            r: {k: v for k, v in d[r].items() if k != "normalized_hypothesis"}
                                            for r in ("first", "selected", "speech_only", "auxiliary_choice", "oracle")
                                        },
                                    ),
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            example_counts[case] += 1
                if seen != set(qunits) or seen != set(refs):
                    raise ValueError("Incomplete candidate/standalone coverage")
                results.append(
                    dict(
                        family=choice["family"],
                        setting=setting,
                        counters=counters,
                        scores={k: dict(v, wer=100 * v["total"] / v["ref_len"]) for k, v in totals.items()},
                        pool_path=params["candidates"],
                        pool_sha256=cc.sha256(params["candidates"]),
                        qwen_source=q["score"],
                        selection_source=selected["score"],
                    )
                )
        output = dict(
            dataset=self.dataset,
            normalizer=cc.NORMALIZER,
            protocol_sha256=cc.sha256(self.protocol.get_path()),
            note="Cached whole-sentence scores; not an attribution of within-word pruning. First-pass LM/fallback costs retained. "
            "Augmented oracle adds Qwen's output without a fabricated AM score; no deployable union ranking is claimed.",
            results=results,
        )
        with open(self.out_summary.get_path(), "w") as dst:
            json.dump(output, dst, indent=2)
        with open(self.out_report.get_path(), "w") as dst:
            dst.write("# Candidate coverage and ranking: " + self.dataset + "\n\n" + output["note"] + "\n\n")
            dst.write(
                "| Pair | First-pass WER | Rescored WER | Qwen WER | Pool oracle WER | Coverage gaps vs Qwen | Ranking gaps vs Qwen | First-pass overturns better auxiliary choice |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
            )
            for r in results:
                s, c = r["scores"], r["counters"]
                dst.write(
                    f"| {r['family'].replace(' | ',' / ')} | {s['first']['wer']:.3f} | {s['selected']['wer']:.3f} | {s['qwen']['wer']:.3f} | {s['oracle']['wer']:.3f} | {c['coverage_gap_vs_qwen']} | {c['ranking_gap_vs_qwen']} | {c['first_pass_overturns_better_speech_choice']} |\n"
                )


class TrainingWordFrequenciesJob(Job):
    def __init__(self, bliss_corpus, *, code_version):
        self.bliss_corpus, self.code_version = bliss_corpus, code_version
        self.out_frequencies = self.output_path("frequencies.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=8, time=12))

    def run(self):
        from .hf_normalizer.normalizer import EnglishTextNormalizer

        normalizer = EnglishTextNormalizer()
        counts = Counter()
        segments = 0
        with uopen(self.bliss_corpus.get_path(), "rb") as src:
            iterator = ET.iterparse(src, events=("start", "end"))
            event, root = next(iterator)
            if root.tag != "corpus" or root.attrib.get("name") != "train-other-960":
                raise ValueError("Rarity frequencies must come from LibriSpeech train-other-960, not evaluation text")
            for event, elem in iterator:
                if event == "end" and elem.tag == "orth":
                    counts.update(normalizer(" ".join(elem.itertext())).split())
                    segments += 1
                    elem.clear()
                elif event == "end" and elem.tag == "segment":
                    elem.clear()
        if not segments or not counts:
            raise ValueError("Empty training frequency source")
        with open(self.out_frequencies.get_path(), "w") as dst:
            json.dump(
                dict(
                    source="LibriSpeech train-other-960 transcripts",
                    source_sha256=cc.sha256(self.bliss_corpus.get_path()),
                    normalizer=cc.NORMALIZER,
                    segments=segments,
                    tokens=sum(counts.values()),
                    counts=dict(sorted(counts.items())),
                ),
                dst,
            )


def frequency_bin(count):
    return (
        "unseen"
        if count == 0
        else "1-5" if count <= 5 else "6-20" if count <= 20 else "21-100" if count <= 100 else ">100"
    )


def duration_bin(seconds):
    for upper in (5, 10, 20, 30, 60, 300):
        if seconds < upper:
            return "<" + str(upper) + "s"
    return ">=300s"


def source_bin(dataset, seq_id):
    if not dataset.startswith("loquacious-"):
        return dataset
    from ...data.loquacious_benchmark import FILTERS

    matches = [name for name, pattern in FILTERS.items() if re.search(pattern, seq_id)]
    return matches[0] if len(matches) == 1 else "unknown/ambiguous"


class ScoringUnitDurationsJob(Job):
    """Use recorded input audio length; never decode audio or infer from text."""

    def __init__(self, hypotheses, references, *, code_version):
        self.hypotheses, self.references, self.code_version = hypotheses, references, code_version
        self.out_durations = self.output_path("durations.json")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=4, time=2))

    def run(self):
        refs = {}
        with open(self.references.get_path()) as src:
            for line in src:
                r = json.loads(line)
                if r["id"] in refs:
                    raise ValueError("Duplicate reference id")
                refs[r["id"]] = r
        groups = defaultdict(lambda: dict(seconds=0.0, chunks=0, max_chunk_seconds=0.0))
        seen = set()
        with open(self.hypotheses.get_path()) as src:
            for line in src:
                r = json.loads(line)
                key = r["id"]
                if key not in refs or key in seen:
                    raise ValueError("Duration/reference coverage mismatch")
                seconds = r.get("audio_seconds")
                if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError("Missing valid recorded audio duration; do not estimate it from runtime/text")
                seen.add(key)
                g = groups[refs[key]["group_id"]]
                g["seconds"] += seconds
                g["chunks"] += 1
                g["max_chunk_seconds"] = max(g["max_chunk_seconds"], seconds)
        if seen != set(refs):
            raise ValueError("Missing audio duration ids")
        with open(self.out_durations.get_path(), "w") as dst:
            json.dump(
                dict(
                    reference_sha256=cc.sha256(self.references.get_path()),
                    units=groups,
                    semantics="Summed input-chunk duration per official scoring unit; not necessarily unsegmented model input",
                ),
                dst,
            )


class StratifiedErrorsJob(Job):
    def __init__(self, dataset, snapshots, protocol, durations, frequencies, *, code_version):
        self.dataset, self.snapshots, self.protocol = dataset, snapshots, protocol
        self.durations, self.frequencies, self.code_version = durations, frequencies, code_version
        self.out_summary = self.output_path("stratified.json")
        self.out_report = self.output_path("stratified.md")

    def tasks(self):
        yield Task("run", rqmt=dict(cpu=1, mem=12, time=8))

    def run(self):
        protocol = cc.read_json(self.protocol.get_path())
        durations = cc.read_json(self.durations.get_path())
        frequencies = cc.read_json(self.frequencies.get_path())
        if frequencies["normalizer"] != cc.NORMALIZER:
            raise ValueError("Frequency normalization mismatch")
        records = []
        for path in self.snapshots:
            records += [r for r in cc.read_json(path.get_path())["records"] if r["dataset"] == self.dataset]
        records = list({(r["group"], r["condition"]): r for r in records}.values())
        panels = {}
        for panel in ("native", "beam-controls", "dev-selected"):
            controls, fusions, missing = ec.panel_records(records, protocol, panel)
            active = list(controls.values()) + fusions
            if not active:
                panels[panel] = dict(systems=[], missing=missing)
                continue
            anchor = active[0]
            units = [ec.load_units(r) for r in active]
            for r, u in zip(active, units):
                ec.check_pair(anchor, r, units[0], u)
                if r["reference_sha256"] != durations["reference_sha256"]:
                    raise ValueError("Duration/reference provenance differs")
            ids = set(units[0])
            if ids != set(durations["units"]):
                raise ValueError("Duration/scoring-unit coverage mismatch")
            slices = [defaultdict(Counter) for _ in active]
            words = [defaultdict(Counter) for _ in active]
            changes = [defaultdict(lambda: defaultdict(Counter)) for _ in active]
            control_indices = {r["model"]: i for i, r in enumerate(active[: len(controls)])}
            excluded = 0
            for key in sorted(ids):
                duration = durations["units"][key]
                src = source_bin(self.dataset, key)
                labels = [
                    "duration/" + duration_bin(duration["seconds"]),
                    "source/" + src,
                    "input/" + ("multi-chunk" if duration["chunks"] > 1 else "single-chunk"),
                ]
                for j, u in enumerate(units):
                    for label in labels:
                        slices[j][label].update(
                            {k: u[key]["counts"][k] for k in ("ins", "del", "sub", "total", "ref_len")}
                        )
                        slices[j][label]["units"] += 1
                refs = [u[key]["reference"] for u in units]
                if len(set(refs)) != 1:
                    excluded += 1
                    continue
                tokens = refs[0].split()
                freq = [frequencies["counts"].get(t, 0) for t in tokens]
                neighbors = {
                    j
                    for i, c in enumerate(freq)
                    if c <= 5
                    for j in (i - 1, i + 1)
                    if 0 <= j < len(tokens) and freq[j] > 5
                }
                aligned = [ec.aligned_errors(u[key])[0] for u in units]
                for j, u in enumerate(units):
                    outcomes = aligned[j]
                    for i, (kind, _) in enumerate(outcomes):
                        bins = [
                            "frequency/" + frequency_bin(freq[i]),
                            "category/" + ("contains-digit" if re.search(r"\d", tokens[i]) else "no-digit"),
                        ]
                        if i in neighbors:
                            bins.append("context/common-neighbor-of-rare")
                        for label in bins:
                            words[j][label]["reference_words"] += 1
                            words[j][label][kind] += 1
                            if j >= len(controls):
                                for baseline in {active[j]["model"], active[j]["lm"]} & set(control_indices):
                                    base_kind = aligned[control_indices[baseline]][i][0]
                                    delta = changes[j][baseline][label]
                                    delta["reference_words"] += 1
                                    delta["corrected"] += int(base_kind != "correct" and kind == "correct")
                                    delta["introduced"] += int(base_kind == "correct" and kind != "correct")
            systems = []
            for j, (r, s, w) in enumerate(zip(active, slices, words)):
                systems.append(
                    dict(
                        model=r["model"],
                        auxiliary=r["lm"],
                        method=r["decoding"],
                        condition=r["condition"],
                        source=r["score"],
                        slices={
                            k: dict(v, wer=100 * v["total"] / v["ref_len"] if v["ref_len"] else None)
                            for k, v in sorted(s.items())
                        },
                        words={
                            k: dict(v, accuracy=100 * v["correct"] / v["reference_words"]) for k, v in sorted(w.items())
                        },
                        paired_word_changes={
                            b: {label: dict(counts) for label, counts in sorted(bins.items())}
                            for b, bins in sorted(changes[j].items())
                        },
                    )
                )
            panels[panel] = dict(
                systems=systems,
                missing=missing,
                word_units_included=len(ids) - excluded,
                word_units_excluded_reference_mismatch=excluded,
            )
        output = dict(
            dataset=self.dataset,
            panels=panels,
            frequency_source=frequencies["source"],
            frequency_source_sha256=frequencies["source_sha256"],
            normalizer=cc.NORMALIZER,
            protocol_sha256=cc.sha256(self.protocol.get_path()),
            frequencies_sha256=cc.sha256(self.frequencies.get_path()),
            durations_sha256=cc.sha256(self.durations.get_path()),
            note="Exploratory fixed-bin analysis; no pooling or tuning. Frequency means LibriSpeech-training frequency, not generic rarity or pretrained-model exposure. "
            "Word statistics use a common reference grid within each panel, excluding compound-normalization mismatches for every system. "
            "Insertions belong to unit WER, not reference-word accuracy. Neighbor means a >5-frequency word adjacent to a <=5-frequency word. "
            "Duration is summed chunk audio per scoring unit; grouped long recordings are not evidence of unsegmented decoding. "
            "No proper-name claim is made without independent annotation.",
        )
        with open(self.out_summary.get_path(), "w") as dst:
            json.dump(output, dst, indent=2)
        with open(self.out_report.get_path(), "w") as dst:
            dst.write("# Stratified errors: " + self.dataset + "\n\n" + output["note"] + "\n\n")
            for panel, p in panels.items():
                dst.write("## " + panel + "\n\n")
                if p["missing"]:
                    dst.write(f"Unavailable controls/conditions: {len(p['missing'])}; details in summary.json.\n\n")
                dst.write(
                    f"Common-grid word units: {p.get('word_units_included',0)}; excluded: {p.get('word_units_excluded_reference_mismatch',0)}.\n\n"
                )
                dst.write(
                    "| Model / auxiliary | Slice | Units | Reference words | WER / word accuracy % |\n|---|---|---:|---:|---:|\n"
                )
                for s in p["systems"]:
                    for k, v in s["slices"].items():
                        val = f"{v['wer']:.3f}" if v["wer"] is not None else "—"
                        dst.write(
                            f"| {s['model']} / {s['auxiliary']} ({s['condition']}) | {k} WER | {v['units']} | {v['ref_len']} | {val} |\n"
                        )
                    for k, v in s["words"].items():
                        dst.write(
                            f"| {s['model']} / {s['auxiliary']} ({s['condition']}) | {k} accuracy | — | {v['reference_words']} | {v['accuracy']:.3f} |\n"
                        )
                dst.write("\nPaired reference-word changes (common normalized reference grid):\n\n")
                dst.write(
                    "| Fusion | Standalone control | Slice | Reference words | Corrected | Introduced |\n|---|---|---|---:|---:|---:|\n"
                )
                for s in p["systems"]:
                    for baseline, bins in s["paired_word_changes"].items():
                        for label, v in bins.items():
                            dst.write(
                                f"| {s['model']} / {s['auxiliary']} ({s['condition']}) | {baseline} | {label} | {v['reference_words']} | {v['corrected']} | {v['introduced']} |\n"
                            )
