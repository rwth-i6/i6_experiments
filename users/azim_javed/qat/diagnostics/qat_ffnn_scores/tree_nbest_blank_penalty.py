"""Optimize corpus word errors by reranking completed tree-search N-best output.

Reads the standard nbest.json written by common.recog. No model, alignment,
decoder, or Sisyphus manager is run. Costs are the existing exported traceback
totals, not independently verified complete lattice costs.
"""

import configparser
import gzip
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Optional

from sisyphus import Job, Task, tk


def number(value):
    """Exact rational representation of a saved decimal, avoiding epsilon ties."""
    return Fraction(str(value))


def iter_nbest(stream, chunk_size=1024 * 1024):
    """Stream a standard JSON object without retaining all raw tracebacks.

    Each top-level value must be one complete utterance record. The file must
    have its closing brace: partial output from a running job is not accepted.
    """
    def invalid_constant(value):
        raise ValueError(f"Non-standard JSON number: {value}")

    decoder = json.JSONDecoder(parse_constant=invalid_constant)
    buffer, position, eof = "", 0, False

    def refill():
        nonlocal buffer, position, eof
        text = stream.read(chunk_size)
        buffer = buffer[position:] + text
        position = 0
        eof = not text

    def peek():
        nonlocal position
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                return buffer[position]
            if eof:
                return None
            refill()

    def expect(character):
        nonlocal position
        if peek() != character:
            raise ValueError(f"Invalid/incomplete nbest.json: expected {character!r}")
        position += 1

    def value():
        nonlocal position
        peek()
        while True:
            try:
                result, end = decoder.raw_decode(buffer, position)
                position = end
                return result
            except json.JSONDecodeError as error:
                if eof:
                    raise ValueError("Invalid/incomplete nbest.json; use completed recognition output") from error
                refill()

    expect("{")
    first = True
    seen = set()
    while peek() != "}":
        if not first:
            expect(",")
        tag = value()
        if tag in seen:
            raise ValueError(f"Duplicate utterance ID: {tag!r}")
        seen.add(tag)
        expect(":")
        yield tag, value()
        first = False
    expect("}")
    if peek() is not None:
        raise ValueError("Unexpected data after the nbest.json object")


def lexicon_counts(path):
    """Recover pronunciation lengths, not token IDs or frame alignments."""
    counts, sentence_ends = {}, set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as stream:
        for _, lemma in ET.iterparse(stream, events=("end",)):
            if lemma.tag != "lemma":
                continue
            lengths = {
                sum(token not in ("<blank>", "[BLANK]") for token in (phon.text or "").split())
                for phon in lemma.findall("phon")
            }
            names = {(orth.text or "").strip() for orth in lemma.findall("orth")}
            if lemma.get("name"):
                names.add(lemma.get("name"))
            names.discard("")
            for name in names:
                counts.setdefault(name, set()).update(lengths)
            if lemma.get("special") in ("sentence-end", "sentence-boundary"):
                sentence_ends.update(names)
            lemma.clear()
    sentence_ends.update(name for name in ("[SENTENCE-END]", "</s>") if name in counts)
    if not sentence_ends or any(counts[name] != {0} for name in sentence_ends):
        raise ValueError("Requires an empty-pronunciation sentence-end lemma")
    return counts, sentence_ends


def read_context(path, lexicons):
    config = configparser.ConfigParser(interpolation=None, strict=False)
    config.optionxform = str
    with path.open() as stream:
        config.read_file(stream)
    search = "lib-rasr.search-algorithm"
    scorer = "lib-rasr.label-scorer"
    recognition = scorer + ".recognition"
    required = (
        (search, "type", "tree-timesync-beam-search"),
        (search, "tree-builder-type", "rna"),
        ("lib-rasr.acoustic-model", "state-tying.type", "monophone"),
        ("lib-rasr.acoustic-model.hmm", "states-per-phone", "1"),
        ("lib-rasr.acoustic-model.hmm", "state-repetitions", "1"),
    )
    for section, key, expected in required:
        if config.get(section, key, fallback="") != expected:
            raise ValueError(f"Requires {section}.{key} = {expected}: {path}")
    for section, key in (
        (search, "collapse-repeated-labels"),
        (scorer, "presoftmax-blank-penalty"),
        (scorer, "vertical-label-transition"),
    ):
        if config.getboolean(section, key, fallback=False):
            raise ValueError(f"Unsupported enabled setting: {section}.{key}")
    if config.has_option("lib-rasr", "num-label-scorers"):
        raise ValueError("V1 supports a single label scorer")
    if config.get(scorer, "type", fallback="") not in ("fixed-context-py", "fixed-context-py-cache"):
        raise ValueError("V1 requires the fixed-context Python scorer")
    for section in (scorer, recognition):
        if number(config.get(section, "scale", fallback="1")) != 1:
            raise ValueError("V1 requires unit scorer scale")
    lm_scale = number(config.get("lib-rasr.lm", "scale", fallback=config.get("lib-rasr", "lm.scale", fallback="1")))
    lexicon = Path(config.get("lib-rasr", "lexicon.file"))
    if not lexicon.is_absolute():
        raise ValueError(f"Expected an absolute saved lexicon path: {lexicon}")
    if lexicon not in lexicons:
        lexicons[lexicon] = lexicon_counts(lexicon)
    anchor = number(config.get(recognition, "blank-penalty", fallback="0"))
    metadata = {
        "rasr_config": str(path), "lexicon": str(lexicon),
        "collection_penalty": float(anchor),
        "lm_scale": float(lm_scale),
        "ilm_scale": float(number(config.get(recognition, "ilm-scale", fallback="0"))),
        "sentence_end_fallback_enabled": config.getboolean(search, "sentence-end-fall-back", fallback=True),
        "joint_l1_hardware": config.get(recognition + ".memristor", "joint-l1-converter-hardware-settings", fallback=None),
        "joint_l2_hardware": config.get(recognition + ".memristor", "joint-l2-converter-hardware-settings", fallback=None),
    }
    return config, anchor, lexicons[lexicon], metadata


def _open_rasr_log(path, mode="rt"):
    with open(path, "rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    return gzip.open(path, mode) if compressed else open(path, mode)


def check_fallback(contexts, log_file):
    """Allow enabled fallback only when the saved log establishes it was unused."""
    enabled = [context for context in contexts.values() if context[3]["sentence_end_fallback_enabled"]]
    if not enabled:
        return {"required": False, "events": 0}
    for config, _, _, _ in enabled:
        for channel in ("warning.channel", "log.channel"):
            if config.get("*", channel, fallback="") != log_file.name:
                raise ValueError(f"Cannot audit fallback: {channel} does not point to {log_file.name}")
    events = 0
    with _open_rasr_log(log_file) as stream:
        for line in stream:
            if "Use sentence-end fallback" in line or "No active word-end hypothesis at segment end." in line:
                events += 1
    if events:
        raise ValueError(
            f"Sentence-end fallback was used ({events} diagnostic messages in {log_file}). "
            "Unfinished-word tokens are hidden; blank counts cannot be inferred safely. "
            "Collect complete paths with sentence-end-fall-back=no."
        )
    return {"required": True, "log_file": str(log_file), "events": 0}


def edit_counts(reference, hypothesis):
    """Whitespace word edits (S,D,I); ties prefer substitution, then deletion."""
    ref, hyp = reference.split(), hypothesis.split()
    previous = [(0, 0, j) for j in range(len(hyp) + 1)]
    for i, word in enumerate(ref, 1):
        current = [(0, i, 0)]
        for j, predicted in enumerate(hyp, 1):
            s, d, ins = previous[j - 1]
            diagonal = (s + (word != predicted), d, ins)
            s, d, ins = previous[j]
            deletion = (s, d + 1, ins)
            s, d, ins = current[-1]
            insertion = (s, d, ins + 1)
            current.append(min((diagonal, deletion, insertion), key=sum))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class Candidate:
    rank: int
    transcript: str
    intercept: Fraction
    blanks: int
    edits: tuple[int, int, int]
    source_index: int = 0
    source_rank: int = 0

    @property
    def errors(self):
        return sum(self.edits)

    def cost(self, penalty):
        return self.intercept + penalty * self.blanks

    def record(self):
        return {
            "rank": self.rank, "transcription": self.transcript,
            "source_index": self.source_index,
            "source_rank": self.source_rank,
            "zero_penalty_cost": float(self.intercept),
            "zero_penalty_cost_exact": str(self.intercept), "blank_count": self.blanks,
            "word_errors": self.errors,
            "substitutions": self.edits[0], "deletions": self.edits[1], "insertions": self.edits[2],
        }


def prepare_candidates(record, anchor, counts, sentence_ends, lm_scale=0):
    """Trust the recognition JSON schema; reject unusable path observations."""
    frames = record["encoder_frames"]
    reference = record["reference"]
    candidates, rejected = [], Counter()
    for rank, candidate in enumerate(record["candidates"], 1):
        try:
            trace = candidate["traceback"]
            if not trace or trace[-1]["lemma"] not in sentence_ends:
                raise ValueError("missing sentence-end item")
            # N-best LM components are unscaled; the ILM term is already in AM.
            if lm_scale:
                cost = number(trace[-1]["am_score"]) + lm_scale * number(trace[-1]["lm_score"])
            else:
                cost = number(candidate["traceback_terminal_cost"])
            tokens = 0
            for item in trace:
                lengths = counts.get(item["lemma"], set())
                if len(lengths) != 1:
                    raise ValueError("unknown/ambiguous lemma token count")
                tokens += next(iter(lengths))
            transcript = candidate["transcription"]
            blanks = frames - tokens
            if blanks < 0:
                raise ValueError("negative blank count: more model tokens than encoder frames")
            candidates.append(Candidate(
                rank, transcript, cost - anchor * blanks, blanks,
                edit_counts(reference, transcript),
            ))
        except ValueError as error:
            rejected[str(error)] += 1
    if not candidates:
        raise ValueError(f"No usable complete candidates: {dict(rejected)}")
    return candidates, rejected


def lower_envelope(candidates):
    """Return (inclusive start, candidate) pairs; first start is minus infinity.

    At cost ties prefer fewer blanks, then earlier exported candidate order.
    This makes each switch right-continuous, including multi-line crossings.
    """
    by_slope = {}
    for line in candidates:
        old = by_slope.get(line.blanks)
        if old is None or (line.intercept, line.rank) < (old.intercept, old.rank):
            by_slope[line.blanks] = line
    hull = []
    for line in sorted(by_slope.values(), key=lambda item: -item.blanks):
        start = None
        while hull:
            previous_start, previous = hull[-1]
            start = (line.intercept - previous.intercept) / (previous.blanks - line.blanks)
            if previous_start is None or start > previous_start:
                break
            hull.pop()
        hull.append((start if hull else None, line))
    return hull


def winner(candidates, penalty):
    return min(candidates, key=lambda line: (line.cost(penalty), line.blanks, line.rank))


def error_record(edits, words):
    return {"word_errors": sum(edits), "substitutions": edits[0],
            "deletions": edits[1], "insertions": edits[2],
            "wer_percent": 100.0 * sum(edits) / words}


def representative(lower, upper, upper_inclusive, preferred):
    if lower == upper:
        ideal = lower
    elif preferred > lower and (upper is None or preferred < upper):
        ideal = preferred
    else:
        ideal = lower + 1 if upper is None else (lower + upper) / 2
    value = float(ideal)
    if not math.isfinite(value):
        return None
    if number(value) < lower:
        value = math.nextafter(value, math.inf)
    if upper is not None and (number(value) > upper or (number(value) == upper and not upper_inclusive)):
        value = math.nextafter(value, -math.inf)
    if number(value) < lower or (upper is not None and (
        number(value) > upper or (number(value) == upper and not upper_inclusive)
    )):
        return None
    return value


def optimize(utterances, *, min_penalty=0.0, max_penalty=None,
             preferred_penalty=1.44, penalties=(0.0, 1.44, 2.0, 4.0)):
    lower = number(min_penalty)
    upper = None if max_penalty is None else number(max_penalty)
    preferred = number(preferred_penalty)
    if upper is not None and upper < lower:
        raise ValueError("max_penalty must be >= min_penalty")
    words = sum(len(record["reference"].split()) for record in utterances.values())
    if not utterances or words == 0:
        raise ValueError("Need a nonempty corpus with reference words")
    active, events, edits = {}, defaultdict(list), [0, 0, 0]
    for tag, record in utterances.items():
        hull = lower_envelope(record["candidates"])
        line = hull[0][1]
        for start, candidate in hull[1:]:
            if start <= lower:
                line = candidate
            elif upper is None or start <= upper:
                events[start].append((tag, candidate))
        active[tag] = line
        edits = [a + b for a, b in zip(edits, line.edits)]

    curve = []

    def region(left, right, inclusive):
        curve.append({
            "lower": float(left), "upper": None if right is None else float(right),
            "lower_exact": str(left), "upper_exact": None if right is None else str(right),
            "lower_inclusive": True, "upper_inclusive": inclusive,
            "representative_penalty": representative(left, right, inclusive, preferred),
            **error_record(edits, words),
        })

    start = lower
    for threshold in sorted(events):
        region(start, threshold, False)
        for tag, new in events[threshold]:
            old = active[tag]
            edits = [total + new_value - old_value
                     for total, new_value, old_value in zip(edits, new.edits, old.edits)]
            active[tag] = new
        start = threshold
    region(start, upper, upper is not None)
    minimum = min(row["word_errors"] for row in curve)
    optimal = [row for row in curve if row["word_errors"] == minimum]
    usable = [row for row in optimal if row["representative_penalty"] is not None]
    if not usable:
        raise ValueError("No optimal interval has a representable floating-point penalty")
    selected = min(usable, key=lambda row: (
        abs(number(row["representative_penalty"]) - preferred), number(row["lower_exact"]),
    ))
    comparisons = []
    for penalty in sorted({number(value) for value in penalties}):
        totals = [0, 0, 0]
        for record in utterances.values():
            line = winner(record["candidates"], penalty)
            totals = [a + b for a, b in zip(totals, line.edits)]
        comparisons.append({"penalty": float(penalty), **error_record(totals, words)})
    return {
        "reference_words": words, "minimum_word_errors": minimum,
        "minimum_wer_percent": 100.0 * minimum / words,
        "selected_penalty": selected["representative_penalty"],
        "selected_interval": selected, "optimal_intervals": optimal,
        "unique_corpus_switches": len(events), "curve": curve, "comparisons": comparisons,
    }


def compatible_settings(config):
    """Compare scoring/topology settings while allowing penalty/beam changes.

    Keep checkpoint paths, lexicon, hardware, LM/ILM and scorer settings in the
    comparison. Logging destinations and search pruning settings are not scores.
    """
    beam_keys = {"max-beam-size", "max-word-end-beam-size", "score-threshold",
                 "word-end-score-threshold"}
    return {
        (section, key): value
        for section in config.sections() if section != "*"
        for key, value in config.items(section)
        if not (section == "lib-rasr.label-scorer.recognition" and key == "blank-penalty")
        and not (section == "lib-rasr.search-algorithm" and key in beam_keys)
        and not key.endswith(".channel")
    }


def read_source(filename, log_file, contexts, lexicons):
    """Stream one variant into compact candidate records and source statistics."""
    utterances, source_contexts, rejected = {}, {}, Counter()
    with filename.open() as stream:
        for tag, record in iter_nbest(stream):
            config_path = Path(record["rasr_config"])
            if config_path not in contexts:
                context = read_context(config_path, lexicons)
                if contexts:
                    expected = compatible_settings(next(iter(contexts.values()))[0])
                    actual = compatible_settings(context[0])
                    differences = [key for key in expected.keys() | actual.keys()
                                   if expected.get(key) != actual.get(key)]
                    if differences:
                        raise ValueError(f"{filename}: incompatible recognition settings: {sorted(differences)}")
                contexts[config_path] = context
            source_contexts[config_path] = contexts[config_path]
            _, anchor, (counts, sentence_ends), metadata = contexts[config_path]
            try:
                candidates, invalid = prepare_candidates(
                    record, anchor, counts, sentence_ends, lm_scale=number(metadata["lm_scale"]),
                )
            except ValueError as error:
                raise ValueError(f"{filename}, {tag}: {error}") from error
            utterances[tag] = {"reference": record["reference"], "encoder_frames": record["encoder_frames"],
                               "candidates": candidates}
            rejected.update(invalid)
    if not utterances:
        raise ValueError(f"No utterances in {filename}")
    records = list(utterances.values())
    return utterances, {
        "nbest_file": str(filename), "recognition_log": str(log_file),
        "collection_penalties": sorted({float(context[1]) for context in source_contexts.values()}),
        "utterances": len(records),
        "usable_candidate_observations": sum(len(record["candidates"]) for record in records),
        "rejected_candidates": sum(rejected.values()), "rejection_reasons": dict(rejected),
        "reference_present_utterances": sum(any(line.errors == 0 for line in record["candidates"]) for record in records),
        "candidate_oracle_word_errors": sum(min(line.errors for line in record["candidates"]) for record in records),
        "fallback_audit": check_fallback(source_contexts, log_file),
    }


def analyze_nbest(nbest_files, recognition_logs, *, min_penalty=0.0,
                  max_penalty=None, preferred_penalty=1.44,
                  penalties=(0.0, 1.44, 2.0, 4.0)):
    contexts, lexicons, utterances, sources = {}, {}, {}, []
    rejected = Counter()
    for source_index, (filename, log_file) in enumerate(zip(nbest_files, recognition_logs, strict=True)):
        source, stats = read_source(Path(filename), Path(log_file), contexts, lexicons)
        if utterances and source.keys() != utterances.keys():
            raise ValueError(f"{filename}: sources must contain identical utterances")
        for tag, record in source.items():
            pooled = utterances.setdefault(tag, {"reference": record["reference"],
                                                "encoder_frames": record["encoder_frames"], "candidates": []})
            if (pooled["reference"], pooled["encoder_frames"]) != (record["reference"], record["encoder_frames"]):
                raise ValueError(f"{filename}, {tag}: reference or encoder-frame count differs between sources")
            offset = len(pooled["candidates"])
            pooled["candidates"].extend(
                replace(line, rank=offset + index, source_index=source_index, source_rank=line.rank)
                for index, line in enumerate(record["candidates"], 1)
            )
        sources.append({"source_index": source_index, **stats})
        rejected.update(stats["rejection_reasons"])

    # Identical transcript/count pairs have parallel lines and equal edit errors.
    # Keep their lowest intercept, breaking exact ties by source/export order.
    candidate_count, oracle_errors, reference_present = 0, 0, 0
    for record in utterances.values():
        unique = {}
        for line in sorted(record["candidates"], key=lambda line: (line.intercept, line.rank)):
            unique.setdefault((line.transcript, line.blanks), line)
        record["candidates"] = sorted(unique.values(), key=lambda line: line.rank)
        candidate_count += len(unique)
        oracle_errors += min(line.errors for line in unique.values())
        reference_present += any(line.errors == 0 for line in unique.values())
    anchors = {context[1] for context in contexts.values()}
    result = optimize(
        utterances, min_penalty=min_penalty, max_penalty=max_penalty,
        preferred_penalty=preferred_penalty, penalties=tuple(penalties) + tuple(anchors),
    )
    observations = sum(source["usable_candidate_observations"] for source in sources)
    result.update({
        "objective": "Corpus word errors from fixed exported N-best candidate costs",
        "nbest_files": [source["nbest_file"] for source in sources], "sources": sources,
        "utterances": len(utterances),
        "usable_candidates": candidate_count, "rejected_candidates": sum(rejected.values()),
        "usable_candidate_observations": observations,
        "duplicate_candidate_observations": observations - candidate_count,
        "rejection_reasons": dict(rejected), "reference_present_utterances": reference_present,
        "reference_coverage_percent": 100.0 * reference_present / len(utterances),
        "candidate_oracle_word_errors": oracle_errors,
        "candidate_oracle_wer_percent": 100.0 * oracle_errors / result["reference_words"],
        "min_penalty": float(number(min_penalty)),
        "max_penalty": None if max_penalty is None else float(number(max_penalty)),
        "preferred_penalty": float(number(preferred_penalty)),
        "tie_rule": "Minimum cost; fewer blanks at crossings; earlier export order for identical lines",
        "word_scoring": "Whitespace word Levenshtein; S/D/I ties prefer substitution, deletion, insertion; not Sclite",
        "score_limitation": "Exported labeled-arc totals may omit epsilon/final costs; fresh decoding is required",
        "arithmetic": "Exact rational arithmetic on saved decimal costs, penalties and crossings",
        "pooling": "Union of transcripts/counts; minimum intercept per duplicate; source order breaks ties",
        "noise_limitation": "Across fresh noisy runs, minimum-intercept pooling favors lucky reads; it is not mean-score estimation",
        "contexts": [context[3] for context in contexts.values()],
    })
    for source in sources:
        source["reference_coverage_percent"] = 100.0 * source["reference_present_utterances"] / len(utterances)
        source["candidate_oracle_wer_percent"] = 100.0 * source["candidate_oracle_word_errors"] / result["reference_words"]
    return result, utterances


def format_results(result):
    lines = [
        result["objective"], "Inputs:",
        *[f"  {filename}" for filename in result["nbest_files"]],
        f"Utterances: {result['utterances']}; reference words: {result['reference_words']}",
        f"Candidates: {result['usable_candidates']} unique transcript/count pairs from "
        f"{result['usable_candidate_observations']} usable observations; {result['rejected_candidates']} rejected",
        f"Reference coverage: {result['reference_present_utterances']}/{result['utterances']} "
        f"({result['reference_coverage_percent']:.2f}%)",
        f"Candidate oracle WER: {result['candidate_oracle_wer_percent']:.4f}%",
        f"Selected penalty: {result['selected_penalty']:.12g}",
        f"Minimum candidate-set WER: {result['minimum_wer_percent']:.4f}% "
        f"({result['minimum_word_errors']} errors)",
        f"Tie rule: {result['tie_rule']}", "", "Per-source coverage:",
    ]
    for source in result["sources"]:
        lines.append(f"  source {source['source_index']}, anchors={source['collection_penalties']}: "
                     f"reference coverage={source['reference_coverage_percent']:.2f}%; "
                     f"oracle WER={source['candidate_oracle_wer_percent']:.4f}%")
    lines.extend(["", "Optimal intervals:"])
    for row in result["optimal_intervals"]:
        right = "infinity)" if row["upper"] is None else (
            f"{row['upper']:.12g}" + ("]" if row["upper_inclusive"] else ")"))
        lines.append(f"  [{row['lower']:.12g}, {right}; representative={row['representative_penalty']}")
    lines.extend(["", "Comparison penalties:"])
    for row in result["comparisons"]:
        lines.append(f"  b={row['penalty']:.12g}: {row['wer_percent']:.4f}% WER; "
                     f"S/D/I={row['substitutions']}/{row['deletions']}/{row['insertions']}")
    lines.extend(["", result["score_limitation"], result["word_scoring"], result["noise_limitation"],
                  "Readout scores are held fixed. No model rescoring or new decoding was performed."])
    return "\n".join(lines) + "\n"


def write_candidate_outputs(utterances, selected_penalty, candidates_file, hypotheses_file):
    """Write compact standard JSON, without copying the large raw tracebacks."""
    penalty = number(selected_penalty)
    candidates_file.write("{\n")
    hypotheses_file.write("{\n")
    for index, (tag, record) in enumerate(utterances.items()):
        if index:
            candidates_file.write(",\n")
            hypotheses_file.write(",\n")
        candidates_file.write(json.dumps(tag) + ": ")
        json.dump({"reference": record["reference"], "encoder_frames": record["encoder_frames"],
                   "candidates": [line.record() for line in record["candidates"]]},
                  candidates_file, allow_nan=False)
        hypotheses_file.write(json.dumps(tag) + ": ")
        line = winner(record["candidates"], penalty)
        json.dump({"reference": record["reference"], "selected_penalty": float(penalty), **line.record()},
                  hypotheses_file, allow_nan=False)
    candidates_file.write("\n}\n")
    hypotheses_file.write("\n}\n")


class TreeNBestBlankPenaltyJob(Job):
    """CPU-only Sisyphus optimizer consuming completed standard N-best JSON."""

    def __init__(self, nbest_files: tuple[tk.Path, ...], recognition_logs: tuple[tk.Path, ...],
                 min_penalty: float = 0.0, max_penalty: Optional[float] = None,
                 preferred_penalty: float = 1.44,
                 penalties: tuple[float, ...] = (0.0, 1.44, 2.0, 4.0)):
        self.nbest_files = nbest_files
        self.recognition_logs = recognition_logs
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.preferred_penalty = preferred_penalty
        self.penalties = penalties
        self.out_summary = self.output_path("optimal_tree_blank_penalty.json")
        self.out_report = self.output_path("optimal_tree_blank_penalty.txt")
        self.out_candidates = self.output_path("candidate_lines.json")
        self.out_hypotheses = self.output_path("best_hypotheses.json")
        self.out_blank_penalty = self.output_var("blank_penalty")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 8, "time": 2})

    def run(self):
        result, utterances = analyze_nbest(
            [Path(path.get_path()) for path in self.nbest_files],
            [Path(path.get_path()) for path in self.recognition_logs],
            min_penalty=self.min_penalty, max_penalty=self.max_penalty,
            preferred_penalty=self.preferred_penalty, penalties=self.penalties,
        )
        with open(self.out_summary.get_path(), "w") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        with open(self.out_report.get_path(), "w") as stream:
            stream.write(format_results(result))
        with open(self.out_candidates.get_path(), "w") as candidates_file, \
                open(self.out_hypotheses.get_path(), "w") as hypotheses_file:
            write_candidate_outputs(utterances, result["selected_penalty"], candidates_file, hypotheses_file)
        self.out_blank_penalty.set(result["selected_penalty"])


def run_tree_nbest_blank_penalty(recognition_jobs, *,
                                 output_prefix="diagnostics/tree_nbest_blank_penalty",
                                 min_penalty=0.0, max_penalty=None,
                                 preferred_penalty=1.44, penalties=(0.0, 1.44, 2.0, 4.0)):
    """Pool a list of recognition jobs; source order breaks exact score ties."""
    job = TreeNBestBlankPenaltyJob(
        nbest_files=tuple(source.out_files["nbest.json"] for source in recognition_jobs),
        recognition_logs=tuple(source.out_files["rasr.recog.log"] for source in recognition_jobs),
        min_penalty=min_penalty, max_penalty=max_penalty,
        preferred_penalty=preferred_penalty, penalties=penalties,
    )
    job.add_alias(f"{output_prefix}/optimize")
    for filename, output in (
        ("optimal_tree_blank_penalty.json", job.out_summary),
        ("optimal_tree_blank_penalty.txt", job.out_report),
        ("candidate_lines.json", job.out_candidates),
        ("best_hypotheses.json", job.out_hypotheses),
        ("blank_penalty", job.out_blank_penalty),
    ):
        tk.register_output(f"{output_prefix}/{filename}", output)
    return job
