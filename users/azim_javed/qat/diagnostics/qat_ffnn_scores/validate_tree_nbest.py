"""Statistically validate blank costs using repeated, naturally noisy reads.

Consumes probe_tree_nbest.py outputs, including encoder_NNN.npy. Reuses the
actual Python scorer initialization at two interleaved blank penalties, with
fresh caches and unmodified readout noise. No seeds are reset or noise draws
replayed. Reports cost distributions and Welch confidence intervals for mean
shifts. Monotonic Viterbi conditions on each candidate transcript; no lexical
beam search or digital model is used.

This first validator requires a single fixed-context scorer, no LM/ILM, and
the complete RNA, one-state monophone topology used by the default probe.
"""

import argparse
import gzip
import io
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np

if __package__:
    from .probe_tree_nbest import (
        DEFAULT_SISYPHUS, PROBE_ROOT, SETUP, config_without_decoder, read_lexicon,
        read_rasr_config, require_joint_l2_analysis_settings,
    )
else:
    from probe_tree_nbest import (
        DEFAULT_SISYPHUS, PROBE_ROOT, SETUP, config_without_decoder, read_lexicon,
        read_rasr_config, require_joint_l2_analysis_settings,
    )


def validate_supported_config(config):
    """Reject settings that would invalidate the independent cost calculation."""
    checks = (
        ("lib-rasr.search-algorithm", "tree-builder-type", "rna"),
        ("lib-rasr.acoustic-model", "state-tying.type", "monophone"),
        ("lib-rasr.acoustic-model.hmm", "states-per-phone", "1"),
        ("lib-rasr.acoustic-model.hmm", "state-repetitions", "1"),
    )
    for section, key, expected in checks:
        if config.get(section, key, fallback="") != expected:
            raise ValueError(f"Requires {section}.{key} = {expected}")
    for section, key in (
        ("lib-rasr.search-algorithm", "collapse-repeated-labels"),
        ("lib-rasr.search-algorithm", "sentence-end-fall-back"),
        ("lib-rasr.search-algorithm", "force-blank-between-repeated-labels"),
        ("lib-rasr.label-scorer", "presoftmax-blank-penalty"),
        ("lib-rasr.label-scorer", "blank-updates-history"),
        ("lib-rasr.label-scorer", "silence-updates-history"),
        ("lib-rasr.label-scorer", "loop-updates-history"),
        ("lib-rasr.label-scorer", "vertical-label-transition"),
    ):
        if config.getboolean(section, key, fallback=False):
            raise ValueError(f"Unsupported enabled setting: {section}.{key}")
    if config.has_option("lib-rasr", "num-label-scorers"):
        raise ValueError("First validator supports exactly one label scorer")
    if config.get("lib-rasr.label-scorer", "type") not in ("fixed-context-py", "fixed-context-py-cache"):
        raise ValueError("Requires the existing fixed-context Python scorer")
    if float(config.get("lib-rasr", "lm.scale", fallback="0")) != 0:
        raise ValueError("First validator requires word LM scale zero")
    recognition = "lib-rasr.label-scorer.recognition"
    if float(config.get(recognition, "ilm-scale", fallback="0")) != 0:
        raise ValueError("First validator requires ILM scale zero")
    for section in ("lib-rasr.label-scorer", recognition):
        if float(config.get(section, "scale", fallback="1")) != 1:
            raise ValueError("First validator requires unit scorer scale")
    for kind in ("*", "silence"):
        section = f"lib-rasr.acoustic-model.tdp.{kind}"
        for key in ("loop", "forward", "exit"):
            if float(config.get(section, key, fallback="0")) != 0:
                raise ValueError("Nonzero acoustic transition costs are not supported")
    for section in ("lib-rasr.acoustic-model.phonology",):
        for key in ("history-length", "future-length"):
            if int(config.get(section, key, fallback="0")) != 0:
                raise ValueError("Requires context-independent monophone mapping")


def read_token_mapping(lexicon_path):
    """RASR monophone, one-state emission index = phoneme ID - 1.

    RASR assigns phoneme IDs starting at 1 as inventory entries are parsed.
    Thus XML inventory order determines dense score indices under the guarded
    topology, not sorting tokens or iterating word pronunciations.
    """
    symbols = []
    opener = gzip.open if lexicon_path.suffix == ".gz" else open
    with opener(lexicon_path, "rb") as stream:
        for _, element in ET.iterparse(stream, events=("end",)):
            if element.tag == "phoneme":
                symbol = element.findtext("symbol")
                if not symbol:
                    raise ValueError("Phoneme inventory contains an empty symbol")
                symbols.append(symbol)
                element.clear()
            elif element.tag == "phoneme-inventory":
                break
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError("Expected a nonempty phoneme inventory with unique symbols")
    mapping = {symbol: index for index, symbol in enumerate(symbols)}
    if mapping.get("<blank>") != len(symbols) - 1:
        raise ValueError("Requires blank at the final scorer output index")
    return mapping


def candidate_tokens(candidate, pronunciations, mapping):
    """Reconstruct tokens from original lemma items, omitting explicit blanks."""
    tokens, symbols = [], []
    if not candidate["traceback"] or candidate["traceback"][-1]["lemma"] != "[SENTENCE-END]":
        raise ValueError("Candidate lacks a complete sentence-end traceback item")
    for item in candidate["traceback"]:
        lemma = item["lemma"]
        variants = pronunciations.get(lemma, set())
        if len(variants) != 1:
            raise ValueError(f"Cannot recover unique tokenization for lemma {lemma!r}")
        pronunciation = next(iter(variants))
        if lemma == "[SENTENCE-END]" and pronunciation:
            raise ValueError("Explicit sentence-end model tokens are not supported")
        for symbol in pronunciation:
            if symbol == "<blank>":
                continue
            if symbol not in mapping:
                raise ValueError(f"Unknown pronunciation token: {symbol!r}")
            symbols.append(symbol)
            tokens.append(mapping[symbol])
    return tokens, symbols


def select_candidates(candidates, limit):
    """Choose distinct transcripts: top result, best-error result, then rank order."""
    valid = [c for c in candidates if c.get("traceback_terminal_cost") is not None]
    if not valid:
        return []
    ordered = sorted(valid, key=lambda c: c["rank"])
    best_error = min(valid, key=lambda c: (c["word_errors"], c["traceback_terminal_cost"]))
    selected, seen = [], set()
    for candidate in [ordered[0], best_error] + ordered:
        key = candidate["transcription"]
        if key not in seen:
            seen.add(key)
            selected.append(candidate)
            if len(selected) == limit:
                break
    return selected


def candidates_for_validation(candidates, limit, transcripts=None):
    if not transcripts:
        return select_candidates(candidates, limit)
    selected = {}
    for candidate in candidates:
        text, cost = candidate["transcription"], candidate.get("traceback_terminal_cost")
        if text in transcripts and cost is not None:
            if text not in selected or cost < selected[text]["traceback_terminal_cost"]:
                selected[text] = candidate
    return sorted(selected.values(), key=lambda candidate: candidate["rank"])[:limit]


def monotonic_viterbi(blank_costs, label_costs):
    """Minimum cost over complete RNA alignments, returning emission frames.

    blank_costs: [T, U+1]; label_costs: [T, U]. Both transitions consume a frame.
    The result includes all T frames, including trailing blanks at prefix U.
    """
    frames, positions = blank_costs.shape
    labels = positions - 1
    if frames < 1 or label_costs.shape != (frames, labels) or labels > frames:
        raise ValueError("Invalid monotonic alignment dimensions")
    if not np.all(np.isfinite(blank_costs)) or not np.all(np.isfinite(label_costs)):
        raise ValueError("Nonfinite model costs")
    previous = np.full(positions, np.inf)
    previous[0] = 0.0
    emitted = np.zeros((frames, positions), dtype=bool)
    for t in range(frames):
        current = previous + blank_costs[t]
        alternative = previous[:-1] + label_costs[t]
        choose_label = alternative < current[1:]
        current[1:] = np.minimum(current[1:], alternative)
        emitted[t, 1:] = choose_label
        previous = current
    if not np.isfinite(previous[-1]):
        raise ValueError("Candidate has no complete alignment")
    emission_frames = []
    u = labels
    for t in range(frames - 1, -1, -1):
        if emitted[t, u]:
            emission_frames.append(t)
            u -= 1
    if u != 0:
        raise ValueError("Viterbi backtrace did not reach the initial prefix")
    return float(previous[-1]), list(reversed(emission_frames))


def fixed_alignment_cost(blank_costs, label_costs, emission_frames):
    """Replay a fixed alignment against independently evaluated model costs."""
    u, total = 0, 0.0
    emissions = set(emission_frames)
    if len(emissions) != label_costs.shape[1]:
        raise ValueError("Alignment must emit each candidate token once")
    for t in range(len(blank_costs)):
        if t in emissions:
            total += label_costs[t, u]
            u += 1
        else:
            total += blank_costs[t, u]
    if u != label_costs.shape[1]:
        raise ValueError("Alignment contains invalid emission frames")
    return float(total)


def prefix_histories(tokens, initial_history):
    history = tuple(initial_history)
    histories = [history]
    for token in tokens:
        history = history[1:] + (token,)
        histories.append(history)
    return histories


def score_lattice(wrapper, features, tokens, blank_id, penalty):
    """Call the actual recognition scorer afresh, not an algebraic score shift."""
    wrapper._model.blank_penalty = float(penalty)
    wrapper.reset()
    # reset() clears dense scores, but the cached variant retains prediction
    # states. Clear those too so repeat tests really recompute the model.
    if hasattr(wrapper._model, "prediction_net_cache"):
        wrapper._model.prediction_net_cache.clear()
    wrapper.add_inputs(features)
    wrapper.signal_no_more_features()
    initial_step, initial_history = wrapper.get_initial_scoring_context()
    if initial_step != 0 or not initial_history:
        raise ValueError("Expected a nonempty fixed initial history at frame zero")
    histories = prefix_histories(tokens, initial_history)
    blank = np.empty((len(features), len(histories)), dtype=np.float64)
    label = np.empty((len(features), len(tokens)), dtype=np.float64)
    for t in range(len(features)):
        contexts = [(t, history) for history in histories]
        scores = wrapper.compute_scores_with_times(contexts)
        for u, result in enumerate(scores):
            if result is None or result[1] != t:
                raise ValueError(f"Missing or incorrectly timed score at {(t, u)}")
            dense = result[0]
            if len(dense) != wrapper._model.target_size:
                raise ValueError("Scorer output dimension mismatch")
            blank[t, u] = dense[blank_id]
            if u < len(tokens):
                label[t, u] = dense[tokens[u]]
    return blank, label


def sample_summary(values, confidence):
    """Read-level mean uncertainty, not pseudo-replicated lattice-cell errors."""
    from scipy.stats import t
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite read-level samples are required")
    mean = float(np.mean(values))
    stddev = float(np.std(values, ddof=1))
    standard_error = stddev / math.sqrt(len(values))
    half_width = float(t.ppf((1 + confidence) / 2, len(values) - 1)) * standard_error
    return {
        "n": len(values), "mean": mean, "stddev": stddev,
        "standard_error": standard_error, "mean_ci": [mean - half_width, mean + half_width],
        "min": float(np.min(values)), "max": float(np.max(values)),
    }


def mean_shift_check(anchor_values, shifted_values, expected, confidence, margin):
    """Unpaired Welch interval for shifted mean minus anchor mean.

    Includes separate compatibility and practical-equivalence assessments.
    A wide interval containing the target is not a validation pass.
    """
    from scipy.stats import t
    anchor = sample_summary(anchor_values, confidence)
    shifted = sample_summary(shifted_values, confidence)
    a = anchor["stddev"] ** 2 / anchor["n"]
    b = shifted["stddev"] ** 2 / shifted["n"]
    standard_error = math.sqrt(a + b)
    df = ((a + b) ** 2 / (a ** 2 / (anchor["n"] - 1) + b ** 2 / (shifted["n"] - 1))
          if standard_error else None)
    half_width = float(t.ppf((1 + confidence) / 2, df)) * standard_error if df is not None else 0.0
    difference = shifted["mean"] - anchor["mean"]
    interval = [difference - half_width, difference + half_width]
    residual = [value - expected for value in interval]
    if residual[0] >= -margin and residual[1] <= margin:
        assessment = "within_practical_margin"
    elif residual[0] > margin or residual[1] < -margin:
        assessment = "material_difference_detected"
    elif residual[0] <= 0 <= residual[1]:
        assessment = "consistent_but_uncertain"
    else:
        assessment = "difference_detected_margin_decision_uncertain"
    return {
        "anchor": anchor, "shifted": shifted, "expected_shift": float(expected),
        "observed_mean_shift": difference, "mean_shift_ci": interval,
        "mean_shift_standard_error": standard_error, "welch_degrees_of_freedom": df,
        "mean_shift_residual": difference - expected, "residual_ci": residual,
        "target_inside_ci": interval[0] <= expected <= interval[1],
        "practical_margin": margin, "assessment": assessment,
    }


def collect_noisy_samples(wrapper, features, tokens, blank_id, anchor, delta, repeats,
                          on_sample=None, save_costs=False):
    """Fresh stochastic evaluations; never reset or restore any RNG state.

    An independent pilot chooses the replay alignment. Its score is excluded
    from inference to avoid selecting a path on a tested anchor sample.
    """
    pilot_blank, pilot_label = score_lattice(wrapper, features, tokens, blank_id, anchor)
    _, replay_frames = monotonic_viterbi(pilot_blank, pilot_label)
    samples = {condition: [] for condition in ("anchor", "shifted")}
    saved = {"pilot_blank": pilot_blank, "pilot_label": pilot_label} if save_costs else {}
    for round_index in range(repeats):
        order = ("anchor", "shifted") if round_index % 2 == 0 else ("shifted", "anchor")
        for evaluation_order, condition in enumerate(order):
            penalty = anchor if condition == "anchor" else anchor + delta
            blank, label = score_lattice(wrapper, features, tokens, blank_id, penalty)
            cost, _ = monotonic_viterbi(blank, label)
            sample = {
                "round": round_index + 1, "evaluation_order": evaluation_order,
                "condition": condition, "penalty": penalty,
                "viterbi_cost": cost,
                "fixed_alignment_cost": fixed_alignment_cost(blank, label, replay_frames),
                "mean_blank_cost": float(np.mean(blank)),
                "mean_label_cost": float(np.mean(label)) if label.size else None,
            }
            samples[condition].append(sample)
            if save_costs and round_index == 0:
                saved[f"blank_{condition}_first"] = blank
                saved[f"label_{condition}_first"] = label
            if on_sample is not None:
                on_sample(sample)
    return samples, replay_frames, saved


def summarize_candidate_samples(samples, blank_count, delta, confidence, cost_margin, score_margin):
    checks = {}
    metrics = (
        ("viterbi_cost", delta * blank_count, cost_margin),
        ("fixed_alignment_cost", delta * blank_count, cost_margin),
        ("mean_blank_cost", delta, score_margin),
        ("mean_label_cost", 0.0, score_margin),
    )
    for metric, expected, margin in metrics:
        anchor = [sample[metric] for sample in samples["anchor"]]
        shifted = [sample[metric] for sample in samples["shifted"]]
        checks[metric] = (None if anchor[0] is None else
                          mean_shift_check(anchor, shifted, expected, confidence, margin))
    assessments = [check["assessment"] for check in checks.values() if check is not None]
    if "material_difference_detected" in assessments:
        overall = "material_difference_detected"
    elif all(value == "within_practical_margin" for value in assessments):
        overall = "within_practical_margins"
    else:
        overall = "uncertain_collect_more_reads"
    return checks, overall


def load_inputs(args):
    summary = json.loads((args.probe_dir / "summary.json").read_text())
    with (args.probe_dir / "nbest.jsonl").open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    config_path = args.probe_dir / "rasr.probe.config"
    config = read_rasr_config(config_path)
    validate_supported_config(config)
    hardware = require_joint_l2_analysis_settings(config)
    if summary.get("joint_l2_converter_hardware_settings") != hardware:
        raise ValueError("Probe summary/config joint L2 ADC settings disagree")
    lexicon = Path(config.get("lib-rasr", "lexicon.file"))
    mapping = read_token_mapping(lexicon)
    pronunciations = read_lexicon(lexicon)
    if int(config.get("lib-rasr.label-scorer", "start-label-index")) != mapping["<blank>"]:
        raise ValueError("Initial-history label disagrees with the blank output index")
    anchor = float(config.get("lib-rasr.label-scorer.recognition", "blank-penalty"))
    if anchor != float(summary["anchor_penalty"]):
        raise ValueError("Probe summary/config anchor penalties disagree")
    if Path(config.get("lib-rasr.label-scorer.recognition", "model-path")).resolve() != Path(summary["checkpoint"]).resolve():
        raise ValueError("Probe summary/config checkpoints disagree")
    if summary.get("sentence_end_fallback") or summary.get("presoftmax_blank_penalty"):
        raise ValueError("Probe used an unsupported fallback or penalty convention")
    return summary, records, config, mapping, pronunciations, anchor


def load_wrapper(summary, effective_path, scorer_type):
    sys.path.insert(0, summary["returnn_root"])
    sys.path.insert(0, str(DEFAULT_SISYPHUS))
    sys.path.insert(0, str(SETUP))
    from returnn.config import Config, set_global_config
    config = Config()
    set_global_config(config)
    config.load_file(io.StringIO(config_without_decoder(Path(summary["returnn_config"]))))
    from librasr import Configuration
    from recipe.model_pipelines.common.pyscorers import (
        FixedContextTransducerPy, FixedContextTransducerPy_Cached,
    )
    rasr = Configuration()
    rasr.set_from_file(str(effective_path))
    rasr.set_selection("lib-rasr.label-scorer")
    cls = FixedContextTransducerPy_Cached if scorer_type.endswith("-cache") else FixedContextTransducerPy
    return cls(rasr)


def run_validation(args, summary, records, config, mapping, pronunciations, anchor):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: use an allocated GPU node in the recognition container")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    recognition = "lib-rasr.label-scorer.recognition"
    config.set(recognition, "device", args.device)
    config.set("lib-rasr.label-scorer", "debug-joint", "no")
    for kind in ("log", "warning", "error", "statistics"):
        config.set("*", f"{kind}.channel", str(args.output_dir / "rasr.validation.log"))
    effective = args.output_dir / "rasr.validation.config"
    with effective.open("w") as stream:
        config.write(stream)
    wrapper = load_wrapper(summary, effective, config.get("lib-rasr.label-scorer", "type"))
    if wrapper._model.target_size != len(mapping):
        raise ValueError("Lexicon inventory size does not match scorer vocabulary")
    blank_id = mapping["<blank>"]
    shifted_penalty = anchor + args.penalty_delta
    results = {
        "probe_dir": str(args.probe_dir), "checkpoint": config.get(recognition, "model-path"),
        "anchor_penalty": anchor, "shifted_penalty": shifted_penalty,
        "mapping_source": "XML inventory order, RASR one-state monophone emission index = phoneme ID - 1",
        "symbol_to_output_index": mapping, "device": args.device,
        "joint_l2_converter_hardware_settings": require_joint_l2_analysis_settings(config),
        "mode": "fresh_read_statistical", "repeats_per_condition": args.repeats,
        "confidence": args.confidence, "cost_mean_margin": args.cost_mean_margin,
        "score_mean_margin": args.score_mean_margin,
        "noise_handling": "Unmodified readout noise; advancing random streams; no seed resets or RNG replay",
        "evaluation_order": "Alternating anchor/shifted and shifted/anchor rounds",
        "scope": "Repeated noisy complete-transcript RNA rescoring; no LM/ILM or beam search",
        "requested_transcripts": args.transcript,
        "candidates": [],
    }
    with (args.output_dir / "validation.jsonl").open("w") as stream, \
            (args.output_dir / "read_samples.jsonl").open("w") as sample_stream, torch.no_grad():
        for index, record in enumerate(records[:args.num_seqs]):
            selected = candidates_for_validation(record["candidates"], args.max_candidates, args.transcript)
            if not selected:
                continue
            encoder_path = args.probe_dir / f"encoder_{index:03d}.npy"
            features = np.ascontiguousarray(np.load(encoder_path, allow_pickle=False), dtype=np.float32)
            if features.ndim != 2 or len(features) != record["encoder_frames"] or not np.all(np.isfinite(features)):
                raise ValueError(f"Invalid encoder features for {record['seq_tag']}")
            for candidate in selected:
                tokens, symbols = candidate_tokens(candidate, pronunciations, mapping)
                if len(tokens) > len(features):
                    raise ValueError("Candidate has more tokens than encoder frames")
                blank_count = len(features) - len(tokens)
                if candidate.get("blank_count") != blank_count:
                    raise ValueError("Reconstructed token count disagrees with probe inference")
                print(f"\n{record['seq_tag']} rank {candidate['rank']}: {candidate['transcription']!r}\n"
                      f"  Collecting {args.repeats} fresh reads per penalty, plus one independent pilot.", flush=True)

                def on_sample(sample):
                    sample_stream.write(json.dumps({"seq_tag": record["seq_tag"], "rank": candidate["rank"],
                                                    "transcription": candidate["transcription"], **sample},
                                                   allow_nan=False) + "\n")
                    sample_stream.flush()
                    if sample["evaluation_order"] == 1 and (sample["round"] % 5 == 0 or sample["round"] == args.repeats):
                        print(f"  Completed {sample['round']}/{args.repeats} rounds", flush=True)

                samples, replay_frames, saved = collect_noisy_samples(
                    wrapper, features, tokens, blank_id, anchor, args.penalty_delta, args.repeats,
                    on_sample=on_sample, save_costs=args.save_costs,
                )
                checks, assessment = summarize_candidate_samples(
                    samples, blank_count, args.penalty_delta, args.confidence,
                    args.cost_mean_margin, args.score_mean_margin,
                )
                exported = float(candidate["traceback_terminal_cost"])
                anchor_costs = [sample["viterbi_cost"] for sample in samples["anchor"]]
                output = {
                    "seq_tag": record["seq_tag"], "reference": record["reference"],
                    "rank": candidate["rank"], "transcription": candidate["transcription"],
                    "word_errors": candidate["word_errors"], "tokens": symbols, "token_ids": tokens,
                    "encoder_frames": len(features), "blank_count": blank_count,
                    "exported_cost": exported,
                    "exported_minus_independent_anchor_mean": exported - float(np.mean(anchor_costs)),
                    "anchor_sample_fraction_at_or_below_exported": float(np.mean(np.asarray(anchor_costs) <= exported)),
                    "export_comparison_note": "Descriptive only: search-selected noisy path and independently optimized alignments differ",
                    "checks": checks, "assessment": assessment,
                    "pilot_emission_frames": replay_frames, "read_samples": samples,
                }
                results["candidates"].append(output)
                stream.write(json.dumps(output, allow_nan=False) + "\n")
                stream.flush()
                check = checks["viterbi_cost"]
                print(f"  Mean cost shift={check['observed_mean_shift']:.6f}, "
                      f"expected={check['expected_shift']:.6f}, CI={check['mean_shift_ci']}; {assessment}", flush=True)
                if args.save_costs:
                    np.savez_compressed(args.output_dir / f"costs_{index:03d}_rank{candidate['rank']}.npz",
                                        token_ids=np.array(tokens, dtype=np.int64), **saved)
    candidates = results["candidates"]
    results["totals"] = {
        "validated_candidates": len(candidates),
        "noisy_evaluations": len(candidates) * (2 * args.repeats + 1),
        "assessments": dict(Counter(c["assessment"] for c in candidates)),
    }
    with (args.output_dir / "validation.json").open("w") as stream:
        json.dump(results, stream, indent=2, allow_nan=False)
        stream.write("\n")
    report = format_results(results)
    (args.output_dir / "validation.txt").write_text(report)
    print(f"\n{report}\nValidation results: {args.output_dir}", flush=True)


def format_results(results):
    totals = results["totals"]
    lines = [results["scope"], f"Probe: {results['probe_dir']}",
             f"Penalties: {results['anchor_penalty']} -> {results['shifted_penalty']}",
             f"Candidates: {totals['validated_candidates']}",
             f"Fresh reads per condition: {results['repeats_per_condition']} (+ one separate pilot per candidate)",
             f"Read evaluations: {totals['noisy_evaluations']}",
             f"Confidence level: {results['confidence']:.1%}; unpaired Welch intervals",
             f"Joint L2 ADC: {results['joint_l2_converter_hardware_settings']}",
             results["noise_handling"], f"Assessments: {totals['assessments']}", ""]
    for candidate in results["candidates"]:
        lines.extend([
            f"{candidate['seq_tag']} rank {candidate['rank']}: {candidate['transcription']!r}",
            f"  Overall: {candidate['assessment']}; tokens={len(candidate['token_ids'])}, blanks={candidate['blank_count']}",
            f"  Exported noisy cost={candidate['exported_cost']:.6f}; "
            f"exported minus independent anchor mean={candidate['exported_minus_independent_anchor_mean']:+.6f} (descriptive only)",
        ])
        for name, check in candidate["checks"].items():
            if check is None:
                lines.append(f"  {name}: not applicable (no candidate labels)")
                continue
            a, b = check["anchor"], check["shifted"]
            lo, hi = check["mean_shift_ci"]
            lines.extend([
                f"  {name}: anchor mean={a['mean']:.6f}, SD={a['stddev']:.6f}; "
                f"shifted mean={b['mean']:.6f}, SD={b['stddev']:.6f}",
                f"    Mean shift={check['observed_mean_shift']:.6f}; expected={check['expected_shift']:.6f}; "
                f"CI=[{lo:.6f}, {hi:.6f}]; target inside CI={check['target_inside_ci']}",
                f"    Practical residual margin=+/-{check['practical_margin']}; {check['assessment']}",
            ])
    lines.extend(["", "Fresh-read variation is measured, not treated as an error. A wide CI containing",
                  "the target is inconclusive, not proof of agreement. Intervals are approximate and",
                  "per-metric, not simultaneous across all candidates. Exported scores are search-selected",
                  "noisy observations; these comparisons do not certify export bookkeeping or WER optimality."])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-dir", type=Path,
                        default=PROBE_ROOT / "bp1.44")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--num-seqs", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=5,
                        help="Distinct transcripts per utterance; includes best-error candidate")
    parser.add_argument("--transcript", action="append",
                        help="Only validate this exact retained transcript; repeat to select several")
    parser.add_argument("--repeats", type=int, default=20,
                        help="Fresh independent evaluations per penalty condition")
    parser.add_argument("--penalty-delta", type=float, default=1.0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--cost-mean-margin", type=float, default=0.1,
                        help="Practical agreement margin for expected path-cost shifts")
    parser.add_argument("--score-mean-margin", type=float, default=0.01,
                        help="Practical agreement margin for expected mean lattice-score shifts")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--save-costs", action="store_true",
                        help="Save pilot and first fresh score arrays per condition; all scalar reads are always saved")
    parser.add_argument("--inspect-only", action="store_true",
                        help="CPU-only mapping/token reconstruction; no model loading or output writes")
    args = parser.parse_args()
    args.probe_dir = args.probe_dir.resolve()
    if args.num_seqs < 1 or args.max_candidates < 1:
        parser.error("num-seqs and max-candidates must be positive")
    if args.repeats < 2:
        parser.error("repeats must be at least 2 to estimate uncertainty")
    if not 0 < args.confidence < 1:
        parser.error("confidence must be between zero and one")
    if not math.isfinite(args.penalty_delta) or args.penalty_delta == 0:
        parser.error("penalty-delta must be finite and nonzero")
    if any(not math.isfinite(x) or x < 0 for x in (args.score_mean_margin, args.cost_mean_margin)):
        parser.error("practical margins must be finite and nonnegative")
    summary, records, config, mapping, pronunciations, anchor = load_inputs(args)
    if not math.isfinite(anchor + args.penalty_delta):
        parser.error("shifted penalty must be finite")
    found = set()
    for index, record in enumerate(records[:args.num_seqs]):
        selected = candidates_for_validation(record["candidates"], args.max_candidates, args.transcript)
        if not selected:
            continue
        if not (args.probe_dir / f"encoder_{index:03d}.npy").is_file():
            raise FileNotFoundError("Missing saved encoder features: rerun the probe with --save-encoder")
        for candidate in selected:
            found.add(candidate["transcription"])
            tokens, symbols = candidate_tokens(candidate, pronunciations, mapping)
            print(f"{record['seq_tag']} rank {candidate['rank']}: {candidate['transcription']!r}\n"
                  f"  tokens={symbols}\n  output IDs={tokens}; "
                  f"blanks={record['encoder_frames']-len(tokens)}", flush=True)
    if args.transcript and set(args.transcript) - found:
        parser.error(f"Requested transcripts not selected in the first num-seqs records: {sorted(set(args.transcript) - found)}")
    if not found:
        parser.error("No usable candidates selected")
    if args.inspect_only:
        return
    args.output_dir = (args.output_dir or args.probe_dir / "validation_statistical").resolve()
    run_validation(args, summary, records, config, mapping, pronunciations, anchor)


if __name__ == "__main__":
    main()
