"""Analytically minimize reference-conditioned greedy decision error.

Uses the same digital_model/ and base_model/ score lattices as
fit_blank_penalty.py. Digital forward-backward weights define expected blank
and reference-label transitions. Device logits define the greedy switches.

This optimizes a token-transition surrogate, not free-decoding word error.
Adding a blank cost does not repair nonblank token rankings or history drift.
No scores are extracted and no Sisyphus manager is started by this script.

See OPTIMAL_BLANK_PENALTY_METHODOLOGY.md for the objective, threshold sweep,
weighting, Sisyphus integration, and numerical limitations.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
from sisyphus import Job, Task, tk

from .fit_blank_penalty import log_softmax, read_pair, transition_posteriors

if TYPE_CHECKING:
    from .reference_scoring import PreparedReferenceScores


COMPONENTS = ("false_emission", "missed_emission", "wrong_token")
AVERAGING = ("pooled transitions", "equal utterances")


def group_events(thresholds, changes):
    """Combine simultaneous switches, preserving all component loss changes."""
    order = np.argsort(thresholds)
    thresholds, changes = thresholds[order], changes[order]
    unique, starts = np.unique(thresholds, return_index=True)
    return unique, np.add.reduceat(changes, starts, axis=0)


def decision_events(device_logits, labels, blank_weights, label_weights):
    """Return thresholds, loss changes, all-blank loss, and correct-token weight.

    Component order: false emissions, missed emissions, wrong-token emissions.
    At exact blank/token ties we choose blank. Nonblank ties use NumPy's first
    argmax; different decoder tie-breaking can affect exact ties.
    """
    logits = np.asarray(device_logits, dtype=np.float64)
    best_tokens = np.argmax(logits[..., :-1], axis=-1)
    best_logits = np.max(logits[..., :-1], axis=-1)
    thresholds = logits[..., -1] - best_logits
    label_mass = np.zeros_like(blank_weights)
    label_mass[:, :len(labels)] = label_weights
    correct = np.zeros_like(blank_weights, dtype=bool)
    correct[:, :len(labels)] = best_tokens[:, :len(labels)] == labels[None, :]
    wrong_mass = label_mass * ~correct

    changes = np.stack((blank_weights, -label_mass, wrong_mass), axis=-1)
    active = (blank_weights + label_mass) > 0
    return (
        thresholds[active], changes[active],
        np.array([0.0, label_mass.sum(), 0.0]),
        float((label_mass * correct).sum()),
    )


def component_dict(values, costs):
    # Cumulative summation can leave tiny negative residuals at the all-emitting end.
    values = np.maximum(values, 0.0)
    result = {name: float(value) for name, value in zip(COMPONENTS, values)}
    result["risk"] = float(values @ costs)
    return result


def interval_representative(lower, upper, lower_inclusive):
    if upper is None:
        value = lower + 1.0
        if not np.isfinite(value) or value <= lower:
            value = float(np.nextafter(lower, np.inf))
    elif lower == upper:
        value = lower
    else:
        value = lower / 2.0 + upper / 2.0
        if value <= lower and not lower_inclusive:
            value = upper  # Right endpoint belongs to this interval (blank wins ties).
    if not np.isfinite(value):
        raise ValueError("Cannot represent an optimal interval with a finite penalty")
    return float(value)


def optimize_events(thresholds, changes, initial, *, costs, min_penalty=0.0,
                    max_penalty=None, penalties=(0.0, 2.0, 4.0, 7.3)):
    """Exact threshold sweep for one averaging scheme.

    thresholds must be sorted and unique. Inputs are already normalized.
    All intervals are right-closed because emission requires b > threshold.
    The first interval includes min_penalty; later intervals exclude their left
    endpoint. With no maximum, the last interval extends to positive infinity.
    """
    costs = np.asarray(costs, dtype=np.float64)
    if costs.shape != (3,) or not np.all(np.isfinite(costs)) or np.any(costs < 0):
        raise ValueError("Three finite nonnegative error costs are required")
    if not np.isfinite(min_penalty):
        raise ValueError("min_penalty must be finite")
    if max_penalty is not None and (not np.isfinite(max_penalty) or max_penalty < min_penalty):
        raise ValueError("max_penalty must be finite and >= min_penalty")
    if not np.all(np.isfinite(penalties)):
        raise ValueError("Comparison penalties must be finite")

    states = np.vstack((initial, initial + np.cumsum(changes, axis=0)))
    states = np.maximum(states, 0.0)
    risks = states @ costs
    first = int(np.searchsorted(thresholds, min_penalty, side="left"))
    last = (len(thresholds) if max_penalty is None else
            int(np.searchsorted(thresholds, max_penalty, side="left")))
    minimum = float(np.min(risks[first:last + 1]))
    # Numerical ties only; this is not a tolerance band for near-optimal penalties.
    best_indices = np.flatnonzero(np.isclose(
        risks[first:last + 1], minimum, rtol=1e-12, atol=1e-12
    )) + first
    intervals = []
    for index in best_indices:
        lower = min_penalty if index == first else float(thresholds[index - 1])
        upper = float(thresholds[index]) if index < len(thresholds) else max_penalty
        if max_penalty is not None and (upper is None or upper > max_penalty):
            upper = max_penalty
        inclusive = bool(index == first)
        representative = interval_representative(lower, upper, inclusive)
        intervals.append({
            "lower": float(lower), "upper": upper,
            "lower_inclusive": inclusive, "upper_inclusive": upper is not None,
            "representative_penalty": representative,
            **component_dict(states[index], costs),
        })

    comparisons = []
    for penalty in penalties:
        index = np.searchsorted(thresholds, penalty, side="left")
        comparisons.append({"penalty": float(penalty), **component_dict(states[index], costs)})
    return {"minimum_risk": minimum, "optimal_intervals": intervals,
            "comparisons": comparisons}


def analyze_scores(digital_dir, device_dir, *, costs=(1.0, 1.0, 1.0),
                   min_penalty=0.0, max_penalty=None,
                   penalties=(0.0, 2.0, 4.0, 7.3), num_utterances=None):
    with (digital_dir / "manifest.json").open() as stream:
        digital_manifest = json.load(stream)
    with (device_dir / "manifest.json").open() as stream:
        device_manifest = json.load(stream)
    if not digital_manifest or set(digital_manifest) != set(device_manifest):
        raise ValueError("Digital and device manifests must match and be nonempty")
    tags = sorted(digital_manifest)
    if num_utterances is not None:
        if num_utterances < 1:
            raise ValueError("num_utterances must be positive")
        tags = tags[:num_utterances]

    thresholds_by_utterance, changes_by_utterance = [], []
    initial = np.zeros((2, 3))
    correct_mass = np.zeros(2)
    label_mass = np.zeros(2)
    total_frames = 0
    for tag in tags:
        filename = digital_manifest[tag]["file"]
        if filename != device_manifest[tag]["file"]:
            raise ValueError(f"Different NPZ filenames for {tag}")
        digital, device, labels, blank = read_pair(digital_dir, device_dir, filename)
        if (digital.ndim != 3 or digital.shape[0] == 0 or digital.shape[-1] < 2 or
                blank != digital.shape[-1] - 1 or
                digital.shape[1] != len(labels) + 1 or len(labels) > len(digital) or
                np.any(labels < 0) or np.any(labels >= blank)):
            raise ValueError(f"Invalid lattice, labels, or blank ID: {tag}")
        if not np.all(np.isfinite(digital)) or not np.all(np.isfinite(device)):
            raise ValueError(f"Nonfinite logits: {tag}")
        blank_weights, label_weights = transition_posteriors(log_softmax(digital), labels, blank)
        thresholds, changes, baseline, correct = decision_events(
            device, labels, blank_weights, label_weights
        )
        # Equal-utterance weighting gives each utterance total occupancy weight 1.
        # Pooled weighting gives each encoder frame equal total weight.
        frames = len(digital)
        combined = np.concatenate((changes, changes / frames), axis=1)
        thresholds, combined = group_events(thresholds, combined)
        thresholds_by_utterance.append(thresholds)
        changes_by_utterance.append(combined)
        initial += np.stack((baseline, baseline / frames))
        correct_mass += (correct, correct / frames)
        label_mass += (len(labels), len(labels) / frames)
        total_frames += frames

    thresholds, changes = group_events(
        np.concatenate(thresholds_by_utterance), np.concatenate(changes_by_utterance)
    )
    normalizations = (total_frames, len(tags))
    fits = []
    for index, (averaging, normalization) in enumerate(zip(AVERAGING, normalizations)):
        fit = optimize_events(
            thresholds, changes[:, index * 3:(index + 1) * 3] / normalization,
            initial[index] / normalization, costs=costs,
            min_penalty=min_penalty, max_penalty=max_penalty, penalties=penalties,
        )
        fit["averaging"] = averaging
        fit["correct_best_token_emission_coverage"] = (
            float(correct_mass[index] / label_mass[index]) if label_mass[index] else None
        )
        fits.append(fit)
    return {
        "digital_scores": str(digital_dir), "on_device_scores": str(device_dir),
        "utterances": len(tags), "encoder_frames": total_frames,
        "unique_switching_thresholds": len(thresholds),
        "costs": dict(zip(COMPONENTS, map(float, costs))),
        "min_penalty": min_penalty, "max_penalty": max_penalty,
        "objective": "Reference-conditioned expected greedy token-transition error; not WER",
        "tie_rule": "Blank wins blank/token ties; first token ID wins nonblank ties",
        "equal_utterances_weighting": "Normalize each utterance by its encoder-frame count",
        "fits": fits,
    }


def format_results(results):
    lines = [
        results["objective"],
        f"Digital scores: {results['digital_scores']}",
        f"On-device scores: {results['on_device_scores']}",
        f"Utterances: {results['utterances']}; encoder frames: {results['encoder_frames']}",
        f"Unique switching thresholds: {results['unique_switching_thresholds']}",
        f"Error costs: {results['costs']}",
        "Positive penalty makes blank less attractive. Blank wins exact ties.",
        "Components are expected errors per frame (not word-error percentages).",
        "Equal utterances: average each utterance's per-frame error rate.",
    ]
    for fit in results["fits"]:
        coverage = fit["correct_best_token_emission_coverage"]
        lines.extend(["", fit["averaging"],
                      "Correct best-nonblank coverage at digital emissions: " +
                      (f"{coverage:.4%}" if coverage is not None else "undefined (no labels)"),
                      f"Minimum risk: {fit['minimum_risk']:.9f}",
                      f"{'Penalty / optimal interval':38} {'Risk':>12} {'False':>12} "
                      f"{'Missed':>12} {'Wrong token':>12}"])
        rows = []
        for interval in fit["optimal_intervals"]:
            left = "[" if interval["lower_inclusive"] else "("
            upper = "infinity)" if interval["upper"] is None else f"{interval['upper']:.9g}]"
            name = f"{left}{interval['lower']:.9g}, {upper}"
            rows.append((name, interval))
        rows.extend((f"b={row['penalty']:.9g}", row) for row in fit["comparisons"])
        for name, row in rows:
            lines.append(f"{name:38} {row['risk']:12.9f} {row['false_emission']:12.9f} "
                         f"{row['missed_emission']:12.9f} {row['wrong_token']:12.9f}")
        lines.append("Representative optimal penalty: " +
                     f"{fit['optimal_intervals'][0]['representative_penalty']:.9g}")
    lines.extend(["", "Fixed correct-reference histories only. Validate with free decoding:",
                  "the penalty changes visited histories, and token-transition error is not WER."])
    return "\n".join(lines) + "\n"


class OptimalBlankPenaltyJob(Job):
    """Minimize posterior-weighted greedy decision error from extracted scores."""

    def __init__(
        self,
        digital_scores: tk.Path,
        on_device_scores: tk.Path,
        costs: tuple[float, float, float] = (1.0, 1.0, 1.0),
        min_penalty: float = 0.0,
        max_penalty: Optional[float] = None,
        penalties: tuple[float, ...] = (0.0, 2.0, 4.0, 7.3),
    ):
        self.digital_scores = digital_scores
        self.on_device_scores = on_device_scores
        self.costs = costs
        self.min_penalty = min_penalty
        self.max_penalty = max_penalty
        self.penalties = penalties
        self.out_summary = self.output_path("optimal_blank_penalty.json")
        self.out_report = self.output_path("optimal_blank_penalty.txt")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 2, "mem": 16, "time": 2})

    def run(self):
        results = analyze_scores(
            Path(self.digital_scores.get_path()), Path(self.on_device_scores.get_path()),
            costs=self.costs, min_penalty=self.min_penalty,
            max_penalty=self.max_penalty, penalties=self.penalties,
        )
        with open(self.out_summary.get_path(), "w") as summary_file:
            json.dump(results, summary_file, indent=2, allow_nan=False)
            summary_file.write("\n")
        with open(self.out_report.get_path(), "w") as report_file:
            report_file.write(format_results(results))


def run_optimal_blank_penalty(
    prepared: "PreparedReferenceScores",
    *,
    costs: tuple[float, float, float] = (1.0, 1.0, 1.0),
    min_penalty: float = 0.0,
    max_penalty: Optional[float] = None,
    penalties: tuple[float, ...] = (0.0, 2.0, 4.0, 7.3),
) -> OptimalBlankPenaltyJob:
    """Create the optimization job using prepare_scores() extraction outputs.

    costs are (false emission, missed emission, wrong token), respectively.
    """
    output_prefix = prepared.output_prefix
    job = OptimalBlankPenaltyJob(
        digital_scores=prepared.score_dirs["digital_model"],
        on_device_scores=prepared.score_dirs["base_model"],
        costs=costs, min_penalty=min_penalty, max_penalty=max_penalty, penalties=penalties,
    )
    job.add_alias(f"{output_prefix}/optimal_blank_penalty")
    tk.register_output(f"{output_prefix}/optimal_blank_penalty.json", job.out_summary)
    tk.register_output(f"{output_prefix}/optimal_blank_penalty.txt", job.out_report)
    return job
