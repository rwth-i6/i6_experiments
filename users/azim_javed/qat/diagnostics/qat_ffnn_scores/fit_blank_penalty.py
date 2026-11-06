"""Fit a blank-cost offset for the fully on-device FFNN transducer.

Uses the score lattices saved by reference_scoring.py. The digital model's
reference-conditioned forward-backward alignment provides the weights; both
models are evaluated at identical acoustic frames and reference prefixes.

The result matches teacher-forced scores, not necessarily decoding WER.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SCORES = Path(__file__).resolve().parents[2] / "output/diagnostics/qat_ffnn_scores"


def log_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    return logits - np.logaddexp.reduce(logits, axis=-1, keepdims=True)


def transition_posteriors(log_probs: np.ndarray, labels: np.ndarray, blank: int):
    """Digital P(blank/next-reference-label transition | complete reference).

    Follows the time-synchronous topology of reference_scoring.py: either
    transition consumes exactly one encoder frame. Includes blank transitions
    after the last reference label.
    """
    frames, positions, _ = log_probs.shape
    num_labels = len(labels)
    assert positions == num_labels + 1 and num_labels <= frames

    alpha = np.full((frames, positions), -np.inf)
    for t in range(frames):
        for u in range(positions):
            before_blank = 0.0 if (t, u) == (0, 0) else (alpha[t - 1, u] if t else -np.inf)
            before_label = 0.0 if (t, u) == (0, 1) else (alpha[t - 1, u - 1] if t and u else -np.inf)
            label_score = before_label + log_probs[t, u - 1, labels[u - 1]] if u else -np.inf
            alpha[t, u] = np.logaddexp(before_blank + log_probs[t, u, blank], label_score)

    likelihood = alpha[-1, -1]
    if not np.isfinite(likelihood):
        raise ValueError("Reference has no finite-probability alignment")
    beta = np.full_like(alpha, -np.inf)
    beta[-1, -1] = 0.0
    for t in range(frames - 2, -1, -1):
        for u in range(positions):
            blank_score = log_probs[t + 1, u, blank] + beta[t + 1, u]
            label_score = log_probs[t + 1, u, labels[u]] + beta[t + 1, u + 1] if u < num_labels else -np.inf
            beta[t, u] = np.logaddexp(blank_score, label_score)

    blank_weights = np.zeros((frames, positions))
    label_weights = np.zeros((frames, num_labels))
    for t in range(frames):
        for u in range(positions):
            before = 0.0 if (t, u) == (0, 0) else (alpha[t - 1, u] if t else -np.inf)
            blank_weights[t, u] = np.exp(before + log_probs[t, u, blank] + beta[t, u] - likelihood)
            if u < num_labels:
                label_weights[t, u] = np.exp(
                    before + log_probs[t, u, labels[u]] + beta[t, u + 1] - likelihood
                )

    if not np.allclose(blank_weights.sum(axis=1) + label_weights.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Alignment transition weights do not sum to one per frame")
    if not np.allclose(label_weights.sum(axis=0), 1.0, atol=1e-5):
        raise ValueError("Each reference label should be emitted exactly once")
    return blank_weights, label_weights


def weighted_fit(rows, *, equal_utterances: bool):
    """Fit an offset with either equal utterance or pooled transition weights."""
    if equal_utterances:
        normalized_rows = [(w / np.sum(w), d) for w, d in rows]
        normalization = len(rows)
    else:
        normalized_rows = rows
        normalization = sum(np.sum(w) for w, _ in rows)

    offset = sum(np.sum(w * d) for w, d in normalized_rows) / normalization
    before = sum(np.sum(w * d**2) for w, d in normalized_rows) / normalization
    after = sum(np.sum(w * (d - offset) ** 2) for w, d in normalized_rows) / normalization
    return offset, np.sqrt(before), np.sqrt(after)


def read_pair(digital_dir: Path, device_dir: Path, filename: str):
    with np.load(digital_dir / filename) as digital, np.load(device_dir / filename) as device:
        if (digital["logits"].shape != device["logits"].shape or
                not np.array_equal(digital["labels"], device["labels"]) or
                int(digital["blank_label"]) != int(device["blank_label"])):
            raise ValueError(f"Mismatched lattice, reference labels, or blank ID: {filename}")
        return (digital["logits"], device["logits"], digital["labels"], int(digital["blank_label"]))


def analyze_scores(digital_dir: Path, device_dir: Path) -> dict:
    """Fit penalties from the two extraction jobs' score directories."""
    with (digital_dir / "manifest.json").open() as stream:
        digital_manifest = json.load(stream)
    with (device_dir / "manifest.json").open() as stream:
        device_manifest = json.load(stream)
    if not digital_manifest or set(digital_manifest) != set(device_manifest):
        raise ValueError("Digital and on-device manifests must contain the same nonempty set of utterances")

    fits = {"emission: reference token": [], "emission: all nonblank": [],
            "blank: all nonblank": [], "path: all nonblank": []}
    for tag in sorted(digital_manifest):
        if digital_manifest[tag]["file"] != device_manifest[tag]["file"]:
            raise ValueError(f"Different NPZ filenames for {tag}")
        digital, device, labels, blank = read_pair(digital_dir, device_dir, digital_manifest[tag]["file"])
        if blank != digital.shape[-1] - 1:
            raise ValueError(f"Expected blank to be last vocabulary symbol: {tag}")
        digital_lp, device_lp = log_softmax(digital), log_softmax(device)
        blank_weights, label_weights = transition_posteriors(digital_lp, labels, blank)
        occupancy = blank_weights.copy()
        occupancy[:, :len(labels)] += label_weights

        # Adding b to blank's negative-log-probability score raises each
        # nonblank-vs-blank margin by exactly b (no renormalization needed).
        def nonblank_margin(lp):
            return np.logaddexp.reduce(lp[..., :blank], axis=-1) - lp[..., blank]

        all_difference = nonblank_margin(digital_lp) - nonblank_margin(device_lp)
        label_indices = labels[None, :, None]
        digital_label = np.take_along_axis(digital_lp[:, :len(labels)],
                                           np.broadcast_to(label_indices, (len(digital), len(labels), 1)), -1)[..., 0]
        device_label = np.take_along_axis(device_lp[:, :len(labels)],
                                          np.broadcast_to(label_indices, (len(device), len(labels), 1)), -1)[..., 0]
        label_difference = (digital_label - digital_lp[:, :len(labels), blank]
                            - device_label + device_lp[:, :len(labels), blank])

        fits["emission: reference token"].append((label_weights, label_difference))
        fits["emission: all nonblank"].append((label_weights, all_difference[:, :len(labels)]))
        fits["blank: all nonblank"].append((blank_weights, all_difference))
        fits["path: all nonblank"].append((occupancy, all_difference))

    results = {"utterances": len(digital_manifest), "digital_scores": str(digital_dir),
               "on_device_scores": str(device_dir), "fits": []}
    for name, rows in fits.items():
        for equal_utterances, averaging in ((True, "equal utterances"), (False, "pooled transitions")):
            offset, before, after = weighted_fit(rows, equal_utterances=equal_utterances)
            results["fits"].append({"margin": name, "averaging": averaging, "blank_penalty": float(offset),
                                    "rmse_before": float(before), "rmse_after": float(after)})
    return results


def format_results(results: dict) -> str:
    lines = [f"Digital scores: {results['digital_scores']}",
             f"Fully on-device scores: {results['on_device_scores']}",
             f"Utterances: {results['utterances']}",
             "Positive b penalizes blank in the on-device decoder; digital uses b=0.",
             f"{'Weight / margin':32} {'Averaging':19} {'fit b':>9} {'RMSE before':>13} {'RMSE after':>12}"]
    for row in results["fits"]:
        lines.append(f"{row['margin']:32} {row['averaging']:19} {row['blank_penalty']:9.3f} "
                     f"{row['rmse_before']:13.3f} {row['rmse_after']:12.3f}")
    lines.append("Diagnostic score-matching estimates only; confirm candidates with dev-set decoding WER.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES,
                        help="Directory containing digital_model/ and base_model/ NPZ score lattices")
    args = parser.parse_args()
    print(format_results(analyze_scores(args.scores_dir / "digital_model", args.scores_dir / "base_model")), end="")


if __name__ == "__main__":
    main()
