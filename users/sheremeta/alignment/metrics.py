"""Metrics for perturbation consistency and cross-aligner agreement of onset timestamps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# report the fraction of boundaries at or below each tolerance, in milliseconds
DEFAULT_TOL_MS = (20.0, 40.0, 80.0)
BOOTSTRAP_SAMPLES = 500
_BOOTSTRAP_SEED = 0


@dataclass
class DeviationStats:
    """Summary statistics for absolute boundary deviations, in milliseconds."""

    n: int
    median_ms: float
    p90_ms: float
    p95_ms: float
    mean_ms: float
    frac_within: Dict[float, float]

    def as_row(self) -> Dict[str, float]:
        row = {
            "n": self.n,
            "median_ms": self.median_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "mean_ms": self.mean_ms,
        }
        for tol, f in self.frac_within.items():
            row[f"within_{int(tol)}ms"] = f
        return row


def deviation_stats(
    dev_ms: Sequence[float], tolerances: Sequence[float] = DEFAULT_TOL_MS
) -> DeviationStats:
    """Summarize a flat sequence of absolute deviations in milliseconds.

    An empty input produces NaN-valued statistics. Non-finite deviations raise
    :class:`ValueError` because silently dropping them would bias the summary.
    """
    d = np.asarray(dev_ms, dtype=np.float64)
    if d.size == 0:
        return DeviationStats(
            0,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            {float(t): float("nan") for t in tolerances},
        )
    if not np.all(np.isfinite(d)):
        raise ValueError("deviations contain NaN or infinite values")
    return DeviationStats(
        n=int(d.size),
        median_ms=float(np.median(d)),
        p90_ms=float(np.percentile(d, 90)),
        p95_ms=float(np.percentile(d, 95)),
        mean_ms=float(np.mean(d)),
        # keep the threshold inclusive: frame-quantized onsets often land exactly
        # on a tolerance and should count as being within it
        frac_within={float(t): float(np.mean(d <= t)) for t in tolerances},
    )


@dataclass
class FailureCounts:
    """Counts of dropped utterance/source or utterance/transform pairs.

    Failure counts remain separate from deviation statistics so a method cannot
    appear more stable merely by failing on difficult inputs.
    """

    missing_output: int = 0
    count_mismatch: int = 0
    nonfinite_output: int = 0
    nonmonotonic_output: int = 0
    label_mismatch: int = 0

    @property
    def total(self) -> int:
        return (
            self.missing_output
            + self.count_mismatch
            + self.nonfinite_output
            + self.nonmonotonic_output
            + self.label_mismatch
        )

    def rate(self, attempted: int) -> float:
        return self.total / attempted if attempted else float("nan")

    def as_dict(self) -> Dict[str, int]:
        return {
            "missing_output": self.missing_output,
            "count_mismatch": self.count_mismatch,
            "nonfinite_output": self.nonfinite_output,
            "nonmonotonic_output": self.nonmonotonic_output,
            "label_mismatch": self.label_mismatch,
            "total": self.total,
        }

    def add(self, reason: str):
        setattr(self, reason, getattr(self, reason) + 1)


def onset_failure_reason(onsets_s: np.ndarray) -> Optional[str]:
    """Return the failure category for an unusable onset array, otherwise ``None``."""
    arr = np.asarray(onsets_s, dtype=np.float64)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        return "nonfinite_output"
    if arr.size and (np.any(arr < 0) or np.any(np.diff(arr) < 0)):
        return "nonmonotonic_output"
    return None


def bootstrap_median_ci(
    per_utt_devs: Sequence[np.ndarray],
    n_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> Tuple[float, float]:
    """Estimate a 95% CI for the pooled median with an utterance bootstrap.

    Whole utterances are resampled because boundaries from the same utterance
    are correlated. Empty utterances are ignored; fewer than two non-empty
    utterances produce an undefined interval represented by ``(nan, nan)``.
    """
    utts = [np.asarray(d, dtype=np.float64) for d in per_utt_devs]
    utts = [u for u in utts if u.size]
    if len(utts) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = len(utts)
    medians = np.empty(n_samples)
    for i in range(n_samples):
        idx = rng.integers(0, n, size=n)
        medians[i] = np.median(np.concatenate([utts[j] for j in idx]))
    return (float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5)))


# method 1: perturbation consistency
@dataclass
class TransformResult:
    """Boundary-level and utterance-level statistics for one transform."""

    boundary: DeviationStats
    utterance: DeviationStats  # distribution of per-utterance median deviations
    attempted: int  # number of utterance pairs attempted for this transform
    failures: FailureCounts

    @property
    def failure_rate(self) -> float:
        return self.failures.rate(self.attempted)


@dataclass
class PerturbationResult:
    """Aggregated perturbation-consistency metrics across all transforms."""

    per_transform: Dict[str, TransformResult]
    overall_boundary: DeviationStats  # micro-average over all surviving boundaries
    overall_utterance: DeviationStats  # distribution of per-utterance medians
    # equal-weight mean of per-transform boundary summaries
    mean_of_transforms: Dict[str, float]
    overall_ci_ms: Tuple[float, float]  # utterance-bootstrap CI for the pooled median
    n_utts: int
    failures: FailureCounts = field(default_factory=FailureCounts)
    attempted_pairs: int = 0

    @property
    def failure_rate(self) -> float:
        return self.failures.rate(self.attempted_pairs)


def perturbation_deviations_ms(
    base_onsets_s: np.ndarray, mapped_back_onsets_s: np.ndarray
) -> np.ndarray:
    """Compute index-matched onset drift for one transformed utterance.

    The metric is ``|phi_g^-1(t_k^g) - t_k|`` in milliseconds, where transformed
    onsets have already been mapped back to the original timeline.
    """
    base = np.asarray(base_onsets_s, dtype=np.float64)
    back = np.asarray(mapped_back_onsets_s, dtype=np.float64)
    if base.shape != back.shape:
        raise ValueError(f"onset count mismatch: {base.shape} vs {back.shape}")
    return np.abs(back - base) * 1000.0


def aggregate_perturbation(
    per_pair: List[tuple],  # entries are (transform_name, utterance_id, deviations_ms)
    n_utts: int,
    failures_by_transform: Optional[Dict[str, FailureCounts]] = None,
    attempted_by_transform: Optional[Dict[str, int]] = None,
    tolerances: Sequence[float] = DEFAULT_TOL_MS,
) -> PerturbationResult:
    """Aggregate utterance/transform deviations into per-transform and overall metrics."""
    failures_by_transform = failures_by_transform or {}
    attempted_by_transform = attempted_by_transform or {}

    by_tr_bnd: Dict[str, List[float]] = {}
    by_tr_utt: Dict[str, List[float]] = {}
    by_utt: Dict[str, List[float]] = {}
    for name, utt_id, dev in per_pair:
        dev = np.asarray(dev, dtype=np.float64)
        by_tr_bnd.setdefault(name, []).extend(dev.tolist())
        if dev.size:
            by_tr_utt.setdefault(name, []).append(float(np.median(dev)))
        by_utt.setdefault(utt_id, []).extend(dev.tolist())

    names = sorted(set(by_tr_bnd) | set(failures_by_transform))
    per_transform = {}
    for name in names:
        per_transform[name] = TransformResult(
            boundary=deviation_stats(by_tr_bnd.get(name, []), tolerances),
            utterance=deviation_stats(by_tr_utt.get(name, []), tolerances),
            attempted=attempted_by_transform.get(name, 0),
            failures=failures_by_transform.get(name, FailureCounts()),
        )

    all_bnd = [d for v in by_tr_bnd.values() for d in v]
    all_utt = [m for v in by_tr_utt.values() for m in v]

    # weight transforms equally so transforms with more surviving boundaries do
    # not dominate the single-number summary
    finite_rows = [t.boundary for t in per_transform.values() if t.boundary.n > 0]
    mean_of_transforms: Dict[str, float] = {}
    if finite_rows:
        mean_of_transforms = {
            "median_ms": float(np.mean([r.median_ms for r in finite_rows])),
            "p90_ms": float(np.mean([r.p90_ms for r in finite_rows])),
            "p95_ms": float(np.mean([r.p95_ms for r in finite_rows])),
            "mean_ms": float(np.mean([r.mean_ms for r in finite_rows])),
        }
        for t in tolerances:
            mean_of_transforms[f"within_{int(t)}ms"] = float(
                np.mean([r.frac_within[float(t)] for r in finite_rows])
            )

    total_failures = FailureCounts()
    for fc in failures_by_transform.values():
        for k in (
            "missing_output",
            "count_mismatch",
            "nonfinite_output",
            "nonmonotonic_output",
            "label_mismatch",
        ):
            setattr(total_failures, k, getattr(total_failures, k) + getattr(fc, k))

    return PerturbationResult(
        per_transform=per_transform,
        overall_boundary=deviation_stats(all_bnd, tolerances),
        overall_utterance=deviation_stats(all_utt, tolerances),
        mean_of_transforms=mean_of_transforms,
        overall_ci_ms=bootstrap_median_ci([np.asarray(v) for v in by_utt.values()]),
        n_utts=n_utts,
        failures=total_failures,
        attempted_pairs=sum(attempted_by_transform.values()),
    )


# method 2: cross-aligner agreement
@dataclass
class BoundaryDisagreement:
    """Per-boundary disagreement across aligners, in milliseconds."""

    mad_ms: np.ndarray  # median distance from the consensus onset
    max_ms: (
        np.ndarray
    )  # largest distance from the consensus; exposes one-source outliers
    range_ms: (
        np.ndarray
    )  # latest minus earliest onset; the pairwise gap for two aligners


@dataclass
class FlaggedBoundary:
    """A high-disagreement boundary retained for manual inspection."""

    utt_id: str
    index: int
    label: str
    mad_ms: float
    max_ms: float
    range_ms: float
    onsets_s: List[float]


@dataclass
class AgreementResult:
    """Cross-aligner agreement metrics and the highest-disagreement boundaries."""

    mad: DeviationStats
    max_dev: DeviationStats
    span: DeviationStats  # statistics for the latest-minus-earliest onset range
    # per-source deviation from the consensus of all other sources, unavailable for M < 3
    loo_per_source: Optional[List[DeviationStats]]
    ci_ms: Dict[str, Tuple[float, float]]  # utterance-bootstrap CIs for pooled medians
    n_boundaries: int
    n_utts: int
    failures: FailureCounts
    attempted_utts: int
    flagged: List[FlaggedBoundary]  # highest max-deviation boundaries, descending

    @property
    def failure_rate(self) -> float:
        return self.failures.rate(self.attempted_utts)


def boundary_disagreement(
    onsets_by_aligner: Sequence[np.ndarray],
) -> BoundaryDisagreement:
    """Measure per-boundary disagreement among aligners for one utterance.

    The consensus is the median onset across aligners. Median absolute deviation
    describes consensus concentration, while maximum deviation and range expose
    a single aligner that disagrees with an otherwise consistent majority.
    """
    if len(onsets_by_aligner) < 2:
        raise ValueError("at least two aligners are required")
    arrays = [np.asarray(o, dtype=np.float64) for o in onsets_by_aligner]
    if len({a.size for a in arrays}) != 1:
        raise ValueError(f"onset count mismatch: {[a.size for a in arrays]}")
    arr = np.stack(arrays, axis=0)  # shape: (aligners, boundaries)
    consensus = np.median(arr, axis=0, keepdims=True)  # one consensus per boundary
    abs_dev = np.abs(arr - consensus)
    return BoundaryDisagreement(
        mad_ms=np.median(abs_dev, axis=0) * 1000.0,
        max_ms=np.max(abs_dev, axis=0) * 1000.0,
        range_ms=np.ptp(arr, axis=0) * 1000.0,
    )


def leave_one_out_deviations_ms(
    onsets_by_aligner: Sequence[np.ndarray],
) -> np.ndarray:
    """Score each aligner against a consensus that excludes that aligner.

    Returns ``|t_k^m - median_{j != m}(t_k^j)|`` in milliseconds with shape
    ``(aligners, boundaries)``. At least three aligners are required.
    """
    arrays = [np.asarray(o, dtype=np.float64) for o in onsets_by_aligner]
    if len(arrays) < 3:
        raise ValueError("leave-one-out consensus needs at least three aligners")
    if len({a.size for a in arrays}) != 1:
        raise ValueError(f"onset count mismatch: {[a.size for a in arrays]}")
    arr = np.stack(arrays, axis=0)
    out = np.empty_like(arr)
    for m in range(arr.shape[0]):
        others = np.delete(arr, m, axis=0)
        out[m] = np.abs(arr[m] - np.median(others, axis=0))
    return out * 1000.0


def aggregate_agreement(
    per_utt: List[tuple],  # (utterance_id, labels, onsets_by_aligner[, orig_boundary_indices])
    top_k_flag: int = 50,
    tolerances: Sequence[float] = DEFAULT_TOL_MS,
    precounted_failures: Optional[FailureCounts] = None,
    attempted_utts: Optional[int] = None,
) -> AgreementResult:
    """Aggregate cross-aligner onsets and retain the largest disagreements.

    ``precounted_failures`` contains failures already filtered by the caller,
    such as missing outputs or label mismatches. This function adds count and
    onset-validity failures encountered during aggregation.
    """
    failures = precounted_failures or FailureCounts()
    all_mad: List[float] = []
    all_max: List[float] = []
    all_rng: List[float] = []
    mad_by_utt: List[np.ndarray] = []
    rng_by_utt: List[np.ndarray] = []
    loo_pool: Optional[List[List[float]]] = None
    flagged: List[FlaggedBoundary] = []
    n_utts = 0

    for entry in per_utt:
        # optional 4th element: original word indices of the retained boundaries. label-mismatch
        # filtering compacts the arrays, so flag with true positions, not compacted ones
        utt_id, labels, onsets_by_aligner = entry[0], entry[1], entry[2]
        orig_idx = entry[3] if len(entry) > 3 else None
        if len({len(o) for o in onsets_by_aligner}) != 1:
            failures.add("count_mismatch")
            continue
        reason = None
        for o in onsets_by_aligner:
            reason = onset_failure_reason(o)
            if reason:
                break
        if reason:
            failures.add(reason)
            continue
        n_utts += 1
        d = boundary_disagreement(onsets_by_aligner)
        all_mad.extend(d.mad_ms.tolist())
        all_max.extend(d.max_ms.tolist())
        all_rng.extend(d.range_ms.tolist())
        mad_by_utt.append(d.mad_ms)
        rng_by_utt.append(d.range_ms)
        if len(onsets_by_aligner) >= 3:
            loo = leave_one_out_deviations_ms(onsets_by_aligner)
            if loo_pool is None:
                loo_pool = [[] for _ in range(loo.shape[0])]
            for m in range(loo.shape[0]):
                loo_pool[m].extend(loo[m].tolist())
        for k in range(d.mad_ms.size):
            flagged.append(
                FlaggedBoundary(
                    utt_id=utt_id,
                    index=orig_idx[k] if orig_idx is not None else k,
                    label=labels[k] if k < len(labels) else str(k),
                    mad_ms=float(d.mad_ms[k]),
                    max_ms=float(d.max_ms[k]),
                    range_ms=float(d.range_ms[k]),
                    onsets_s=[float(o[k]) for o in onsets_by_aligner],
                )
            )

    flagged.sort(key=lambda f: f.max_ms, reverse=True)
    return AgreementResult(
        mad=deviation_stats(all_mad, tolerances),
        max_dev=deviation_stats(all_max, tolerances),
        span=deviation_stats(all_rng, tolerances),
        loo_per_source=(
            [deviation_stats(p, tolerances) for p in loo_pool] if loo_pool else None
        ),
        ci_ms={
            "mad_median": bootstrap_median_ci(mad_by_utt),
            "range_median": bootstrap_median_ci(rng_by_utt),
        },
        n_boundaries=len(all_mad),
        n_utts=n_utts,
        failures=failures,
        attempted_utts=attempted_utts if attempted_utts is not None else len(per_utt),
        flagged=flagged[:top_k_flag],
    )


# human-readable reporting
def format_stats_table(
    rows: Dict[str, DeviationStats], value_label: str = "deviation (ms)"
) -> str:
    """Format named deviation summaries as a fixed-width text table."""
    tols = sorted({t for s in rows.values() for t in s.frac_within}) or list(
        DEFAULT_TOL_MS
    )
    header = (
        f"{value_label[:24]:24s} {'n':>8s} {'median':>8s} {'P90':>8s} {'P95':>8s} {'mean':>8s}"
        + "".join(f" {'<=' + str(int(t)) + 'ms':>8s}" for t in tols)
    )
    lines = [header, "-" * len(header)]
    for name, s in rows.items():
        cells = "".join(
            f" {100 * s.frac_within.get(float(t), float('nan')):7.1f}%" for t in tols
        )
        lines.append(
            f"{name[:24]:24s} {s.n:8d} {s.median_ms:8.1f} {s.p90_ms:8.1f} {s.p95_ms:8.1f} "
            f"{s.mean_ms:8.1f}{cells}"
        )
    return "\n".join(lines)
