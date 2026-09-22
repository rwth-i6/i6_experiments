"""
The positional unigram matrix: the exact first E-step of a VQ-HMM under uniform
initialization.

Under a uniform table ``pi[c, k] = 1/K`` the emission product over a label
sequence of fixed length is ``K^-T`` for every sequence, so it cancels out of the
sequence posterior and leaves

    p(c_1^T | x_1^T) = p(c_1^T)        exactly, given |c| = T
    gamma_t(c)       = p(c_t = c)      the positional unigram of the label prior

The first E-step therefore uses no acoustic information at all, and the whole of
it can be precomputed once from label sequences - no search, no features. The
audio is touched exactly once afterwards, to build a binned codeword histogram,
and the first M-step is then a single GEMM (:func:`first_m_step`).

Everything here is plain numpy and knows nothing about sisyphus, RASR or HDF
files; ``..setup.positional_unigram`` wraps it in jobs and
``test_positional_unigram`` checks it against the closed form. The reference for
every choice below is ``docs/positional_unigram_init/segmented_base.md``, cited
by section.

**The position convention is load-bearing.** A ``gamma`` built under one
convention and consumed under another is silently wrong and produces no runtime
signal at all, which is why :data:`POSITION_CONVENTION` is written into the
artifact and checked on consumption.
"""

from __future__ import annotations

__all__ = [
    "POSITION_CONVENTION",
    "PositionalUnigram",
    "accumulate_binned",
    "accumulate_binned_keyed",
    "accumulate_tokens",
    "alignment_kernel",
    "binned_alignment_kernel",
    "build_gamma_tokens",
    "resolution_report",
    "suggest_num_bins",
    "suggested_sigma_bins",
    "backoff",
    "bakis_gamma",
    "band_of",
    "build_gamma",
    "effective_rank",
    "first_m_step",
    "gamma_diagnostics",
    "kernel_smooth_tau",
    "make_bands",
    "overlap_weights",
    "smoothing_matrix",
    "table_diagnostics",
]

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

#: Identifies the position mapping in §2.2. Bump it if ``overlap_weights``
#: changes what ``w(t, beta)`` means, so that an artifact built under the old
#: mapping is rejected instead of quietly reinterpreted.
POSITION_CONVENTION = "mass-overlap-v1"


# --- §2.2 position mapping ---------------------------------------------------


@lru_cache(maxsize=4096)
def overlap_weights(
    num_frames: int, num_bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The mass-preserving map from frame index to relative-position bin (§2.2).

    Frame ``t`` (1-based) covers ``I_t = [(t-1)/T, t/T)`` and bin ``beta`` covers
    ``J_b = [beta/B, (beta+1)/B)``; the weight is the overlap scaled so that each
    frame carries unit mass::

        w(t, beta) = T * |I_t ∩ J_b|

    Returned as the three parallel arrays of a sparse ``[T, B]`` matrix -
    ``(frame, bin, weight)``, one entry per non-empty intersection, so
    ``len(frame) <= T + B``. Both index arrays are 0-based.

    The point of the overlap map, rather than the obvious point map
    ``tau_t = (t-1)/(T-1)`` with ``floor(tau * B)``: the point map aliases badly
    once ``T < B``, and it double-counts the endpoints, which are exactly the
    positions carrying the most structure. The overlap map has two exact
    invariants instead, both asserted below and both tested:

    * ``sum_beta w(t, beta) = 1`` for every frame - unit mass per frame;
    * ``sum_t w(t, beta) = T/B`` for every bin - mass exactly balanced across
      bins, so the smoothing strength of §3 is uniform in ``tau`` and there is no
      boundary deficit.

    ``T = 1`` degenerates to ``w(1, beta) = 1/B`` for every bin, which is the
    correct "this frame carries no positional information" statement rather than
    a special case to guard.

    Computed on the common refinement of the two grids in **integer** arithmetic
    (denominator ``lcm(T, B)``), so a breakpoint shared by both grids is detected
    exactly rather than by float comparison, and no zero-width pieces appear.
    """
    if num_frames < 1 or num_bins < 1:
        raise ValueError(f"need T >= 1 and B >= 1, got T={num_frames}, B={num_bins}")

    denominator = math.lcm(num_frames, num_bins)
    step_frame = denominator // num_frames
    step_bin = denominator // num_bins
    edges = np.union1d(
        np.arange(0, denominator + 1, step_frame),
        np.arange(0, denominator + 1, step_bin),
    )
    lo, hi = edges[:-1], edges[1:]
    # Each piece lies inside exactly one frame and one bin, so its left endpoint
    # names both.
    frame_index = (lo // step_frame).astype(np.int64)
    bin_index = (lo // step_bin).astype(np.int64)
    weight = num_frames * (hi - lo) / denominator

    per_frame = np.bincount(frame_index, weights=weight, minlength=num_frames)
    per_bin = np.bincount(bin_index, weights=weight, minlength=num_bins)
    if not np.allclose(per_frame, 1.0, atol=1e-12):
        raise AssertionError(
            f"overlap weights do not carry unit mass per frame at T={num_frames}, "
            f"B={num_bins}: {per_frame.min()}..{per_frame.max()}"
        )
    if not np.allclose(per_bin, num_frames / num_bins, atol=1e-12):
        raise AssertionError(
            f"overlap weights are not balanced across bins at T={num_frames}, "
            f"B={num_bins}: {per_bin.min()}..{per_bin.max()} against "
            f"{num_frames / num_bins}"
        )
    return frame_index, bin_index, weight


# --- §2.3 length bands -------------------------------------------------------


def make_bands(
    lengths: Sequence[int],
    r_min: int = 200,
    ratio: float = 2.0,
    max_bands: int = 8,
) -> np.ndarray:
    """
    Half-open length bands ``[lo, hi)`` as ``int32[L + 1]`` edges (§2.3).

    Relative position fixes the *location* of a profile but not its *width*: the
    hypergeometric standard deviation is ``O(sqrt(T))`` absolute, hence
    ``O(1/sqrt(T))`` in ``tau``. So lengths still have to be conditioned on,
    coarsely. A band spanning a factor ``ratio`` in length spans ``sqrt(ratio)``
    in width, and ``ratio = 2`` gives 1.41x variation inside a band, which is
    acceptable.

    Geometric candidates first, then adjacent bands are merged until every band
    holds at least ``r_min`` sequences, then merged further to respect
    ``max_bands``. A narrow length distribution collapses to ``L = 1``, and the
    whole band machinery with it - harmlessly, since that is then the correct
    statement about the corpus.

    :param r_min: segments a band needs before it is trusted on its own. The
        backoff of §3.4 is written for ``R_l >= 200``, where ``kappa1 = 10``
        leaves the interpolation weight at or below 0.05.
    """
    lengths = np.asarray(list(lengths), dtype=np.int64)
    if lengths.size == 0:
        raise ValueError("no lengths given")
    if ratio <= 1.0:
        raise ValueError(f"ratio must exceed 1, got {ratio}")
    if max_bands < 1:
        raise ValueError(f"max_bands must be >= 1, got {max_bands}")

    low, high = int(lengths.min()), int(lengths.max())
    edges = [low]
    while edges[-1] <= high:
        nxt = max(int(math.ceil(edges[-1] * ratio)), edges[-1] + 1)
        edges.append(nxt)
    edges[-1] = high + 1
    if len(edges) < 2:
        edges = [low, high + 1]

    def counts_of(candidate):
        return np.array(
            [
                int(((lengths >= candidate[i]) & (lengths < candidate[i + 1])).sum())
                for i in range(len(candidate) - 1)
            ]
        )

    counts = counts_of(edges)
    # Merge the thinnest band into whichever neighbour is itself thinner: that
    # keeps a well-populated band from being diluted by a starved one.
    while len(counts) > 1 and counts.min() < r_min:
        i = int(counts.argmin())
        if i == 0:
            j = 1
        elif i == len(counts) - 1:
            j = i - 1
        else:
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        drop = max(i, j)          # the edge between i and j
        edges.pop(drop)
        counts = counts_of(edges)

    while len(counts) > max_bands:
        totals = counts[:-1] + counts[1:]
        edges.pop(int(totals.argmin()) + 1)
        counts = counts_of(edges)

    return np.asarray(edges, dtype=np.int32)


def band_of(lengths, band_edges: np.ndarray) -> np.ndarray:
    """
    Band index per length, **clamped** into range.

    Clamping rather than rejecting: the bands are built from one corpus (the
    features, whose lengths the consumer sees) and Gamma may be counted on
    another (the transcriptions), so a handful of sequences can fall outside.
    The callers report how many did - see ``clamped`` in
    :class:`PositionalUnigram` - which is the signal that the two corpora are
    not the same corpus after all.
    """
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.int64))
    band_edges = np.asarray(band_edges)
    index = np.searchsorted(band_edges[1:-1], lengths, side="right")
    return np.clip(index, 0, len(band_edges) - 2)


def outside_bands(lengths, band_edges: np.ndarray) -> int:
    """How many lengths :func:`band_of` had to clamp."""
    lengths = np.atleast_1d(np.asarray(lengths, dtype=np.int64))
    return int(((lengths < band_edges[0]) | (lengths >= band_edges[-1])).sum())


# --- §2.5 accumulation -------------------------------------------------------


def accumulate_binned(
    sequences: Iterable[np.ndarray],
    band_edges: np.ndarray,
    num_bins: int,
    num_symbols: int,
    progress: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    One pass over sequences of integer symbols, binned by relative position.

    Returns ``(counts[L, B, S], mass[L, B], sequences_per_band[L], clamped)``,
    accumulated in float64 (§2.5 stores float32 but does not accumulate in it).

    The same function serves both accumulations the method needs, because they
    are the same accumulation over different symbols:

    * ``Gamma[l, beta, c]`` over **label** sequences - the positional unigram;
    * ``H[l, beta, k]`` over **codeword** sequences - the binned codeword
      histogram of §4, the one place the audio is read.

    Sharing it is not just economy: the two have to use the identical position
    mapping and the identical bands or the GEMM in :func:`first_m_step` pairs up
    cells that mean different things, and nothing downstream would notice.

    Invariant on return, asserted by :func:`build_gamma`:
    ``sum_s counts[l, beta, s] == mass[l, beta]`` and
    ``mass[l, beta] == sum_{r in l} T_r / B`` for every ``beta``.
    """
    return accumulate_binned_keyed(
        ((symbols, len(np.asarray(symbols).reshape(-1))) for symbols in sequences),
        band_edges,
        num_bins,
        num_symbols,
        progress=progress,
    )


def accumulate_binned_keyed(
    items: Iterable[Tuple[np.ndarray, int]],
    band_edges: np.ndarray,
    num_bins: int,
    num_symbols: int,
    progress: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    :func:`accumulate_binned` with the banding quantity supplied per item.

    The two come apart in the ``L << T`` case: the frames are still binned by
    relative position, but the band is chosen by the **token** count, because
    that is what sets the profile width (addendum §9.2). The codeword histogram
    of §4 has to be banded the same way as gamma or the M-step pairs up cells
    that mean different things, so it needs this rather than the length of the
    sequence it is accumulating.
    """
    num_bands = len(band_edges) - 1
    counts = np.zeros((num_bands, num_bins, num_symbols), dtype=np.float64)
    mass = np.zeros((num_bands, num_bins), dtype=np.float64)
    per_band = np.zeros(num_bands, dtype=np.int64)
    clamped = 0

    for index, (symbols, band_key) in enumerate(items):
        symbols = np.asarray(symbols).reshape(-1)
        num_frames = len(symbols)
        if num_frames == 0:
            continue
        if symbols.min() < 0 or symbols.max() >= num_symbols:
            raise ValueError(
                f"symbol out of range: got {symbols.min()}..{symbols.max()}, "
                f"inventory is {num_symbols}"
            )
        band = int(band_of(band_key, band_edges)[0])
        clamped += outside_bands(band_key, band_edges)
        per_band[band] += 1

        frame_index, bin_index, weight = overlap_weights(num_frames, num_bins)
        # One bincount over the flattened (bin, symbol) index rather than
        # np.add.at, which is an order of magnitude slower and is called once
        # per sequence over a corpus of tens of thousands.
        flat = np.bincount(
            bin_index * num_symbols + symbols[frame_index],
            weights=weight,
            minlength=num_bins * num_symbols,
        )
        counts[band] += flat.reshape(num_bins, num_symbols)
        mass[band] += np.bincount(bin_index, weights=weight, minlength=num_bins)
        if progress is not None:
            progress.progress(index)

    return counts, mass, per_band, clamped


# --- §3.3 kernel smoothing ---------------------------------------------------


@lru_cache(maxsize=64)
def smoothing_matrix(num_bins: int, sigma: float) -> np.ndarray:
    """
    The ``[B, B]`` discrete Gaussian smoother along ``tau``, reflecting at both
    ends (§3.3).

    Built explicitly rather than convolved with padding, for one reason: mass
    preservation is then a property that can be *checked* rather than argued.
    The matrix is doubly stochastic by construction (each column distributes a
    normalized kernel; the reflection makes it symmetric), and both are asserted
    here, so ``sum_c Gamma[l, beta, c]`` is provably unchanged by the smoothing
    step.

    Reflecting, not zero-padded: zero padding leaks mass out at ``beta = 0`` and
    ``beta = B-1`` and so flattens the first and last labels' profiles, which are
    exactly the sharpest ones and the only ones carrying structure when the label
    prior mixes quickly.

    ``sigma <= 0`` returns the identity, i.e. no smoothing.
    """
    if sigma <= 0:
        return np.eye(num_bins)
    radius = max(1, int(math.ceil(4 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    taps /= taps.sum()

    matrix = np.zeros((num_bins, num_bins))
    for source in range(num_bins):
        for offset, tap in zip(range(-radius, radius + 1), taps):
            target = source + offset
            # Half-sample reflection, applied until the index lands in range:
            # -1 -> 0, -2 -> 1, B -> B-1, and so on.
            while target < 0 or target >= num_bins:
                if target < 0:
                    target = -1 - target
                if target >= num_bins:
                    target = 2 * num_bins - 1 - target
            matrix[target, source] += tap

    if not np.allclose(matrix.sum(axis=0), 1.0, atol=1e-12):
        raise AssertionError("smoothing matrix columns do not sum to 1")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1e-12):
        raise AssertionError("smoothing matrix rows do not sum to 1")
    return matrix


def kernel_smooth_tau(counts: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Smooth ``[L, B, S]`` counts along the bin axis (§3.3).

    Applied to the **unnormalized** counts and before the backoff of §3.4. The
    order matters: backing off first and smoothing afterwards mixes the prior
    into neighbouring bins twice.

    This is strictly better than backing off to the flat marginal, which is the
    obvious alternative: profiles are smooth in ``tau``, so averaging
    neighbouring bins buys variance reduction while *preserving* positional
    structure, where backoff removes structure by construction.
    """
    matrix = smoothing_matrix(counts.shape[1], float(sigma))
    smoothed = np.einsum("ij,ljs->lis", matrix, counts)

    before, after = counts.sum(axis=2), smoothed.sum(axis=2)
    if not np.allclose(before.sum(axis=1), after.sum(axis=1), atol=0.0, rtol=1e-12):
        raise AssertionError(
            f"kernel smoothing changed a band's total mass by up to "
            f"{np.abs(before.sum(1) - after.sum(1)).max():.3e}; the smoothing matrix "
            f"is not doubly stochastic"
        )
    # §3.3 asserts the stronger per-cell statement, and it holds - but only
    # because §2.2 balances the mass across bins, so the quantity being smoothed
    # is constant in beta and a doubly stochastic operator cannot move it. Check
    # it exactly where it applies, so this function stays usable (and honest) on
    # an unbalanced input.
    balanced = np.ptp(before, axis=1) <= 1e-9 * np.maximum(before.mean(axis=1), 1.0)
    # Relative, not absolute. The cell mass is sum_r T_r / B, which is ~5e4 on a
    # 3.5M-frame corpus and ~2e5 on a 13.8M-frame one; an absolute 1e-9 there is
    # a 1e-14 relative tolerance, i.e. below the round-off of the sum itself, and
    # the check starts failing on corpus size rather than on correctness. A
    # smoother that actually leaked would move a *fraction* of the mass, so a
    # relative bound is both the correct scale and far more sensitive.
    if balanced.any() and not np.allclose(
        before[balanced], after[balanced], atol=0.0, rtol=1e-12
    ):
        raise AssertionError(
            f"kernel smoothing changed the mass of a bin-balanced cell by up to "
            f"{np.abs(before[balanced] - after[balanced]).max():.3e} "
            f"(relative {np.abs(1 - after[balanced] / np.maximum(before[balanced], 1e-300)).max():.3e})"
        )
    return smoothed


# --- §3.4 hierarchical backoff ----------------------------------------------


def backoff(
    counts: np.ndarray,
    mass: np.ndarray,
    n_eff: np.ndarray,
    kappa0: float = 10.0,
    kappa1: float = 10.0,
) -> np.ndarray:
    """
    Three-level count-based interpolation, cell -> band -> global (§3.4).

        u(c)          = sum_{l,b} Gamma / sum_{l,b} N
        gbar_l(c)     = (R_l * band_estimate + kappa0 * u(c)) / (R_l + kappa0)
        gtilde_lb(c)  = (n_eff_lb * cell_estimate + kappa1 * gbar_l(c)) / (n_eff_lb + kappa1)

    Count-driven on purpose, and **no flat floor**. Structural zeros are real
    here - a label that cannot occur at a position has an exactly zero profile
    there, not a small one - and a floor ``(gamma + eps)/(1 + C*eps)`` destroys
    precisely the positional structure that makes uniform initialization work,
    hardest where the structure is strongest. The Dirichlet form leaves a cell
    with many segments and a zero count at zero, while pulling a cell with few
    segments towards the band prior.

    :param n_eff: the **effective** sample size per cell, which is the number of
        *segments* in the band, not the frame mass (§3.1). Frames inside a
        segment are strongly dependent - consecutive frames usually share a
        label - so the independent units are segments, and using frame counts
        here under-smooths by roughly ``T/B`` per cell.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        global_unigram = counts.sum(axis=(0, 1)) / max(mass.sum(), np.finfo(float).tiny)

        band_mass = mass.sum(axis=1, keepdims=True)                    # [L, 1]
        band_estimate = np.where(
            band_mass > 0, counts.sum(axis=1) / np.maximum(band_mass, 1e-300), 0.0
        )                                                              # [L, S]
        n_band = np.asarray(n_eff, dtype=np.float64).reshape(-1, 1)
        band_prior = (n_band * band_estimate + kappa0 * global_unigram) / (
            n_band + kappa0
        )                                                              # [L, S]

        cell_estimate = np.where(
            mass[..., None] > 0, counts / np.maximum(mass[..., None], 1e-300), 0.0
        )                                                              # [L, B, S]
        n_cell = n_band[..., None]                                     # [L, 1, 1]
        gamma = (n_cell * cell_estimate + kappa1 * band_prior[:, None, :]) / (
            n_cell + kappa1
        )

    total = gamma.sum(axis=-1, keepdims=True)
    # A band with no data at all leaves a zero row; the global unigram is the
    # only thing left to say about it, and saying nothing produces NaN.
    empty = total.squeeze(-1) <= 0
    if empty.any():
        gamma[empty] = global_unigram
        total = gamma.sum(axis=-1, keepdims=True)
    return gamma / np.maximum(total, np.finfo(float).tiny)


# --- the artifact ------------------------------------------------------------


@dataclass
class PositionalUnigram:
    """
    ``gamma[L, B, C]`` and everything needed to consume it safely.

    :param gamma: smoothed ``p(c_tau = c | band)``, rows summing to 1 over ``c``
    :param n_eff: effective sample size per cell, ``= R_l`` (§3.1)
    :param band_edges: half-open length bands, ``int32[L + 1]``
    :param mass: ``N[l, beta]``, the frame mass per cell, needed by the M-step
    :param counts: the raw ``Gamma`` before smoothing and backoff, kept because
        every diagnostic that asks "is this cell empty because it cannot happen
        or because nothing was seen" needs it
    :param meta: convention id, corpus hash, timestamp, hyperparameters
    """

    gamma: np.ndarray
    n_eff: np.ndarray
    band_edges: np.ndarray
    mass: np.ndarray
    counts: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_bins(self) -> int:
        return int(self.gamma.shape[1])

    @property
    def num_labels(self) -> int:
        return int(self.gamma.shape[2])

    def save(self, path: str) -> None:
        np.savez(
            path,
            gamma=self.gamma.astype(np.float32),
            n_eff=np.asarray(self.n_eff, dtype=np.int32),
            band_edges=np.asarray(self.band_edges, dtype=np.int32),
            mass=self.mass.astype(np.float64),
            counts=self.counts.astype(np.float64),
            meta=np.asarray(json.dumps(self.meta)),
        )

    @classmethod
    def load(cls, path: str) -> "PositionalUnigram":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("position_convention") != POSITION_CONVENTION:
                raise ValueError(
                    f"{path} was built under position convention "
                    f"{meta.get('position_convention')!r}, this code speaks "
                    f"{POSITION_CONVENTION!r}. A gamma consumed under the wrong "
                    f"convention is silently wrong; rebuild it."
                )
            return cls(
                gamma=data["gamma"].astype(np.float64),
                n_eff=data["n_eff"],
                band_edges=data["band_edges"],
                mass=data["mass"],
                counts=data["counts"],
                meta=meta,
            )


def build_gamma(
    sequences: Iterable[np.ndarray],
    num_labels: int,
    band_edges: np.ndarray,
    num_bins: int = 64,
    sigma_bins: float = 1.0,
    kappa0: float = 10.0,
    kappa1: float = 10.0,
    meta: Optional[Dict[str, Any]] = None,
    progress: Optional[Any] = None,
) -> PositionalUnigram:
    """
    §6's reference pipeline: accumulate, check the invariants, smooth,
    normalize, back off.

    Frame-level labels, i.e. one label per frame and ``|c| = T``. For token
    sequences with ``L << T`` see :func:`build_gamma_tokens`.
    """
    counts, mass, per_band, clamped = accumulate_binned(
        sequences, band_edges, num_bins, num_labels, progress=progress
    )
    return _finish_gamma(
        counts, mass, per_band, clamped, band_edges,
        num_bins=num_bins, num_labels=num_labels, sigma_bins=sigma_bins,
        kappa0=kappa0, kappa1=kappa1, meta=meta,
    )


def _finish_gamma(
    counts: np.ndarray,
    mass: np.ndarray,
    per_band: np.ndarray,
    clamped: int,
    band_edges: np.ndarray,
    *,
    num_bins: int,
    num_labels: int,
    sigma_bins: float,
    kappa0: float,
    kappa1: float,
    meta: Optional[Dict[str, Any]] = None,
) -> PositionalUnigram:
    """Everything after the corpus pass, shared by both accumulation modes."""
    if counts.sum() <= 0:
        raise RuntimeError("no label sequence contributed a single frame")

    # §2.5's invariant. Not a formality: it is the one check that catches a
    # position mapping and a band assignment disagreeing about a sequence.
    residual = np.abs(counts.sum(axis=2) - mass).max()
    if residual > 1e-6 * max(mass.max(), 1.0):
        raise AssertionError(f"counts and mass disagree by up to {residual:.3e}")
    populated = mass[per_band > 0]
    if populated.size:
        spread = np.ptp(populated, axis=1) / np.maximum(populated.mean(axis=1), 1e-12)
        if spread.max() > 1e-9:
            raise AssertionError(
                f"mass is not balanced across bins (relative spread up to "
                f"{spread.max():.3e}); the position mapping is not mass-preserving"
            )

    smoothed = kernel_smooth_tau(counts, sigma_bins)
    gamma = backoff(smoothed, mass, per_band, kappa0=kappa0, kappa1=kappa1)
    if not np.allclose(gamma.sum(axis=-1), 1.0, atol=1e-9):
        raise AssertionError("gamma rows do not sum to 1 after backoff")

    full_meta = {
        "position_convention": POSITION_CONVENTION,
        "num_bins": int(num_bins),
        "num_labels": int(num_labels),
        "sigma_bins": float(sigma_bins),
        "kappa0": float(kappa0),
        "kappa1": float(kappa1),
        "sequences_per_band": per_band.tolist(),
        "clamped_sequences": int(clamped),
    }
    full_meta.update(meta or {})
    return PositionalUnigram(
        gamma=gamma,
        n_eff=per_band.astype(np.int32),
        band_edges=np.asarray(band_edges, dtype=np.int32),
        mass=mass,
        counts=counts,
        meta=full_meta,
    )


# --- §4 consumption ----------------------------------------------------------


def first_m_step(
    gamma: np.ndarray,
    histogram: np.ndarray,
    mass: np.ndarray,
    table_floor: float = 0.0,
    unseen: str = "global",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    ``pi1[c, k]`` from gamma and the binned codeword histogram - §4, one GEMM.

        num[c, k] = sum_{l,b} gtilde_lb(c) * H[l, b, k]
        den[c]    = sum_{l,b} gtilde_lb(c) * N[l, b]
        pi1[c, k] = num[c, k] / den[c]

    which is ``(C x LB) @ (LB x K)``. At ``C = 40``, ``LB = 256``, ``K = 512``
    that is milliseconds; the spec's worst case of ``C = 1e4, K = 1e3`` is ~2.6
    GFLOP, seconds in BLAS. No forward-backward is run for iteration 1 at all.

    :param table_floor: added to the soft counts before normalizing, exactly as
        ``VectorQuantizedAccumulator`` and ``SupervisedVQTableJob`` do it. A zero
        entry scores ``+inf`` and is absorbing under a frozen codebook, and a
        codeword no label admits leaves a frame with no viable label at all.
    :param unseen: what to do with a label that gamma gives no mass anywhere -
        ``"global"`` for the global codeword histogram (the maximum-entropy
        choice consistent with having no evidence), ``"uniform"``, or
        ``"error"``. §3.5 rules out silently substituting a uniform row, because
        that makes the label indistinguishable from a genuinely uniform one and
        hides an inventory bug; here the substitution happens, but it is
        reported in the returned info and logged by the caller.
    """
    num_bands, num_bins, num_labels = gamma.shape
    if histogram.shape[:2] != (num_bands, num_bins) or mass.shape != (num_bands, num_bins):
        raise ValueError(
            f"shape mismatch: gamma {gamma.shape}, histogram {histogram.shape}, "
            f"mass {mass.shape}"
        )

    flat_gamma = gamma.reshape(num_bands * num_bins, num_labels).T      # [C, LB]
    flat_hist = histogram.reshape(num_bands * num_bins, -1)             # [LB, K]
    numerator = flat_gamma @ flat_hist                                  # [C, K]
    denominator = flat_gamma @ mass.reshape(-1, 1)                      # [C, 1]

    # sum_k H[l,b,k] == N[l,b], so the two agree by construction - which makes
    # this a free check that gamma and H were built over the same cells.
    drift = np.abs(numerator.sum(axis=1) - denominator.ravel()).max()
    scale = max(float(denominator.max()), 1.0)
    if drift > 1e-6 * scale:
        raise AssertionError(
            f"numerator and denominator disagree by up to {drift:.3e}; gamma and the "
            f"codeword histogram were not accumulated over the same cells"
        )

    seen = denominator.ravel() > 0
    floored = numerator + table_floor
    row_sums = floored.sum(axis=1, keepdims=True)
    table = np.where(row_sums > 0, floored / np.maximum(row_sums, 1e-300), 0.0)

    info: Dict[str, Any] = {
        "unseen_labels": np.flatnonzero(~seen).tolist(),
        "unseen_policy": unseen,
        "table_floor": float(table_floor),
        "total_mass": float(mass.sum()),
    }
    if not seen.all():
        if unseen == "error":
            raise ValueError(
                f"labels {info['unseen_labels']} have no mass in gamma; the M-step "
                f"denominator is zero for them"
            )
        total = flat_hist.sum(axis=0)
        if unseen == "global" and total.sum() > 0:
            replacement = total / total.sum()
        elif unseen in ("global", "uniform"):
            replacement = np.full(flat_hist.shape[1], 1.0 / flat_hist.shape[1])
        else:
            raise ValueError(f"unknown unseen policy {unseen!r}")
        table[~seen] = replacement

    return table, info


# --- §5 diagnostics ----------------------------------------------------------


def effective_rank(matrix: np.ndarray) -> float:
    """
    ``exp(H(sigma_hat))`` over the normalized singular spectrum.

    Upper-bounds the number of distinguishable rows of ``pi1``: the M-step is
    linear in gamma, so whatever ``gamma[l]`` cannot distinguish, ``pi1`` cannot
    either. A value near 1 says every row of the table will come out the same.
    """
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.float64), compute_uv=False)
    total = singular.sum()
    if total <= 0:
        return 0.0
    spectrum = singular / total
    nonzero = spectrum[spectrum > 0]
    return float(np.exp(-(nonzero * np.log(nonzero)).sum()))


def gamma_diagnostics(unigram: PositionalUnigram) -> Dict[str, Any]:
    """
    §5's table, computed before any audio is touched.

    The matrix carries no acoustic information, so these predict whether uniform
    initialization can work at all. The failure they look for is not a numerical
    accident: for a stationary ergodic label prior ``gamma_t(c) = pi_stat(c)``
    for every ``t``, every row of ``pi1`` equals the global codeword histogram,
    the emissions are label-independent again, and EM sits at an exact fixed
    point it never leaves. Left-to-right topologies escape only because they are
    maximally non-mixing with both endpoints pinned.
    """
    gamma = unigram.gamma
    num_bands, num_bins, num_labels = gamma.shape
    bands = []
    for band in range(num_bands):
        band_mean = gamma[band].mean(axis=0)
        deviation = np.abs(gamma[band] - band_mean[None, :]).sum(axis=1)
        bands.append(
            {
                "band": band,
                "length_range": [
                    int(unigram.band_edges[band]),
                    int(unigram.band_edges[band + 1]),
                ],
                "sequences": int(unigram.n_eff[band]),
                "frames": float(unigram.mass[band].sum() * num_bins),
                "effective_rank": effective_rank(gamma[band]),
                "non_stationarity_l1": deviation.tolist(),
                "non_stationarity_max": float(deviation.max()),
                "non_stationarity_mean": float(deviation.mean()),
                "non_stationarity_interior_max": float(
                    deviation[1:-1].max() if num_bins > 2 else deviation.max()
                ),
                "band_entropy": float(
                    -(band_mean[band_mean > 0] * np.log(band_mean[band_mean > 0])).sum()
                ),
            }
        )

    observed = unigram.counts.sum(axis=(0, 1))
    return {
        "num_bands": num_bands,
        "num_bins": num_bins,
        "num_labels": num_labels,
        "max_effective_rank": float(max(b["effective_rank"] for b in bands)),
        "labels_unseen": np.flatnonzero(observed <= 0).tolist(),
        "bands": bands,
        "meta": unigram.meta,
    }


def table_diagnostics(table: np.ndarray, threshold: float = 0.99) -> Dict[str, Any]:
    """
    Row separation of ``pi1`` (§5, last row of the table).

    ``max cos(pi1_c, pi1_c')`` over label pairs. Pairs above ``threshold`` are
    tied for good: with a frozen codebook the table is the whole model, so two
    labels scoring alike at initialization score alike after every subsequent
    E-step as well.
    """
    table = np.asarray(table, dtype=np.float64)
    norms = np.linalg.norm(table, axis=1, keepdims=True)
    normalized = table / np.maximum(norms, 1e-300)
    cosine = normalized @ normalized.T
    np.fill_diagonal(cosine, -np.inf)
    pairs = int(((cosine > threshold).sum()) // 2)
    total_pairs = table.shape[0] * (table.shape[0] - 1) // 2
    return {
        "max_row_cosine": float(cosine.max()),
        "mean_row_cosine": float(
            cosine[np.triu_indices(table.shape[0], k=1)].mean()
        ),
        "tied_pairs": pairs,
        "tied_pair_fraction": float(pairs / total_pairs) if total_pairs else 0.0,
        "tied_threshold": threshold,
        "row_entropy_mean": float(
            np.mean(
                [
                    -(row[row > 0] * np.log(row[row > 0])).sum()
                    for row in table
                ]
            )
        ),
        "effective_rank": effective_rank(table),
    }


# --- §8-§10: token sequences with L << T -------------------------------------


@lru_cache(maxsize=8)
def _log_factorial(limit: int) -> np.ndarray:
    """``log(n!)`` for ``n = 0..limit``, as a table.

    The kernel is a ratio of three binomials evaluated over a whole ``T x L``
    grid, so the alternative - ``math.comb`` in a double loop over big integers -
    costs a Python-level iteration per cell and overflows float conversion well
    before ``T`` gets interesting. One cumulative-sum table turns the whole grid
    into table lookups plus one ``exp``.
    """
    return np.concatenate([[0.0], np.cumsum(np.log(np.arange(1, limit + 1)))])


def alignment_kernel(
    num_frames: int, num_tokens: int, sub_states: int = 1
) -> np.ndarray:
    """
    ``A[t, i] = p(frame t belongs to token i | total length = T)`` - §9.1.

    The object §8.1 factors out of the token-level problem::

        gamma_t(c) = sum_i A[t, i] * delta(c_i = c)

    so the duration model enters only here, and the label sequence only through
    the one-hot incidence. ``A`` is ``[T, L]``, rows summing to 1.

    **Geometric durations need no parameter.** With ``d_i ~ Geom(1-a)`` iid,
    every composition of ``T`` into ``L`` positive parts has likelihood
    ``a^(T-L) (1-a)^L`` - the same for all of them - so conditioning on
    ``sum_i d_i = T`` leaves the uniform law over compositions and ``a``
    cancels. Counting them gives

        A[t, i] = C(t-1, i-1) C(T-t, L-i) / C(T-1, L-1)

    which is the hypergeometric of the base spec's §0 with ``S -> L``. In
    particular the self-loop probability the search is configured with does not
    enter, which is why this needs nothing fitted and nothing measured.

    :param sub_states: §10.1. Split each token into ``m`` sub-states with tied
        self-loops and the duration becomes a sum of ``m`` geometrics - negative
        binomial, unimodal with its mode away from 1, which is what a real phone
        duration looks like and why 3-state phone models are standard. It costs
        no new code and nothing to fit: build the geometric kernel at chain
        length ``m*L`` and sum each token's ``m`` columns back together. The
        profile width improves from ``1/(2 sqrt(L))`` to ``1/(2 sqrt(mL))``,
        which at ``m = 3`` is 1.7x and moves the §9.3 resolution floor from
        ``L ~ 16`` down to ``L ~ 6``.

        ``m`` is silently reduced where ``m*L > T`` - a chain of ``m*L``
        sub-states cannot be laid over ``T`` frames at all - so callers that
        care should check with :func:`effective_sub_states` first.
    """
    if num_tokens < 1 or num_frames < 1:
        raise ValueError(f"need T >= 1 and L >= 1, got T={num_frames}, L={num_tokens}")
    if num_tokens > num_frames:
        raise ValueError(
            f"{num_tokens} tokens cannot occupy {num_frames} frames; every token "
            f"needs at least one frame"
        )
    states = effective_sub_states(num_frames, num_tokens, sub_states)
    chain = num_tokens * states

    log_fact = _log_factorial(num_frames + 1)
    t = np.arange(1, num_frames + 1)[:, None]
    i = np.arange(1, chain + 1)[None, :]
    support = (i <= t) & (chain - i <= num_frames - t)

    # Indices are clipped before the lookup because numpy wraps negative ones,
    # and the mask is what makes the out-of-support cells irrelevant.
    def lf(values):
        return log_fact[np.clip(values, 0, num_frames + 1)]

    log_kernel = (
        lf(t - 1) - lf(i - 1) - lf(t - i)
        + lf(num_frames - t) - lf(chain - i) - lf(num_frames - t - chain + i)
        - (lf(num_frames - 1) - lf(chain - 1) - lf(num_frames - chain))
    )
    kernel = np.where(support, np.exp(log_kernel), 0.0)

    if states > 1:
        # A_token[t, i] = sum_j A_geom(T, m L)[t, m i + j]   (§10.1)
        kernel = kernel.reshape(num_frames, num_tokens, states).sum(axis=2)

    drift = np.abs(kernel.sum(axis=1) - 1.0).max()
    if drift > 1e-9:
        raise AssertionError(
            f"alignment kernel rows do not sum to 1 at T={num_frames}, "
            f"L={num_tokens}, m={states}: off by up to {drift:.3e}"
        )
    return kernel


def effective_sub_states(num_frames: int, num_tokens: int, sub_states: int) -> int:
    """The largest ``m' <= m`` whose chain of ``m'*L`` states fits in ``T`` frames."""
    return max(1, min(int(sub_states), num_frames // max(num_tokens, 1)))


@lru_cache(maxsize=256)
def binned_alignment_kernel(
    num_frames: int, num_tokens: int, num_bins: int, sub_states: int = 1
) -> np.ndarray:
    """
    ``P = W(T, B)^T A(T, L)``, the ``[B, L]`` kernel the corpus pass scatters
    (§9.4).

    Cached rather than ``A`` itself: it is ``B x L`` instead of ``T x L``,
    typically 5-20x smaller, and it is what the accumulation actually consumes.
    The cache pays off wherever ``(T, L)`` repeats; with frame-level ``T`` that
    is rarer than the spec's segment-level case, so the bound is kept modest and
    the build is cheap enough to repeat.

    Note the mass invariant survives the change of accumulation: since
    ``sum_i A[t, i] = 1``, ``sum_i P[beta, i] = sum_t w(t, beta) = T/B``, exactly
    as in §2.5.
    """
    frame_index, bin_index, weight = overlap_weights(num_frames, num_bins)
    kernel = alignment_kernel(num_frames, num_tokens, sub_states)
    binned = np.zeros((num_bins, num_tokens), dtype=np.float64)
    np.add.at(binned, bin_index, weight[:, None] * kernel[frame_index])

    expected = num_frames / num_bins
    drift = np.abs(binned.sum(axis=1) - expected).max()
    if drift > 1e-9 * max(expected, 1.0):
        raise AssertionError(
            f"binned kernel mass off by up to {drift:.3e} against T/B = {expected}"
        )
    return binned


def accumulate_tokens(
    items: Iterable[Tuple[np.ndarray, int]],
    band_edges: np.ndarray,
    num_bins: int,
    num_symbols: int,
    sub_states: int = 1,
    progress: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    The §9.4 corpus pass: scatter each token's kernel column into its label.

    ``items`` yields ``(tokens, num_frames)``. Returns
    ``(counts[L_bands, B, S], mass, sequences_per_band, clamped, reduced)``,
    where ``reduced`` counts the sequences whose ``sub_states`` had to be cut
    because the sub-state chain did not fit in the frames.

    **Banding is by token count, not by frame count.** The profile width is
    ``sd_tau ~ 1/(2 sqrt(L))``, governed by the chain length; §2.3 banded by
    ``T`` only because there ``S`` was proportional to ``T`` and the two
    coincided. Passing frame lengths here would band on the wrong variable and
    put segments of very different resolution in the same cell.

    The same label at several token positions contributes the **sum** of their
    kernel columns, which is what makes a frequent label's profile flat and a
    rare one's sharp. That is the mechanism, not an artifact.
    """
    num_bands = len(band_edges) - 1
    # [bands, symbols, bins] so the scatter writes contiguous rows; transposed
    # to the [bands, bins, symbols] the rest of the module speaks on return.
    counts = np.zeros((num_bands, num_symbols, num_bins), dtype=np.float64)
    mass = np.zeros((num_bands, num_bins), dtype=np.float64)
    per_band = np.zeros(num_bands, dtype=np.int64)
    clamped = reduced = 0

    for index, (tokens, num_frames) in enumerate(items):
        tokens = np.asarray(tokens).reshape(-1)
        num_tokens = len(tokens)
        if num_tokens == 0 or num_frames < num_tokens:
            continue
        if tokens.min() < 0 or tokens.max() >= num_symbols:
            raise ValueError(
                f"symbol out of range: got {tokens.min()}..{tokens.max()}, "
                f"inventory is {num_symbols}"
            )
        states = effective_sub_states(num_frames, num_tokens, sub_states)
        reduced += states != sub_states

        band = int(band_of(num_tokens, band_edges)[0])
        clamped += outside_bands(num_tokens, band_edges)
        per_band[band] += 1

        kernel = binned_alignment_kernel(
            int(num_frames), int(num_tokens), int(num_bins), int(states)
        )
        np.add.at(counts[band], tokens, kernel.T)
        mass[band] += kernel.sum(axis=1)
        if progress is not None:
            progress.progress(index)

    return counts.transpose(0, 2, 1), mass, per_band, clamped, reduced


def build_gamma_tokens(
    items: Iterable[Tuple[np.ndarray, int]],
    num_labels: int,
    band_edges: np.ndarray,
    num_bins: int = 64,
    sub_states: int = 3,
    sigma_bins: Optional[float] = None,
    kappa0: float = 10.0,
    kappa1: float = 10.0,
    meta: Optional[Dict[str, Any]] = None,
    progress: Optional[Any] = None,
) -> PositionalUnigram:
    """
    §6's pipeline for token sequences: kernel-scatter, check, smooth, back off.

    ``band_edges`` are over **token counts**. ``sigma_bins`` defaults to §9.5's
    rule rather than the base spec's 1.0 - see :func:`suggested_sigma_bins`.
    """
    counts, mass, per_band, clamped, reduced = accumulate_tokens(
        items, band_edges, num_bins, num_labels, sub_states, progress=progress
    )
    if sigma_bins is None:
        # §9.5's default needs the corpus, which the caller has and this
        # generator-consuming function no longer does once the pass is over.
        sigma_bins = 0.5
    extra = {
        "duration_model": "geometric",
        "sub_states": int(sub_states),
        "sub_states_reduced": int(reduced),
        "banded_by": "token_count",
    }
    extra.update(meta or {})
    return _finish_gamma(
        counts, mass, per_band, clamped, band_edges,
        num_bins=num_bins, num_labels=num_labels, sigma_bins=sigma_bins,
        kappa0=kappa0, kappa1=kappa1, meta=extra,
    )


def suggest_num_bins(token_counts) -> int:
    """``B ~ 4 sqrt(max L)``, clipped to ``[32, 128]`` (§9.2)."""
    longest = int(np.max(np.asarray(token_counts)))
    return int(np.clip(round(4 * math.sqrt(longest)), 32, 128))


def kernel_width_tau(token_counts, frame_counts, sub_states: int = 1) -> np.ndarray:
    """
    The exact mid-segment profile width in relative position, per segment.

    §9.2 quotes ``sd_tau ~ 1/(2 sqrt(mL))``, which is the ``mL << T`` limit. The
    exact width follows from ``i_t - 1 ~ Hypergeometric(N = T-1, K = mL-1,
    n = t-1)``: at ``t - 1 = (T-1)/2`` its variance is
    ``K (N-K) / (4 (N-1))``, so in token units

        sd(i) = sqrt((mL-1)(T-mL) / (4(T-2))) / m
        sd_tau = sd(i) / L

    and the asymptotic drops the ``(T-mL)/T`` factor. **That factor is not small
    here.** The unsegmented corpus runs at ``T/L ~ 3.9``, so with ``m = 3`` the
    sub-state chain covers 77% of the frames, the correction is ``sqrt(0.23) =
    0.48``, and the real profiles are about twice as sharp as the asymptotic
    says. Using the asymptotic would understate the available resolution by 2x
    and mis-set the smoothing that §9.5 derives from it.
    """
    tokens = np.asarray(token_counts, dtype=np.float64)
    frames = np.asarray(frame_counts, dtype=np.float64)
    chain = np.minimum(sub_states * tokens, frames)
    variance = np.maximum(
        (chain - 1) * (frames - chain) / (4 * np.maximum(frames - 2, 1.0)), 0.0
    )
    return np.sqrt(variance) / np.maximum(chain, 1.0)


def suggested_sigma_bins(
    num_bins: int, token_counts, frame_counts, sub_states: int = 1
) -> float:
    """
    §9.5's rule: the kernel already smooths, so §3.3 should give way.

    ``A`` smears each token over ``B * sd_tau`` bins, which is variance
    reduction of exactly the kind §3.3 provides - applying both double-smooths
    and throws away resolution that was expensive to get. So ``sigma_h = 0``
    once the kernel's own width covers two bins, and 0.5 otherwise, against the
    base spec's 1.0.

    Evaluated on the exact width (:func:`kernel_width_tau`), not the asymptotic,
    and on the *narrowest* segment in the corpus, since that is the one §3.3
    would over-smooth.
    """
    width = num_bins * kernel_width_tau(token_counts, frame_counts, sub_states)
    return 0.0 if float(np.min(width)) >= 2.0 else 0.5


def resolution_report(
    token_counts, frame_counts, sub_states: int = 1
) -> Dict[str, Any]:
    """
    §9.3's screening criterion, computable from the length distribution alone.

    The profile width in relative position decides whether the middle tokens are
    distinguishable at all; below the floor ``G`` collapses to near rank 1 for
    everything but the boundary tokens, which is the ``L << T`` analogue of the
    §5 degeneracy. It is a property of the *duration model*, not of the data, so
    it is worth knowing before any features are read - which is the whole point
    of computing it here.

    Reported both ways: ``sd_tau`` exact, and ``sd_tau_asymptotic`` as the spec's
    ``1/(2 sqrt(mL))`` table is written, so the gap between them is visible
    rather than silently resolved in one direction.
    """
    tokens = np.asarray(token_counts, dtype=np.float64)
    frames = np.asarray(frame_counts, dtype=np.float64)
    width = kernel_width_tau(tokens, frames, sub_states)
    asymptotic = 1.0 / (2 * np.sqrt(np.maximum(sub_states * tokens, 1.0)))
    # The spec's table is thresholds on sd_tau: 0.25 flat, 0.125 marginal,
    # 0.0625 usable, 0.031 good.
    return {
        "sub_states": int(sub_states),
        "tokens_min": int(tokens.min()),
        "tokens_max": int(tokens.max()),
        "tokens_mean": float(tokens.mean()),
        "frames_per_token_mean": float((frames / np.maximum(tokens, 1)).mean()),
        "chain_fill_mean": float(
            (np.minimum(sub_states * tokens, frames) / np.maximum(frames, 1)).mean()
        ),
        "sd_tau_mean": float(width.mean()),
        "sd_tau_median": float(np.median(width)),
        "sd_tau_worst": float(width.max()),
        "sd_tau_asymptotic_mean": float(asymptotic.mean()),
        # Only over sequences that have any spread left. A sequence with
        # T == m*L has every token pinned to exactly m frames - the kernel is a
        # hard indicator, width exactly 0 - and dividing by it turns the ratio
        # into a number about the floor rather than about the corpus. Those are
        # counted separately, because that is §10.3's deterministic limit
        # arriving by itself rather than by choice.
        "sharpening_over_asymptotic": (
            float(np.median(asymptotic[width > 0] / width[width > 0]))
            if np.any(width > 0) else float("inf")
        ),
        "fraction_deterministic": float((width <= 0).mean()),
        "fraction_flat": float((width > 0.125).mean()),
        "fraction_marginal": float(((width <= 0.125) & (width > 0.0625)).mean()),
        "fraction_usable": float(((width <= 0.0625) & (width > 0.031)).mean()),
        "fraction_good": float((width <= 0.031).mean()),
        "verdict": (
            "good" if float(np.median(width)) <= 0.031 else
            "usable" if float(np.median(width)) <= 0.0625 else
            "marginal" if float(np.median(width)) <= 0.125 else "flat"
        ),
    }


# --- the closed form, for tests ---------------------------------------------


def bakis_gamma(num_frames: int, num_states: int) -> np.ndarray:
    """
    ``gamma[t, s]`` for a Bakis chain with pinned endpoints, in closed form.

        gamma_t(s) = C(t-1, s-1) * C(T-t, S-s) / C(T-1, S-1)

    Hypergeometric, and **independent of the self-loop probability** - with both
    endpoints pinned and equal self-loops, every admissible path is equiprobable,
    so the marginal is pure counting. This is the case §0 says not to build a
    matrix for; it exists here as ground truth for the tests, which is the only
    thing that validates the length handling and the position mapping at once.

    Returned 0-based in both axes: ``[T, S]``.
    """
    if not 1 <= num_states <= num_frames:
        raise ValueError(f"need 1 <= S <= T, got S={num_states}, T={num_frames}")
    total = math.comb(num_frames - 1, num_states - 1)
    out = np.zeros((num_frames, num_states))
    for t in range(1, num_frames + 1):
        for s in range(1, num_states + 1):
            if s <= t and (num_frames - t) >= (num_states - s):
                out[t - 1, s - 1] = (
                    math.comb(t - 1, s - 1) * math.comb(num_frames - t, num_states - s)
                ) / total
    return out
