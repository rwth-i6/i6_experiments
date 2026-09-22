"""
Exponential moving average between clustering steps.

The full-epoch update accumulates over the whole corpus and re-estimates once.
Splitting an epoch into B batches and re-estimating after each one converges
faster per unit of compute, but only if a batch's estimate is damped against
what came before - a single batch is a noisy M-step, and taking it whole would
throw the model away every 1/B of an epoch.

There are two places to put that damping, and they are *different algorithms*,
not two spellings of one:

``"parameters"``
    Re-estimate from the batch alone, then interpolate the resulting model with
    the previous one::

        theta_t = a * theta_{t-1} + (1 - a) * M(N_t)

    Each batch gets weight ``1 - a`` regardless of how much evidence it
    carried, so a label seen three times moves as far as one seen three
    thousand times. Needs nothing but the two models, which is what makes it
    the cheap option: no state travels between steps.

``"statistics"``
    Damp the *sufficient statistics* and re-estimate from those::

        S_t = a * S_{t-1} + (1 - a) * N_t,   theta_t = M(S_t)

    This is online/incremental EM in its standard form (Neal & Hinton 1998;
    Cappe & Moulines 2009). A batch now moves the model in proportion to the
    evidence it actually contributed, per label rather than per model, and a
    label absent from a batch simply keeps its decayed statistic instead of
    needing the ``keep_previous_where_dead`` rule to rescue it. The cost is that
    ``S`` has to travel from one step to the next: 160 kB for the VQ table,
    but ~1 GB for a full-covariance mixture, which is the practical reason
    ``"parameters"`` remains worth having.

For a VQ run the two read as "EMA the counts" (statistics) versus "EMA the
normalized table" (parameters).

**The decay is expressed with the operations the accumulators already have.**
``S <- a*S + (1-a)*N`` is ``prior.scale(a).merge(batch.scale(1-a))``: merge
already existed, because reducing chunks is the same addition, and it is exact
for the same reason - the statistics are sums over frames taken under a model
that is fixed for the whole step. So chunking a batch and damping across
batches compose without either knowing about the other.

**Choosing ``alpha``.** The effective averaging window is
``batch_size / (1 - alpha)`` sequences. It is a property of the pair, so
shrinking batches to keep barriers cheap while holding ``alpha`` fixed is a
different experiment, not a faster one: at half the batch size, ``1 - alpha``
has to halve too for the runs to stay comparable. :func:`effective_window`
spells this out.
"""

from __future__ import annotations

__all__ = [
    "EMA_MODES",
    "blend_parameters",
    "ema_statistics",
    "effective_window",
    "load_state",
    "save_state",
]

import pickle
from typing import Optional

import numpy as np

from .interfaces import Accumulator, ScoreModel

#: ``"parameters"`` interpolates the models, ``"statistics"`` the sufficient
#: statistics. See the module docstring - they are different algorithms.
EMA_MODES = ("parameters", "statistics")


def _check_alpha(alpha: float) -> float:
    if not 0.0 <= alpha < 1.0:
        # 1.0 is excluded rather than clamped: it never updates the model at
        # all, which is a silently wasted run rather than a slow one.
        raise ValueError(f"ema alpha must be in [0, 1), got {alpha}")
    return float(alpha)


def effective_window(batch_size: float, alpha: float) -> float:
    """
    Sequences the EMA effectively averages over, ``batch_size / (1 - alpha)``.

    Reported by the epoch job so a run records what it actually averaged rather
    than leaving it to be reconstructed from two knobs in a config.
    """
    return batch_size / (1.0 - _check_alpha(alpha))


def blend_parameters(previous: ScoreModel, updated: ScoreModel, alpha: float) -> ScoreModel:
    """
    ``alpha * previous + (1 - alpha) * updated``, artifact by artifact.

    Generic over model classes: it goes through ``artifacts()`` and
    ``from_artifacts()``, the same pair ``save``/``load`` use, so a model class
    that can be written to disk can be interpolated without naming its
    parameters here. Frozen artifacts (a VQ codebook) are interpolated too and
    are unaffected by it, both sides being the same array.

    Convexity is what keeps the result a valid model: a blend of two
    row-normalized tables is row-normalized, and a blend of two positive
    definite covariances is positive definite. The constructor still checks -
    ``from_artifacts`` runs the model's own validation.
    """
    alpha = _check_alpha(alpha)
    if type(previous) is not type(updated):
        raise TypeError(
            f"cannot interpolate a {type(previous).__name__} with a "
            f"{type(updated).__name__}"
        )
    old, new = previous.artifacts(), updated.artifacts()
    if set(old) != set(new):
        raise ValueError(
            f"artifact sets differ: {sorted(old)} vs {sorted(new)}"
        )

    blended = {}
    for name in new:
        a, b = np.asarray(old[name]), np.asarray(new[name])
        if a.shape != b.shape:
            raise ValueError(
                f"artifact {name!r} changed shape: {a.shape} -> {b.shape}"
            )
        if not (np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating)):
            # An index array (a density-to-label map, say) has no meaningful
            # midpoint, and averaging one would produce a plausible-looking
            # model that is wrong everywhere it is read.
            raise TypeError(
                f"artifact {name!r} is not floating point ({a.dtype}, {b.dtype}); "
                f"interpolating it is not meaningful. Use mode='statistics' for a "
                f"model whose parameters are not all continuous."
            )
        blended[name] = alpha * a + (1.0 - alpha) * b
    return type(updated).from_artifacts(blended, updated.meta())


def ema_statistics(
    prior: Optional[Accumulator], batch: Accumulator, alpha: float
) -> Accumulator:
    """
    ``alpha * prior + (1 - alpha) * batch``, on the sufficient statistics.

    Both accumulators are consumed: ``scale`` and ``merge`` work in place, and
    every caller here has just built them from a state dict.

    :param prior: the previous step's running statistic, or None at the first
        step of a run - there is nothing to damp against yet, so the batch is
        taken whole. That is the usual initialization for a stochastic
        approximation and it means a run's first step is an ordinary (small)
        epoch.
    """
    alpha = _check_alpha(alpha)
    if not hasattr(batch, "scale"):
        raise TypeError(
            f"{type(batch).__name__} has no scale(), so its statistics cannot be "
            f"damped. Accumulators built on a Welford-style running estimate "
            f"rather than on raw sums need their own rule; use mode='parameters' "
            f"with this flavor."
        )
    if prior is None:
        return batch
    return prior.scale(alpha).merge(batch.scale(1.0 - alpha))


def save_state(accumulator: Accumulator, path: str) -> None:
    """
    Persist a running statistic for the next step.

    Pickle for the same reason ``save_chunk`` uses it - the state is a dict
    mixing arrays with scalars and, for the mixture accumulator, a nested state
    dict of its own. Unlike a chunk file this one is a job *output*: it has to
    outlive the job's work directory, because the next step reads it.
    """
    with open(path, "wb") as fp:
        pickle.dump(accumulator.state_dict(), fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_state(path: str) -> dict:
    with open(path, "rb") as fp:
        return pickle.load(fp)
