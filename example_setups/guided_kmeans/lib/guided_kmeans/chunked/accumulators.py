"""
Accumulators: the updating mechanism, in mergeable form.

Every accumulator here holds a *sufficient statistic* of the frames it has
seen, and ``merge`` is associative. That is what allows a clustering epoch to
be split across independent chunk tasks and reduced afterwards, and it is why
``num_chunks`` can be excluded from the job hash: the partition provably
cannot change the result (up to float summation order).
"""

from __future__ import annotations

__all__ = [
    "FixedCovarianceAccumulator",
    "MeanAccumulator",
    "GaussianAccumulator",
    "LabelCovarianceAccumulator",
    "MixtureGaussianAccumulator",
    "NullAccumulator",
    "SoftGaussianAccumulator",
    "VectorQuantizedAccumulator",
    "alive_mask",
    "if_alive_else",
    "common_diagonalizer",
    "keep_previous_where_dead",
    "semi_tied_objective",
    "semi_tied_transform",
    "shrink_covariances",
]

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from ..pca import PCAUpdater
from .interfaces import (
    Posteriors,
    ScoreModel,
    as_dense_responsibilities,
    as_hard_labels,
)
from .models import (
    EuclideanModel,
    GaussianModel,
    LabelCovarianceMixtureModel,
    MixtureModelBase,
)


@dataclass
class _PreviousArrays:
    """
    Minimal stand-in exposing the ``.means``/``.covs`` attributes
    :meth:`PCAUpdater.get_model` reads off the previous model, so this does not
    have to reach into a GaussianModel's wrapped GaussianModelNumpy.
    """

    means: np.ndarray
    covs: np.ndarray


def alive_mask(counts: np.ndarray, min_mass: float) -> np.ndarray:
    """
    Which entries hold enough evidence to be re-estimated from.

    The ``> 0`` term is not redundant: it keeps ``min_mass=0.0`` meaning
    "whatever got any mass at all" rather than "everything, including entries
    that would divide by zero".
    """
    return (counts > 0) & (counts >= min_mass)


def if_alive_else(counts: np.ndarray, min_mass: float, default=1.0) -> np.ndarray:
    """
    ``counts`` with the dead entries replaced, so a division by them stays
    quiet. The dead rows are overwritten wholesale afterwards, which is why the
    substituted value is arbitrary - pair this with :func:`alive_mask` to know
    which rows those are.
    """
    return np.where(alive_mask(counts, min_mass), counts, default)


def keep_previous_where_dead(
    updated: Mapping[str, np.ndarray],
    previous: ScoreModel,
    dead: Union[np.ndarray, Mapping[str, np.ndarray]],
) -> Dict[str, np.ndarray]:
    """
    Restore the previous epoch's parameters for clusters that received no data.

    Without this they would come out as zeros and stop being reachable
    candidates in the next recognition pass - the same rule the single-process
    callback applies to centroids, here generalized over a model's artifacts so
    it holds for any parameter set.

    Takes and returns the *artifact mapping* rather than a model, because a
    model may not survive construction from pre-fallback arrays: GaussianModel
    inverts its covariances eagerly, and a dead cluster's covariance is all
    zeros until this rule has replaced it. Callers therefore fix the arrays up
    first and construct once, from values that are already correct.

    Relies on every artifact being indexed by cluster on its first axis, which
    is asserted rather than assumed: a model carrying a genuinely global
    parameter needs this rule extended, and should fail loudly here instead of
    having that parameter silently overwritten.
    """
    new_arrays = dict(updated)
    # A single mask covers every artifact; a dict gives one per artifact, which is
    # what pooled covariances need - a density can be too starved for its own mean
    # while the label it belongs to is perfectly well estimated.
    masks = dead if isinstance(dead, Mapping) else {name: dead for name in new_arrays}
    if not any(np.asarray(mask).any() for mask in masks.values()):
        return new_arrays

    old_arrays = previous.artifacts()
    if set(new_arrays) != set(old_arrays):
        raise ValueError(
            f"cannot carry parameters over from {type(previous).__name__}: its "
            f"artifacts {sorted(old_arrays)} do not match {sorted(new_arrays)}"
        )

    merged = {}
    for name, array in new_arrays.items():
        array = np.asarray(array)
        old = np.asarray(old_arrays[name])
        dead = np.asarray(masks[name])
        if array.shape[0] != len(dead) or old.shape[0] != len(dead):
            raise ValueError(
                f"artifact {name!r} is not indexed by cluster on its first axis "
                f"(got {array.shape} / {old.shape} for {len(dead)} clusters); "
                f"extend keep_previous_where_dead() for non-per-cluster parameters"
            )
        array = np.array(array, copy=True)
        array[dead] = old[dead]
        merged[name] = array

    return merged


def shrink_covariances(
    covs: np.ndarray,
    counts: np.ndarray,
    shrinkage: float,
    alive: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Pull each cluster's covariance towards the pooled within-cluster covariance.

        Sigma_k <- (1 - s) * Sigma_k + s * Sigma_pooled
        Sigma_pooled = sum_k n_k Sigma_k / sum_k n_k     (over live clusters)

    The standard remedy for estimating a ``[D, D]`` covariance from too few
    frames, and the reason it is needed here is arithmetic rather than
    stylistic: a full covariance at D=512 has 512*513/2 ~ 131k free parameters,
    while ls-100 gives about 35k frames per cluster at K=512. Below one frame
    per parameter the sample covariance is singular by construction, and
    ``GaussianModelNumpy`` inverts eagerly with ``np.linalg.inv`` and casts the
    result to float32 - so the failure is not a clean exception but a silently
    meaningless inverse, or ``-inf`` normalization constants, well away from
    the cluster that caused it.

    ``s`` interpolates between the two models the choice is really between:
    ``s=0`` is a free covariance per cluster, ``s=1`` is one covariance tied
    across all of them. That makes it the right axis to sweep for the question
    "does a per-cluster covariance buy anything at this sample size" - if the
    best ``s`` is at or near 1, the answer is no, and the sweep says so with a
    curve rather than with a crash.

    The pooled covariance is computed from this epoch's own statistics rather
    than taken as an input, so nothing has to be threaded through the pipeline
    and the tie is always to the matching set of clusters. Mass-weighted, so a
    starving cluster is pulled towards where the evidence actually is.

    :param counts: per-cluster frame mass, used both to weight the pool and,
        with ``alive``, to decide which clusters contribute to it
    :param alive: which clusters were re-estimated at all; dead ones neither
        contribute to the pool nor get shrunk, since their values are about to
        be replaced by the previous model's.
    """
    if not shrinkage:
        return covs
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError(f"shrinkage must be in [0, 1], got {shrinkage}")

    covs = np.asarray(covs, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    alive = np.ones(len(covs), dtype=bool) if alive is None else np.asarray(alive, dtype=bool)
    if not alive.any():
        return covs

    mass = counts[alive]
    total = mass.sum()
    if total <= 0:
        return covs
    pooled = np.tensordot(mass, covs[alive], axes=(0, 0)) / total

    shrunk = np.array(covs, copy=True)
    shrunk[alive] = (1.0 - shrinkage) * covs[alive] + shrinkage * pooled[np.newaxis, :, :]
    return shrunk


class MeanAccumulator:
    """
    Per-cluster frame counts and feature sums; finalizes to centroids.

    Equivalent to what ``RunningAverageUpdater`` accumulates in the
    single-process callback, but keeping raw sums instead of a running mean -
    same result, one division instead of one per sequence.

    :param num_clusters: size of the label inventory
    :param dim: feature dimension; inferred from the first observation if None
    """

    def __init__(self, num_clusters: int, dim: Optional[int] = None, **runtime_args):
        self.num_clusters = num_clusters
        self.counts = np.zeros(num_clusters, dtype=np.float64)
        self.sums = None if dim is None else np.zeros((num_clusters, dim), dtype=np.float64)

    def bind_model(self, model: ScoreModel) -> None:
        """No-op: counts and feature sums need only the frames and their labels."""

    def _ensure(self, dim: int) -> None:
        if self.sums is None:
            self.sums = np.zeros((self.num_clusters, dim), dtype=np.float64)
        elif self.sums.shape[1] != dim:
            raise ValueError(f"feature dim changed: {self.sums.shape[1]} -> {dim}")

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        features = np.asarray(features, dtype=np.float64)
        self._ensure(features.shape[1])
        # One conversion covers hard labels, n-best and dense FB gammas alike.
        # For hard labels it yields exactly the one-hot matrix the
        # single-process callback contracts (`idx_matrix.T @ hidden_states`),
        # down to the summation order, keeping the two pipelines bit-comparable;
        # soft posteriors put non-binary weights in the same matrix.
        responsibilities = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(responsibilities) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs "
                f"{len(responsibilities)} posteriors"
            )
        self.counts += responsibilities.sum(0)
        self.sums += responsibilities.T @ features

    def merge(self, other: "MeanAccumulator") -> "MeanAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if other.sums is None:
            return self
        self._ensure(other.sums.shape[1])
        self.counts += other.counts
        self.sums += other.sums
        return self

    def scale(self, factor: float) -> "MeanAccumulator":
        """
        Multiply the accumulated statistics by ``factor``, in place.

        The decay half of the statistics-level EMA ``S <- a*S + (1-a)*N``;
        :meth:`merge` is the addition half, so that recursion needs no
        arithmetic of its own (see :mod:`.ema`). Every statistic here is a sum
        over frames, so scaling it is the stochastic-approximation step of
        online EM: the result is still a (fractional) count and ``finalize``
        normalizes it exactly as it does a plain epoch's.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        self.counts = self.counts * factor
        if self.sums is not None:
            self.sums = self.sums * factor
        return self

    def finalize(self, previous: ScoreModel) -> EuclideanModel:
        if self.sums is None:
            raise RuntimeError("nothing accumulated; cannot finalize")
        centroids = np.divide(
            self.sums,
            self.counts[:, np.newaxis],
            out=np.zeros_like(self.sums),
            where=self.counts[:, np.newaxis] > 0,
        )
        merged = keep_previous_where_dead(
            {"centroids": centroids}, previous, self.counts == 0
        )
        return EuclideanModel(merged["centroids"])

    def state_dict(self) -> dict:
        return {"counts": self.counts, "sums": self.sums, "num_clusters": self.num_clusters}

    def load_state_dict(self, state: dict) -> "MeanAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self.counts = np.asarray(state["counts"], dtype=np.float64)
        self.sums = None if state["sums"] is None else np.asarray(state["sums"], dtype=np.float64)
        return self


class FixedCovarianceAccumulator:
    """
    Per-cluster means under a covariance that is never re-estimated.

    Lloyd's algorithm in the metric a fixed covariance induces: assignment uses
    the full Mahalanobis distance, the update moves only the means, and the
    covariance the next epoch scores with is the one this epoch scored with.
    With a single shared covariance duplicated across every cluster - what
    :class:`...setup.chunked_clustering.GlobalCovarianceJob` plus
    :class:`...setup.chunked_clustering.DuplicateCovsJob` produce - that is
    exactly k-means in the globally whitened space, which is the right default
    for encoder features whose dimensions carry arbitrary relative scaling.

    Why this rather than :class:`MeanAccumulator`: that one finalizes to an
    :class:`.models.EuclideanModel`, which drops the covariance and so cannot
    be the model an epoch of this run starts from. Every epoch's model spec has
    to have the same shape for a continued run to reuse jobs (see
    :func:`...setup.chunked_clustering.chunked_clustering`), so a run that
    scores with a covariance has to keep carrying one, re-estimated or not.

    Why not :class:`GaussianAccumulator` or :class:`SoftGaussianAccumulator`:
    both estimate the covariance too, which is precisely what a
    partition-finding pass should not be doing at this feature dimension. A
    full ``[D, D]`` covariance per cluster needs far more evidence than a mean
    - at D=512 and a few hundred clusters over a 100h corpus there are fewer
    frames per cluster than a covariance has free parameters - so estimating
    one here produces a rank-poor matrix that degrades the very assignments the
    pass exists to compute. Separating the two is what lets the covariance be
    decided later, on the converged partition, and by its own evidence.

    The state is O(K x D) rather than O(K x D^2), which is the same saving
    stated the other way round: at K=512, D=512 this holds ~2 MB where a
    full-covariance accumulator holds ~1 GB per chunk task.

    :param num_clusters: size of the label inventory
    :param dim: feature dimension; inferred from the first observation if None
    :param min_mass: mass a cluster needs before its mean is re-estimated;
        below it the previous model's mean is kept. The default of 0.0 means
        "whatever got any mass at all", which is the natural rule for the hard
        assignments :class:`.recognizers.ArgmaxRecognizer` produces - a frame
        is wholly in a cluster or not in it. Pair with a soft recognizer and
        the floor matters for the reasons :class:`SoftGaussianAccumulator`
        documents.
    """

    def __init__(
        self,
        num_clusters: int,
        dim: Optional[int] = None,
        min_mass: float = 0.0,
        **runtime_args,
    ):
        self.num_clusters = num_clusters
        self.min_mass = min_mass
        self.counts = np.zeros(num_clusters, dtype=np.float64)
        self.sums = None if dim is None else np.zeros((num_clusters, dim), dtype=np.float64)

    def bind_model(self, model: ScoreModel) -> None:
        """No-op: counts and feature sums need only the frames and their labels."""

    def _ensure(self, dim: int) -> None:
        if self.sums is None:
            self.sums = np.zeros((self.num_clusters, dim), dtype=np.float64)
        elif self.sums.shape[1] != dim:
            raise ValueError(f"feature dim changed: {self.sums.shape[1]} -> {dim}")

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
        self._ensure(features.shape[1])
        responsibilities = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(responsibilities) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs "
                f"{len(responsibilities)} posteriors"
            )
        self.counts += responsibilities.sum(0)
        self.sums += responsibilities.T @ features

    def merge(self, other: "FixedCovarianceAccumulator") -> "FixedCovarianceAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if other.sums is None:
            return self
        self._ensure(other.sums.shape[1])
        self.counts += other.counts
        self.sums += other.sums
        return self

    def scale(self, factor: float) -> "FixedCovarianceAccumulator":
        """
        Multiply the accumulated statistics by ``factor``, in place.

        The decay half of the statistics-level EMA ``S <- a*S + (1-a)*N``;
        :meth:`merge` is the addition half, so that recursion needs no
        arithmetic of its own (see :mod:`.ema`). Every statistic here is a sum
        over frames, so scaling it is the stochastic-approximation step of
        online EM: the result is still a (fractional) count and ``finalize``
        normalizes it exactly as it does a plain epoch's.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        self.counts = self.counts * factor
        if self.sums is not None:
            self.sums = self.sums * factor
        return self

    def finalize(self, previous: ScoreModel) -> ScoreModel:
        if self.sums is None:
            raise RuntimeError("nothing accumulated; cannot finalize")
        arrays = previous.artifacts()
        missing = {"centroids", "covs"} - set(arrays)
        if missing:
            raise TypeError(
                f"FixedCovarianceAccumulator needs a model carrying {sorted(missing)} "
                f"to carry the covariance over from, got {type(previous).__name__} "
                f"with {sorted(arrays)}. Use MeanAccumulator for a model with no "
                f"covariance at all."
            )

        alive = alive_mask(self.counts, self.min_mass)
        n = if_alive_else(self.counts, self.min_mass)
        centroids = self.sums / n[:, np.newaxis]

        # The covariance goes through the dead-cluster rule with the means even
        # though it is carried over wholesale, so the artifact sets match and
        # the shared rule stays the single place a fallback happens. It is a
        # no-op on `covs` by construction - the values are already the previous
        # model's - and that is the point: nothing here decides how a frozen
        # artifact is carried over, keep_previous_where_dead() does.
        merged = keep_previous_where_dead(
            {"centroids": centroids, "covs": arrays["covs"]}, previous, ~alive
        )
        return type(previous)(
            centroids=merged["centroids"],
            covs=merged["covs"],
            device=getattr(previous, "device", None),
        )

    def state_dict(self) -> dict:
        return {
            "counts": self.counts,
            "sums": self.sums,
            "num_clusters": self.num_clusters,
        }

    def load_state_dict(self, state: dict) -> "FixedCovarianceAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self.counts = np.asarray(state["counts"], dtype=np.float64)
        self.sums = None if state["sums"] is None else np.asarray(state["sums"], dtype=np.float64)
        return self


class SoftGaussianAccumulator:
    """
    Per-cluster soft sufficient statistics for means and full covariances.

    Accumulates the weighted sums needed for the soft EM M-step:

        n_k      = Σ_t γ_tk                    (soft count)
        S1_k     = Σ_t γ_tk · x_t              (weighted feature sum)
        S2_k     = Σ_t γ_tk · x_t x_t^T        (weighted outer-product sum)

    and at finalize() computes:

        µ_k = S1_k / n_k
        Σ_k = S2_k / n_k − µ_k µ_k^T

    Merge is exact (plain addition of sums), so this works correctly with
    chunked clustering without any approximation.

    Memory: S2 is K × D × D float64. At K=40, D=512 that is ~84 MB per
    accumulator instance. Size the cluster rqmt accordingly (≥ 16 GB is
    safe for the ls-100 setup).

    Only dense 2-D gamma posteriors [T, K] are accepted as posteriors;
    use MeanAccumulator if hard labels are needed, GaussianAccumulator if
    hard labels and covariances are needed.

    :param num_clusters: size of the label inventory
    :param dim: feature dimension; inferred from the first observation if None
    :param shrinkage: pull each covariance towards the pooled within-cluster
        covariance by this fraction before it is used - see
        :func:`shrink_covariances`. 0.0 (the default) leaves every cluster with
        its own free covariance; 1.0 ties them all to one. The knob to sweep
        when asking whether a per-cluster full covariance is supportable at
        this feature dimension, and the one that keeps such a run producing a
        PER curve rather than an uninvertible matrix.
    :param min_mass: expected-frame mass a cluster needs before it is
        re-estimated; below it the previous model's parameters are kept.

        **The default of 1.0 is a floor for the mean, not for the covariance.**
        A sample covariance from m frames has rank at most m, so at D=512 a
        cluster needs more than 512 frames before its ``[D, D]`` covariance is
        even invertible, and several times that before it is worth anything.
        One density per label over a whole corpus never comes close to the
        limit; a mixture with many densities does, and finalize() reports it
        rather than letting the next model's inversion fail.

        A hard alignment gives a cluster whole frames or none, so its implicit
        threshold is exactly one frame. Soft posteriors have no such floor: a
        cluster can end an epoch holding 1e-18 frames' worth of mass, pass a
        ``> 0`` test, and have its mean and covariance re-estimated from a
        gamma-weighted average that is numerically fine but semantically
        meaningless - in the limit it is the corpus average rather than
        anything specific to that cluster, which quietly walks a starving
        cluster into the centre of the data. Defaulting to 1.0 restores the
        hard-alignment rule: a cluster needs at least one frame of evidence to
        be re-estimated from it.
    """

    def __init__(
        self,
        num_clusters: int,
        dim: Optional[int] = None,
        min_mass: float = 1.0,
        pooling_groups: Optional[np.ndarray] = None,
        shrinkage: float = 0.0,
        accumulate_second_moment: bool = True,
        **runtime_args,
    ):
        self.num_clusters = num_clusters
        self.dim = dim
        self.min_mass = min_mass
        self.shrinkage = shrinkage
        # The second moment is the only O(D^2) statistic here - 1.07 GB at 512
        # densities and D=512 - and it exists solely to estimate covariances. A
        # run that moves means under a *fixed* covariance has no use for it, so
        # it can be switched off; finalize_arrays then carries the previous
        # model's covariances through untouched. See MixtureGaussianAccumulator's
        # update_covariances for why such a run is worth having.
        self.accumulate_second_moment = accumulate_second_moment
        self.counts = np.zeros(num_clusters, dtype=np.float64)
        self.weighted_sums = None
        self.weighted_sq = None
        self.set_pooling_groups(pooling_groups)
        if dim is not None:
            self._ensure(dim)

    def set_pooling_groups(self, pooling_groups: Optional[np.ndarray]) -> None:
        """
        Pool covariances over groups of clusters, or over none when None.

        The grouping is applied while *accumulating*, not at finalize: the second
        moment is the only statistic here that is O(D^2), and pooling it collapses
        it from one matrix per cluster to one per group. With 10 densities per label
        at D=512 that is the difference between 800 MB and 80 MB of state per chunk
        task, so it decides whether such a run is practical at all.

        Sound because a group's responsibilities sum to the group's own posterior:
        for the per-label layout, summing gamma_tc over a label's densities gives
        gamma_tl, so accumulating gamma_tl x x^T directly yields exactly the sum of
        the per-density second moments that pooling would otherwise add up.
        """
        if pooling_groups is None:
            groups, num_groups = None, self.num_clusters
        else:
            groups = np.asarray(pooling_groups, dtype=np.int64)
            if groups.shape != (self.num_clusters,):
                raise ValueError(
                    f"expected one group per cluster, got {groups.shape} for "
                    f"{self.num_clusters} clusters"
                )
            num_groups = int(groups.max()) + 1 if len(groups) else 0
        if self.weighted_sq is not None and num_groups != self.num_groups:
            # Re-shaping the second moment discards it, so only before observing.
            if self.counts.any():
                raise RuntimeError("cannot change the covariance pooling after observing")
            self.weighted_sq = np.zeros((num_groups, self.dim, self.dim), dtype=np.float64)
        self.pooling_groups = groups
        self.num_groups = num_groups

    def bind_model(self, model: ScoreModel) -> None:
        """No-op: the weighted moments are a statistic of the observations alone."""

    def _ensure(self, dim: int) -> None:
        if self.weighted_sums is None:
            self.dim = dim
            self.weighted_sums = np.zeros((self.num_clusters, dim), dtype=np.float64)
            if self.accumulate_second_moment:
                self.weighted_sq = np.zeros((self.num_groups, dim, dim), dtype=np.float64)
        elif self.dim != dim:
            raise ValueError(f"feature dim changed: {self.dim} -> {dim}")

    def _group_weights(self, gammas: np.ndarray) -> np.ndarray:
        """``[T, C] -> [T, G]``, the responsibility mass each group carries."""
        if self.pooling_groups is None:
            return gammas
        grouped = np.zeros((len(gammas), self.num_groups), dtype=np.float64)
        np.add.at(grouped.T, self.pooling_groups, gammas.T)
        return grouped

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
        T, D = features.shape
        self._ensure(D)

        # Dense-only on purpose, even though as_dense_responsibilities() would
        # happily one-hot hard labels: doing so would silently duplicate
        # GaussianAccumulator while computing the covariance from raw moments
        # rather than Welford, i.e. pick the worse-conditioned of two equivalent
        # routes. Refuse instead, and say which class to use.
        # if isinstance(posteriors, tuple) or np.asarray(posteriors).ndim != 2:
        #     raise ValueError(
        #         "SoftGaussianAccumulator requires dense 2-D gamma posteriors [T, K]; "
        #         f"got {type(posteriors).__name__} with ndim={getattr(posteriors, 'ndim', '?')}. "
        #         "Use GaussianAccumulator for hard-label input."
        #     )
        gammas = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(gammas) != T:
            raise ValueError(f"frame count mismatch: {T} features vs {len(gammas)} gammas")

        self.counts += gammas.sum(0)
        self.weighted_sums += gammas.T @ features  # [K, D]

        # Weighted outer-product sum: S2[g] += Σ_t γ_tg · x_t x_t^T, where g is the
        # cluster itself unless clusters are pooled into groups.
        # Loop over G to keep the per-call memory footprint at O(T·D) rather
        # than O(T·G·D); at G=40 the loop overhead is negligible.
        if not self.accumulate_second_moment:
            return
        group_gammas = self._group_weights(gammas)
        for g in range(self.num_groups):
            wf = features * group_gammas[:, g : g + 1]  # [T, D], broadcast weight
            self.weighted_sq[g] += wf.T @ features  # [D, T] @ [T, D] = [D, D]

    def merge(self, other: "SoftGaussianAccumulator") -> "SoftGaussianAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if not np.array_equal(
            self.pooling_groups if self.pooling_groups is not None else np.empty(0),
            other.pooling_groups if other.pooling_groups is not None else np.empty(0),
        ):
            raise ValueError("cannot merge accumulators with different covariance pooling")
        if self.accumulate_second_moment != other.accumulate_second_moment:
            raise ValueError(
                "cannot merge accumulators that disagree on accumulate_second_moment: "
                f"{self.accumulate_second_moment} vs {other.accumulate_second_moment}"
            )
        if other.weighted_sums is None:
            return self
        self._ensure(other.dim)
        self.counts += other.counts
        self.weighted_sums += other.weighted_sums
        # A means-only run never allocates the second moment (see observe), so
        # both sides hold None and there is nothing to add - the chunk merge of
        # such a run reaches here with weighted_sums set, unlike a run that never
        # observed at all, which the early return above already handled.
        if self.accumulate_second_moment:
            self.weighted_sq += other.weighted_sq
        return self

    def _finalize_pooled(self, previous: ScoreModel, means: np.ndarray, alive: np.ndarray):
        """
        One covariance per group, shared by every cluster in it.

        Each cluster keeps its own mean; the covariance is the group's total scatter
        about those means, divided by the group's mass::

            Sigma_g = ( S2_g - sum_{c in g} n_c mu_c mu_c^T ) / sum_{c in g} n_c

        ``S2_g`` is already the group's second moment - see set_pooling_groups - so
        the only per-cluster term left is the correction for each mean.

        The evidence floor moves to the group with the parameter: a cluster too
        starved for its own mean is still covered by a group that is not, which is
        most of the point of pooling. Means and covariances therefore have different
        alive masks, hence the per-artifact form of keep_previous_where_dead.
        """
        groups = self.pooling_groups
        group_mass = np.bincount(groups, weights=self.counts, minlength=self.num_groups)

        # sum_{c in g} n_c mu_c mu_c^T, over clusters that actually saw data
        weighted_mu = self.counts[:, np.newaxis] * means
        correction = np.zeros((self.num_groups, means.shape[1], means.shape[1]))
        np.add.at(correction, groups, weighted_mu[:, :, np.newaxis] * means[:, np.newaxis, :])

        group_alive = alive_mask(group_mass, self.min_mass)
        divisor = np.where(group_alive, group_mass, 1.0)[:, np.newaxis, np.newaxis]
        group_covs = (self.weighted_sq - correction) / divisor
        group_covs = (group_covs + group_covs.transpose(0, 2, 1)) / 2

        live = group_covs[group_alive]
        if len(live):
            eigenvalues = np.linalg.eigvalsh(live)
            threshold = eigenvalues.max(axis=1) * live.shape[-1] * np.finfo(np.float64).eps
            singular = eigenvalues.min(axis=1) <= threshold
            if singular.any():
                starved = np.flatnonzero(group_alive)[singular]
                print(
                    f"WARNING: {len(starved)} pooled group(s) produced a singular "
                    f"covariance and keep the previous epoch's; mass "
                    f"{np.array2string(group_mass[starved], precision=1, threshold=12)}",
                    flush=True,
                )
                group_alive[starved] = False

        return keep_previous_where_dead(
            {"centroids": means, "covs": group_covs[groups]},
            previous,
            {"centroids": ~alive, "covs": ~group_alive[groups]},
        )

    def scale(self, factor: float) -> "SoftGaussianAccumulator":
        """
        Multiply the accumulated statistics by ``factor``, in place.

        The decay half of the statistics-level EMA ``S <- a*S + (1-a)*N``;
        :meth:`merge` is the addition half, so that recursion needs no
        arithmetic of its own (see :mod:`.ema`). Every statistic here is a sum
        over frames, so scaling it is the stochastic-approximation step of
        online EM: the result is still a (fractional) count and ``finalize``
        normalizes it exactly as it does a plain epoch's.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        self.counts = self.counts * factor
        if self.weighted_sums is not None:
            self.weighted_sums = self.weighted_sums * factor
        if self.weighted_sq is not None:
            self.weighted_sq = self.weighted_sq * factor
        return self

    def finalize_arrays(self, previous: ScoreModel) -> dict[str, np.ndarray]:
        if self.weighted_sums is None:
            raise RuntimeError("nothing accumulated; cannot finalize")

        missing = {"centroids", "covs"} - set(previous.artifacts())
        if missing:
            raise TypeError(
                f"SoftGaussianAccumulator needs a model with {sorted(missing)}, "
                f"got {type(previous).__name__} with {sorted(previous.artifacts())}"
            )

        alive = alive_mask(self.counts, self.min_mass)
        # Substitute for the dead rows only to keep the division quiet - they
        # are replaced wholesale below, so the value is irrelevant.
        n = if_alive_else(self.counts, self.min_mass)

        means = self.weighted_sums / n[:, np.newaxis]  # [K, D]

        if self.pooling_groups is not None:
            return self._finalize_pooled(previous, means, alive)

        if not self.accumulate_second_moment:
            # Means-only: the covariance is an input to this run, not an
            # estimate from it, so it comes through untouched. The dead-cluster
            # rule still applies to the means, and passing the previous
            # covariances through it is a no-op that keeps the artifact sets
            # matching - the same shape FixedCovarianceAccumulator uses.
            return keep_previous_where_dead(
                {"centroids": means, "covs": previous.artifacts()["covs"]},
                previous,
                ~alive,
            )

        # Σ_k = E[x x^T | k] - µ_k µ_k^T.
        # Raw moments rather than the Welford form used in pca.py: measured on
        # these features (near zero-mean, |mean| ~ 1.3 vs std ~ 9.5) the two
        # agree to a relative 3e-16 and both stay positive definite, so the
        # cancellation that motivates Welford there does not bite here.
        second_moment = self.weighted_sq / n[:, np.newaxis, np.newaxis]  # [K, D, D]
        mu_outer = means[:, :, np.newaxis] * means[:, np.newaxis, :]     # [K, D, D]
        covs = second_moment - mu_outer
        # Symmetrize to correct for floating-point asymmetry in the accumulation.
        covs = (covs + covs.transpose(0, 2, 1)) / 2
        # Before the rank check below, not after: shrinkage exists precisely to
        # make an otherwise-singular covariance usable, so the check has to see
        # what the next model will actually be built from.
        covs = shrink_covariances(covs, self.counts, self.shrinkage, alive)

        # A cluster can clear min_mass and still not support a full [D, D]
        # covariance: a sample covariance from m frames has rank <= m, so below
        # ~D frames it is singular by construction. np.linalg.inv then raises
        # LinAlgError inside the *next* model's constructor, an epoch later and
        # in a place that says nothing about which cluster starved - which is
        # exactly how this was first seen (128 densities at D=512, one
        # re-estimated from a single frame).
        #
        # Treated as dead rather than repaired: this only fires where the
        # alternative is a covariance that cannot be inverted, so it changes no
        # result that was previously valid. min_mass is the knob that stops it
        # arising at all, and unlike the mean's floor it has to scale with the
        # feature dimension - see the class docstring.
        if alive.any():
            live_covs = covs[alive]
            eigenvalues = np.linalg.eigvalsh(live_covs)
            threshold = (
                eigenvalues.max(axis=1) * live_covs.shape[-1] * np.finfo(np.float64).eps
            )
            singular = eigenvalues.min(axis=1) <= threshold
            if singular.any():
                starved = np.flatnonzero(alive)[singular]
                print(
                    f"WARNING: {len(starved)} of {int(alive.sum())} re-estimated clusters "
                    f"produced a singular covariance and keep the previous epoch's "
                    f"parameters instead. Frame mass: "
                    f"{np.array2string(self.counts[starved], precision=1, threshold=12)}. "
                    f"A [{live_covs.shape[-1]}, {live_covs.shape[-1]}] covariance needs more "
                    f"than {live_covs.shape[-1]} frames; raise min_mass above the feature "
                    f"dimension to decide this by evidence rather than by rank.",
                    flush=True,
                )
                alive[starved] = False

        # Not fatal, but the same shortage one step earlier: enough frames to
        # invert, not enough to mean anything.
        underdetermined = alive & (self.counts <= covs.shape[-1])
        if underdetermined.any():
            print(
                f"WARNING: {int(underdetermined.sum())} cluster(s) re-estimated from fewer "
                f"frames than the {covs.shape[-1]} dimensions of their covariance; the "
                f"result is invertible but rank-poor. Consider min_mass > "
                f"{covs.shape[-1]}.",
                flush=True,
            )

        merged = keep_previous_where_dead(
            {"centroids": means, "covs": covs}, previous, ~alive
        )
        return merged

    def finalize(self, previous: ScoreModel) -> GaussianModel:
        merged = self.finalize_arrays(previous)
        return GaussianModel(
            merged["centroids"], merged["covs"], device=getattr(previous, "device", None)
        )

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "dim": self.dim,
            "counts": self.counts,
            "weighted_sums": self.weighted_sums,
            "weighted_sq": self.weighted_sq,
            "accumulate_second_moment": self.accumulate_second_moment,
            "pooling_groups": self.pooling_groups,
            "shrinkage": self.shrinkage,
        }

    def load_state_dict(self, state: dict) -> "SoftGaussianAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self.dim = state["dim"]
        self.shrinkage = float(state.get("shrinkage", self.shrinkage))
        self.accumulate_second_moment = bool(
            state.get("accumulate_second_moment", self.accumulate_second_moment)
        )
        # Before the arrays: it decides how many groups the second moment has.
        self.set_pooling_groups(state.get("pooling_groups"))
        self.counts = np.asarray(state["counts"], dtype=np.float64)
        self.weighted_sums = (
            None if state["weighted_sums"] is None
            else np.asarray(state["weighted_sums"], dtype=np.float64)
        )
        self.weighted_sq = (
            None if state["weighted_sq"] is None
            else np.asarray(state["weighted_sq"], dtype=np.float64)
        )
        return self


class MixtureGaussianAccumulator:
    """
    Sufficient statistics for a :class:`.models.GaussianMixtureModel`: the
    density means and covariances, plus the per-label mixture weights.

    The recognizer reports posteriors over *labels*, but the parameters being
    re-estimated are indexed by *density*, so an E-step sits between the two -
    ``p(density | label, frame)`` under the model this epoch recognized with.
    That step is the model's (:meth:`~.models.GaussianMixtureModel.responsibilities`);
    this class supplies the model via :meth:`bind_model` and then does nothing
    but add up what comes back::

        n_c    = sum_t gamma_tc                 -> delegated to SoftGaussianAccumulator
        S1_c   = sum_t gamma_tc x_t                              "
        S2_c   = sum_t gamma_tc x_t x_t^T                        "
        C_lc   = sum_t gamma_tl p(c | l, x_t)   -> weighted_c, this class

    and finalizes to ``w_lc = C_lc / sum_c C_lc``.

    **Merge is still exact.** Holding a model looks like it should make the
    result depend on evaluation order, but the model is the epoch's input, held
    fixed across every chunk and every frame; the E-step is a function of it
    and of one frame, never of what has been observed so far. Both statistics
    are therefore plain sums, and the tests assert associativity over chunk
    counts the same way they do for the other accumulators.

    Nothing here knows how ``mixtures`` is laid out. A shared codebook carries
    one weight per label per density and per-label densities carry one per
    label per own-density, but either way the statistic has the same shape as
    the model's own ``mixtures`` array and is normalized along its last axis -
    so both layouts, and any later one, use this class unchanged.

    :param num_clusters: the *label* count L, matching what the epoch job
        injects and what the recognizer scores against. The density count is a
        property of the model, not of the pipeline, so it is taken from the
        bound model (or from a loaded state) rather than configured here.
    :param num_densities: density count, if it has to be fixed up front
    :param dim: feature dimension; inferred from the first observation if None
    :param min_mass: evidence a density needs before its mean and covariance
        are re-estimated, and a label needs before its mixture weights are;
        below it the previous model's values are kept. See
        :class:`SoftGaussianAccumulator` for why soft posteriors need a floor
        that hard alignments get for free.
    :param pool_covariances: give every density of a label one shared covariance
        instead of its own. Each density keeps its own mean, so the mixture still
        models several modes per label; only the shape around them is tied. Two
        reasons to want it: a full ``[D, D]`` covariance needs far more evidence
        than a mean, and pooling gives it the label's whole mass rather than one
        density's share; and the second moment is the only O(D^2) statistic, so
        pooling shrinks the per-chunk state by the number of densities per label -
        at 10 densities and D=512, 800 MB becomes 80 MB. Requires a model whose
        densities can be grouped, i.e. per-label rather than a shared codebook.
    :param update_densities: re-estimate the density means and covariances
        alongside the mixture weights. Set False to train the weights alone
        against a frozen codebook - the densities the epoch scores with are the
        densities the next epoch scores with, and only ``p(density | label)``
        moves.

        This is not just the density update skipped at finalize: with it off,
        nothing is accumulated for the densities at all. The second moment is
        the only O(D^2) statistic in the pipeline, so a frozen codebook drops
        the per-chunk state from ``num_densities x D x D`` to the mixture
        statistic's ``mixtures.shape`` - at 512 densities and D=512, from
        ~1 GB to ~160 kB. That is what makes a large shared codebook practical
        here at all, and it is the reason this is a flag on the accumulator
        rather than a post-hoc overwrite of the finalized arrays.

        Note what it does *not* freeze: the search still runs against the model
        each epoch, so the label posteriors change from epoch to epoch and the
        weights keep moving. What is fixed is the codebook the weights are
        defined over.
    :param update_covariances: whether the density *shapes* are re-estimated too.
        None follows ``update_densities``, which is every existing run.
        ``update_densities=True, update_covariances=False`` is the means-only
        mode: densities move to where the labels put them, but their shape stays
        an input to the run.

        Worth having for two reasons that are independent of each other. The
        statistical one: at D=512 a full covariance has 131,328 free parameters,
        and 512 densities over ls-100h leaves ~6,800 vectors each - 0.05 vectors
        per parameter, which is the regime that produced an acoustic model
        scoring below chance. The mechanical one: with a shared codebook the
        covariance *scale* sets how soft the density assignment is, and softness
        is what lets label information reach the means at all - measured, a
        centroid update at the corpus covariance is ~83% label-blind and
        degenerates to k-means. Re-estimating the covariance would pull that
        scale straight back to the data's own within an epoch, so the softness
        one deliberately dials in is only stable while the shapes are frozen.
    :param mixture_floor: weight handed to every density of a label before
        renormalizing. Zero - the default - is textbook EM and is what keeps
        this reproducible against any other EM implementation, but note that
        it makes zero an absorbing state: a density that loses all its weight
        under a label can never regain it, so the effective mixture size only
        ever shrinks. A small value (1e-6) trades exactness for the same
        "stay a reachable candidate" property that ``keep_previous_where_dead``
        gives the means.
    :param over_relaxation: ``eta``, the step taken along the EM update of the
        mixture weights, in log space and per label row:
        ``w <- normalize(w_prev**(1 - eta) * w_em**eta)``. 1.0 - the default - is
        the plain M-step, bit for bit. Near a slow phase EM shrinks the error
        along direction i by a factor ``l_i`` per epoch (the fraction of
        missing information there); this makes it ``1 - eta*(1 - l_i)``, so
        slow directions move ``eta`` times faster and every direction stays
        stable for ``eta < 2``. Larger values are stable only if no direction is
        fast. The result is held at or above the lowest weight the plain
        M-step's ``mixture_floor`` can produce for the row, so a weight the
        extrapolation keeps shrinking cannot underflow to an absorbing zero -
        which is also why a nonzero floor is required. Weights only: a run that
        also re-estimates the densities would mix over-relaxed weights with
        plain-EM densities, so ``update_densities=True`` is refused. Each
        finalize leaves the step it took in ``step_statistics``.
    """

    # observe() takes the density costs the sequence was scored with
    # (``scores_gaussian``), so run_chunk can skip a second Mahalanobis pass.
    takes_scores_gaussian = True

    def __init__(
        self,
        num_clusters: int,
        num_densities: Optional[int] = None,
        dim: Optional[int] = None,
        min_mass: float = 1.0,
        mixture_floor: float = 0.0,
        pool_covariances: bool = False,
        update_densities: bool = True,
        update_covariances: Optional[bool] = None,
        over_relaxation: float = 1.0,
        **runtime_args,
    ):
        if over_relaxation <= 0.0:
            raise ValueError(f"over_relaxation must be positive, got {over_relaxation}")
        if over_relaxation != 1.0 and mixture_floor <= 0.0:
            raise ValueError(
                "over_relaxation extrapolates in log space and needs every weight "
                "positive; pass a nonzero mixture_floor"
            )
        if over_relaxation != 1.0 and update_densities:
            raise ValueError(
                "over_relaxation extrapolates the mixture weights only; with "
                "update_densities=True the densities would take plain EM steps "
                "alongside them. Freeze the densities (update_densities=False)"
            )
        self.num_clusters = num_clusters
        self.num_densities = num_densities
        self.dim = dim
        self.min_mass = min_mass
        self.mixture_floor = mixture_floor
        self.over_relaxation = over_relaxation
        #: Set by finalize when over_relaxation != 1: the step it took.
        self.step_statistics: Optional[Dict[str, float]] = None
        self.pool_covariances = pool_covariances
        self.update_densities = update_densities
        # None means "whatever update_densities says", which is what keeps every
        # existing run's behaviour and job hash unchanged. Setting it False with
        # update_densities True is the means-only mode: the densities move to
        # where the labels put them, but their shape is an input to the run.
        self.update_covariances = (
            update_densities if update_covariances is None else update_covariances
        )
        if self.update_covariances and not update_densities:
            raise ValueError(
                "update_covariances=True with update_densities=False is ill-posed: a "
                "covariance is estimated about a mean this run is not estimating"
            )
        if pool_covariances and not update_densities:
            raise ValueError(
                "pool_covariances ties covariances while they are being re-estimated, "
                "but update_densities=False means they are not re-estimated at all; "
                "pass only one of the two"
            )
        if pool_covariances and not self.update_covariances:
            raise ValueError(
                "pool_covariances has nothing to tie when update_covariances=False: "
                "the covariances come through from the previous model untouched"
            )
        self.model: Optional[MixtureModelBase] = None
        self.weighted_c = None
        self.gaussian_accumulator = SoftGaussianAccumulator(
            num_densities or 0, dim, min_mass=min_mass,
            accumulate_second_moment=self.update_covariances,
        )
        if num_densities is not None:
            # Without a model to ask, assume the shared-codebook layout - the
            # only one whose statistic shape follows from the two counts alone.
            self._allocate(num_densities, (num_clusters, num_densities))

    @property
    def num_labels(self) -> int:
        """Alias for ``num_clusters``; see the model's class docstring."""
        return self.num_clusters

    def _allocate(self, num_densities: int, mixture_shape) -> None:
        if self.num_densities is not None and self.num_densities != num_densities:
            raise ValueError(
                f"density count changed: {self.num_densities} -> {num_densities}"
            )
        if mixture_shape[0] != self.num_clusters:
            raise ValueError(
                f"mixture statistic is indexed by {mixture_shape[0]} labels, "
                f"accumulator has {self.num_clusters}"
            )
        self.num_densities = num_densities
        # Shaped like the model's own mixtures rather than derived from the two
        # counts, which is what keeps this layout-agnostic. Unlike the moment
        # arrays it does not depend on the feature dimension, so it can be
        # allocated as soon as the model is known.
        self.weighted_c = np.zeros(tuple(mixture_shape), dtype=np.float64)
        groups = self.gaussian_accumulator.pooling_groups if self.gaussian_accumulator else None
        self.gaussian_accumulator = SoftGaussianAccumulator(
            num_densities, self.dim, min_mass=self.min_mass, pooling_groups=groups,
            accumulate_second_moment=self.update_covariances,
        )

    def bind_model(self, model: ScoreModel) -> None:
        if not hasattr(model, "responsibilities"):
            raise TypeError(
                f"{type(self).__name__} needs a model that can split label posteriors "
                f"into density posteriors (a responsibilities() method), got "
                f"{type(model).__name__}"
            )
        if model.num_clusters != self.num_clusters:
            raise ValueError(
                f"label count mismatch: accumulator has {self.num_clusters}, "
                f"{type(model).__name__} scores {model.num_clusters}"
            )
        if self.pool_covariances and model.density_groups() is None:
            raise TypeError(
                f"{type(model).__name__} has no per-label density grouping, so its "
                f"covariances cannot be pooled; use a per-label mixture model"
            )
        self.model = model
        if self.weighted_c is None:
            self._allocate(model.num_densities, model.mixtures.shape)
        if self.pool_covariances:
            self.gaussian_accumulator.set_pooling_groups(model.density_groups())
        elif self.weighted_c.shape != model.mixtures.shape:
            raise ValueError(
                f"mixture layout mismatch: accumulator holds {self.weighted_c.shape}, "
                f"{type(model).__name__} has {model.mixtures.shape}"
            )

    def observe(
        self,
        features: np.ndarray,
        posteriors: Posteriors,
        *,
        scores_gaussian: Optional[np.ndarray] = None,
    ) -> None:
        """
        :param scores_gaussian: the bound model's ``scores_gaussian(features)``,
            if the caller already computed it; recomputed if not
        """
        if self.model is None:
            raise RuntimeError(
                "bind_model() must be called before observe(): the E-step splitting "
                "label posteriors into density posteriors is defined against the "
                "model this epoch recognized with"
            )
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
        T, D = features.shape
        if self.dim is None:
            self.dim = D
        elif self.dim != D:
            raise ValueError(f"feature dim changed: {self.dim} -> {D}")

        # Hard labels, n-best and dense gammas alike: a Viterbi-guided mixture
        # run is just the degenerate case where each frame's label posterior is
        # one-hot, and the density split below is unchanged by that.
        gammas = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(gammas) != T:
            raise ValueError(f"frame count mismatch: {T} features vs {len(gammas)} gammas")

        # Only when given: LabelCovarianceMixtureModel names its precomputed
        # costs differently, and run_chunk only hands them to models that have
        # scores_with_gaussian.
        precomputed = {} if scores_gaussian is None else {"scores_gaussian": scores_gaussian}
        density_gammas, joint_counts = self.model.responsibilities(features, gammas, **precomputed)
        if self.update_densities:
            self.gaussian_accumulator.observe(features, density_gammas)
        self.weighted_c += joint_counts

    def merge(self, other: "MixtureGaussianAccumulator") -> "MixtureGaussianAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"label count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if other.weighted_c is None:
            return self
        if self.weighted_c is None:
            self._allocate(other.num_densities, other.weighted_c.shape)
        self.dim = self.dim if self.dim is not None else other.dim
        self.weighted_c += other.weighted_c
        self.gaussian_accumulator.merge(other.gaussian_accumulator)
        return self

    def scale(self, factor: float) -> "MixtureGaussianAccumulator":
        """
        Multiply the accumulated statistics by ``factor``, in place.

        The decay half of the statistics-level EMA ``S <- a*S + (1-a)*N``;
        :meth:`merge` is the addition half, so that recursion needs no
        arithmetic of its own (see :mod:`.ema`). Every statistic here is a sum
        over frames, so scaling it is the stochastic-approximation step of
        online EM: the result is still a (fractional) count and ``finalize``
        normalizes it exactly as it does a plain epoch's.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        if self.weighted_c is not None:
            self.weighted_c = self.weighted_c * factor
        # The density moments live in the nested accumulator, which owns its own
        # counts; scaling only weighted_c here would decay the mixture weights
        # while leaving the densities on an undecayed statistic.
        self.gaussian_accumulator.scale(factor)
        return self

    def finalize(self, previous: ScoreModel) -> MixtureModelBase:
        if self.weighted_c is None:
            raise RuntimeError("nothing accumulated; cannot finalize")
        if not isinstance(previous, MixtureModelBase):
            raise TypeError(
                f"{type(self).__name__} needs a GaussianMixtureModel to fall back on, "
                f"got {type(previous).__name__}"
            )

        # The densities fall back on the previous *density* parameters, which
        # is exactly the artifact set the inner accumulator understands - the
        # generic keep_previous_where_dead() rule assumes artifacts are indexed
        # by the axis it is masking, and `mixtures` is indexed by label, not by
        # density. Handing it the wrapped GaussianModel keeps the rule honest
        # and leaves the mixture weights to the label-indexed rule below.
        if not self.update_densities:
            # Nothing was accumulated for them, so there is nothing to fall back
            # from either: the previous model's densities are the answer, not a
            # substitute for a missing one.
            gaussian_arrays = {"centroids": previous.centroids, "covs": previous.covs}
        else:
            if self.pool_covariances and self.gaussian_accumulator.pooling_groups is None:
                # The reduce step builds accumulators from state alone and never binds a
                # model, so the grouping comes from the model being finalized against.
                self.gaussian_accumulator.set_pooling_groups(previous.density_groups())
            gaussian_arrays = self.gaussian_accumulator.finalize_arrays(previous.gaussian_model)

        if previous.mixtures.shape != self.weighted_c.shape:
            raise ValueError(
                f"mixture shape mismatch: accumulated {self.weighted_c.shape} vs "
                f"{type(previous).__name__}'s {previous.mixtures.shape}"
            )

        label_mass = self.weighted_c.sum(axis=-1)
        alive = alive_mask(label_mass, self.min_mass)
        weighted = self.weighted_c + self.mixture_floor
        mixtures = weighted / if_alive_else(label_mass, self.min_mass)[:, np.newaxis]
        if self.mixture_floor:
            mixtures /= mixtures.sum(axis=-1, keepdims=True)
        # A label nobody aligned to keeps its previous weights: leaving the
        # unnormalized row in would violate the model's own row-sum invariant,
        # and zeroing it would leave the label with no emission probability at
        # all. Same rule as the means, on the axis that actually applies.
        mixtures[~alive] = previous.mixtures[~alive]
        if self.over_relaxation != 1.0:
            mixtures = self._over_relax(previous.mixtures, mixtures, alive, label_mass)

        # type(previous), not a named class: the next model is the same kind
        # of model as the one this epoch recognized with, whichever mixture
        # layout that is. Naming one here is what would force this accumulator
        # to grow a branch per layout.
        return type(previous)(
            centroids=gaussian_arrays["centroids"],
            covs=gaussian_arrays["covs"],
            mixtures=mixtures,
            device=getattr(previous, "device", None),
        )

    def _over_relax(
        self,
        previous: np.ndarray,
        em: np.ndarray,
        alive: np.ndarray,
        label_mass: np.ndarray,
    ) -> np.ndarray:
        """``over_relaxation`` times the EM step, in log space (see the class docstring).

        Labels that are not alive keep ``em``'s row, which is already their
        previous weights.
        """
        eta = self.over_relaxation
        # The lowest weight the plain M-step can give a row: a density that got
        # no mass at all, floor / (mass + K * floor).
        floor = self.mixture_floor
        lower = (floor / (label_mass + em.shape[-1] * floor))[:, np.newaxis]
        log_prev = np.log(np.maximum(np.asarray(previous, dtype=np.float64), lower))
        log_w = log_prev + eta * (np.log(em) - log_prev)
        log_w -= log_w.max(axis=-1, keepdims=True)
        w = np.exp(log_w)
        w /= w.sum(axis=-1, keepdims=True)
        # Clip at the floor and take the mass that costs from the rest of the
        # row. Rescaling the rest can push another weight under, so repeat; each
        # round clips at least one more weight or ends. A row always keeps
        # something above the floor, since K * floor / (mass + K * floor) < 1.
        clipped = np.zeros(w.shape, dtype=bool)
        while True:
            under = (w < lower) & ~clipped
            if not under.any():
                break
            clipped |= under
            rest = np.where(clipped, 0.0, w)
            room = 1.0 - (clipped * lower).sum(axis=-1, keepdims=True)
            w = np.where(clipped, lower, rest * room / rest.sum(axis=-1, keepdims=True))
        w = np.where(alive[:, np.newaxis], w, em)

        def kl_from_previous(p):
            if not alive.any():
                return 0.0
            return float((p * (np.log(p) - log_prev)).sum(axis=-1)[alive].mean())

        self.step_statistics = {
            "mixture_over_relaxation": eta,
            # Mean over live labels of KL(new || previous), in nats: the step EM
            # proposed and the one taken. About eta**2 apart while steps are small.
            "mixture_kl_em_step": kl_from_previous(em),
            "mixture_kl_applied_step": kl_from_previous(w),
            "mixture_clipped_fraction": float(clipped[alive].mean()) if alive.any() else 0.0,
        }
        return w

    def state_dict(self) -> dict:
        # The inner accumulator's state is nested rather than flattened: it
        # owns counts as well as the moment arrays, and copying a subset of its
        # fields up here is how they got silently dropped before.
        return {
            "num_clusters": self.num_clusters,
            "num_densities": self.num_densities,
            "dim": self.dim,
            "weighted_c": self.weighted_c,
            "pool_covariances": self.pool_covariances,
            "update_densities": self.update_densities,
            "update_covariances": self.update_covariances,
            "gaussian": self.gaussian_accumulator.state_dict(),
        }

    def load_state_dict(self, state: dict) -> "MixtureGaussianAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"label count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        num_densities = state["num_densities"]
        if num_densities is not None and self.weighted_c is None:
            # The stored statistic's own shape is the layout: nothing else has
            # to be recorded for a loaded accumulator to match the one that
            # wrote it, whichever model class that was.
            self._allocate(int(num_densities), np.shape(state["weighted_c"]))
        elif num_densities is not None and int(num_densities) != self.num_densities:
            raise ValueError(
                f"density count mismatch: {num_densities} vs {self.num_densities}"
            )
        self.dim = state["dim"]
        self.pool_covariances = bool(state.get("pool_covariances", self.pool_covariances))
        self.update_densities = bool(state.get("update_densities", self.update_densities))
        self.update_covariances = bool(
            state.get("update_covariances", self.update_covariances)
        )
        self.weighted_c = (
            None if state["weighted_c"] is None
            else np.asarray(state["weighted_c"], dtype=np.float64)
        )
        self.gaussian_accumulator.load_state_dict(state["gaussian"])
        return self


def semi_tied_transform(
    scatters: np.ndarray,
    counts: np.ndarray,
    transform: np.ndarray,
    iterations: int,
) -> Tuple[np.ndarray, List[float]]:
    """
    Gales' row-by-row estimate of a semi-tied transform - MLLT, in Gopinath's name.

    Finds the ``A`` that maximizes the likelihood of the covariances
    ``Sigma_l = A^-1 diag(lambda_l) A^-T`` when every ``lambda_l`` sits at its
    optimum ``diag(A W_l A^T)``; per unit of mass and up to constants,

        F(A) = log |det A| - 1/2 sum_l (n_l / N) sum_i log(a_i W_l a_i^T).

    Each iteration holds the variances, then replaces every row in turn by the
    exact maximizer of the auxiliary function in that row (Gales 1999),

        a_i = c_i G_i^-1 sqrt(N / (c_i G_i^-1 c_i^T)),    G_i = sum_l n_l W_l / lambda_li,

    with ``c_i`` the row's cofactors - row ``i`` of ``det(A) A^-T``, of which only
    the direction matters. Neither step can lower ``F``, which the tests assert.
    ``A^-1`` is kept current with a rank-one update per row, and recomputed at
    the start of every iteration so the updates cannot drift.

    :param scatters: ``[L, D, D]``, each class's scatter ``W_l`` (normalized)
    :param counts: ``[L]``, the masses ``n_l``
    :param transform: ``[D, D]``, the starting point; rows are the ``a_i``
    :return: the transform, and ``F`` before the first iteration and after each
    """
    scatters = np.asarray(scatters, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    transform = np.array(transform, dtype=np.float64, copy=True)
    dim = transform.shape[0]
    flat = scatters.reshape(len(scatters), -1)

    def variances_of(a):
        return np.einsum("id,lde,ie->li", a, scatters, a, optimize=True)        # [L, D]

    def objective(a, variances):
        return float(
            np.linalg.slogdet(a)[1] - 0.5 * (counts @ np.log(variances).sum(axis=1)) / total
        )

    variances = variances_of(transform)
    history = [objective(transform, variances)]
    # Linear convergence, and slow where the scatters are far from sharing a
    # basis: on a D=6 toy with cond(H) = 26 the gap to the optimum is 1e-3 after
    # 50 iterations and 1e-10 after 2000. Hence the warm start in
    # LabelCovarianceAccumulator, which spreads the iterations over the epochs.
    for _ in range(iterations):
        inverse = np.linalg.inv(transform)
        for i in range(dim):
            gram = ((counts / variances[:, i]) @ flat).reshape(dim, dim)
            cofactors = inverse[:, i]
            solved = np.linalg.solve(gram, cofactors)
            row = solved * np.sqrt(total / (cofactors @ solved))
            # Sherman-Morrison for A + e_i (row - a_i)^T, evaluated against the
            # inverse before this row changes.
            correction = (row - transform[i]) @ inverse
            inverse -= np.outer(inverse[:, i], correction) / (1.0 + correction[i])
            transform[i] = row
        variances = variances_of(transform)
        history.append(objective(transform, variances))
    return transform, history


def semi_tied_objective(transform: np.ndarray, scatters: np.ndarray, counts: np.ndarray) -> float:
    """``F(A)`` of :func:`semi_tied_transform`, for comparing starting points."""
    _, history = semi_tied_transform(scatters, counts, transform, 0)
    return history[0]


def common_diagonalizer(covs: np.ndarray, tolerance: float = 1e-6) -> Optional[np.ndarray]:
    """
    A transform ``A`` with every ``A Sigma_l A^T`` diagonal, or None if there is none.

    For semi-tied covariances ``A^-1 diag(lambda_l) A^-T`` this recovers their
    transform, up to the order and scale of its rows - neither of which changes
    the model or :func:`semi_tied_transform`'s objective. One generalized
    eigenproblem does it: the eigenvectors of a generic combination of the
    ``Sigma_l`` against their mean diagonalize every one of them. Equal
    covariances are the degenerate case, and any whitening of them is a valid
    answer, which is what comes back.

    :param tolerance: the largest off-diagonal correlation of a rotated
        covariance still accepted as diagonal
    """
    import scipy.linalg

    covs = np.asarray(covs, dtype=np.float64)
    # Distinct weights: two directions get the same eigenvalue - and rotate
    # freely into each other - only if every label scales them alike, in which
    # case any rotation of the pair is as diagonal as the other.
    probe = np.tensordot(np.linspace(1.0, 2.0, len(covs)), covs, axes=1)
    _, vectors = scipy.linalg.eigh(probe, covs.mean(axis=0))
    transform = vectors.T
    rotated = np.einsum("id,lde,je->lij", transform, covs, transform, optimize=True)
    variances = np.einsum("lii->li", rotated)
    correlation = rotated / np.sqrt(variances[:, :, np.newaxis] * variances[:, np.newaxis, :])
    off_diagonal = np.abs(correlation - np.eye(covs.shape[-1])).max()
    return transform if off_diagonal <= tolerance else None


class LabelCovarianceAccumulator:
    """
    Sufficient statistics and M-step for a :class:`.models.LabelCovarianceMixtureModel`.

    The E-step is the model's (:meth:`~.models.LabelCovarianceMixtureModel.joint_posteriors`),
    ``gamma_tlk = gamma_tl p(k | l, x_t)``, and three sums of it cover every
    parameter::

        n_lk  = sum_t gamma_tlk               [L, K]       mixture weights
        F_lk  = sum_t gamma_tlk x_t           [L, K, D]    means, covariances
        S_l   = sum_t gamma_tl x_t x_t^T      [L, D, D]    covariances

    ``F`` stays per label rather than being summed into one first moment per
    codeword, because both other parameters need it that way. A mean shared by
    labels with different covariances is the precision-weighted solution

        mu_k = (sum_l n_lk P_l)^-1  sum_l P_l F_lk,        P_l = Sigma_l^-1,

    and a label's scatter about the codebook is

        n_l W_l = S_l - sum_k (F_lk mu_k^T + mu_k F_lk^T) + sum_k n_lk mu_k mu_k^T,

    which can therefore be taken about the *new* means. The M-step is ECM
    (Meng & Rubin 1993): means under the previous covariances, then covariances
    about the new means. Each step maximizes the auxiliary function exactly in
    its own block, so the likelihood still cannot decrease. With every
    covariance equal, the mean step is the plain weighted average.

    ``F`` and ``S`` are 84 MB each at L=40, K=512, D=512 in float64, and each is
    allocated only when the configured update reads it: a weights-only run
    holds the ``[L, K]`` counts and nothing else.

    :param num_clusters: the label count L, injected by the epoch job
    :param min_mass: evidence a label needs before its weights and covariance
        are re-estimated, and a codeword before its mean is; below it the
        previous model's values are kept
    :param mixture_floor: as for :class:`MixtureGaussianAccumulator`
    :param update_means: re-estimate the codebook means
    :param covariance: how each label's covariance is re-estimated from ``W_l``:

        ``"fixed"``
            not at all; the input covariances are carried through untouched
        ``"full"``
            a free full covariance per label, smoothed towards the pooled
            scatter by ``covariance_relevance``
        ``"semi_tied"``
            ``Sigma_l = A^-1 diag(lambda_l) A^-T``: one transform for every
            label, a diagonal per label - semi-tied covariances, i.e. MLLT.
            Each epoch takes ``semi_tied_iterations`` rounds of
            :func:`semi_tied_transform`, resuming from the previous epoch's
            ``A`` (:func:`common_diagonalizer`) or from the pooled scatter's
            whitening eigenbasis, whichever scores higher.
        ``"diagonal"``
            a diagonal per label in the pooled scatter's eigenbasis - the
            semi-tied form with ``A`` not trained. The basis follows the pooled
            scatter from epoch to epoch, so unlike the other modes this one is
            not guaranteed monotone.

    :param covariance_relevance: ``r`` in the smoothing towards the pooled
        within-codeword scatter ``W = sum_l n_l W_l / sum_l n_l``::

            Sigma_l = (n_l W_l + r W) / (n_l + r)

        on the full matrices for ``"full"``, on the diagonal variances for the
        other two. 0 is maximum likelihood; ``inf`` ties every label to ``W``,
        i.e. one re-estimated covariance for the whole model. Count-based rather
        than a fixed fraction (:func:`shrink_covariances`) because label masses
        span two orders of magnitude here - ~2,000 to ~320,000 segments on ls-100
        - so a frequent label should keep nearly its own estimate while a rare
        one stays near the pool. At D=512 an unsmoothed ``W_l`` is not even
        invertible below 512 segments.
    :param semi_tied_iterations: rounds of the row update per epoch
    """

    COVARIANCE_MODES = ("fixed", "full", "semi_tied", "diagonal")
    #: Codewords per batched solve in the mean update, which forms a
    #: ``[block, D, D]`` system: 134 MB at 64 and D=512. Memory only.
    MEAN_BLOCK = 64

    def __init__(
        self,
        num_clusters: int,
        min_mass: float = 1.0,
        mixture_floor: float = 0.0,
        update_means: bool = False,
        covariance: str = "fixed",
        covariance_relevance: float = 0.0,
        semi_tied_iterations: int = 20,
        **runtime_args,
    ):
        if covariance not in self.COVARIANCE_MODES:
            raise ValueError(
                f"covariance={covariance!r}; expected one of {self.COVARIANCE_MODES}"
            )
        if covariance_relevance < 0:
            raise ValueError(f"covariance_relevance must be >= 0, got {covariance_relevance}")
        if covariance == "fixed" and covariance_relevance:
            raise ValueError(
                "covariance_relevance smooths re-estimated covariances, but "
                "covariance='fixed' re-estimates none; it would do nothing"
            )
        if semi_tied_iterations < 0:
            raise ValueError(f"semi_tied_iterations must be >= 0, got {semi_tied_iterations}")
        self.num_clusters = num_clusters
        self.min_mass = min_mass
        self.mixture_floor = mixture_floor
        self.update_means = update_means
        self.covariance = covariance
        self.covariance_relevance = float(covariance_relevance)
        self.semi_tied_iterations = semi_tied_iterations
        self.model: Optional[LabelCovarianceMixtureModel] = None
        self.dim: Optional[int] = None
        self.weighted_c: Optional[np.ndarray] = None
        self.first: Optional[np.ndarray] = None
        self.second: Optional[np.ndarray] = None
        #: Set by finalize: what the M-step did, for the epoch statistics.
        self.step_statistics: Optional[Dict[str, float]] = None

    @property
    def num_labels(self) -> int:
        return self.num_clusters

    @property
    def _needs_first(self) -> bool:
        return self.update_means or self.covariance != "fixed"

    @property
    def _needs_second(self) -> bool:
        return self.covariance != "fixed"

    def _allocate(self, num_densities: int, dim: int) -> None:
        self.dim = dim
        self.weighted_c = np.zeros((self.num_clusters, num_densities), dtype=np.float64)
        self.first = (
            np.zeros((self.num_clusters, num_densities, dim), dtype=np.float64)
            if self._needs_first else None
        )
        self.second = (
            np.zeros((self.num_clusters, dim, dim), dtype=np.float64)
            if self._needs_second else None
        )

    def bind_model(self, model: ScoreModel) -> None:
        if not isinstance(model, LabelCovarianceMixtureModel):
            raise TypeError(
                f"{type(self).__name__} needs a LabelCovarianceMixtureModel, got "
                f"{type(model).__name__}"
            )
        if model.num_labels != self.num_clusters:
            raise ValueError(
                f"label count mismatch: accumulator has {self.num_clusters}, "
                f"model scores {model.num_labels}"
            )
        self.model = model
        if self.weighted_c is None:
            self._allocate(model.num_densities, model.dim)
        elif self.weighted_c.shape != model.mixtures.shape or self.dim != model.dim:
            raise ValueError(
                f"statistics shaped for {self.weighted_c.shape} x D={self.dim}, model is "
                f"{model.mixtures.shape} x D={model.dim}"
            )

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        if self.model is None:
            raise RuntimeError(
                "bind_model() must be called before observe(): the E-step is defined "
                "against the model this epoch recognized with"
            )
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
        T = len(features)
        gammas = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(gammas) != T:
            raise ValueError(f"frame count mismatch: {T} features vs {len(gammas)} gammas")

        joint = self.model.joint_posteriors(features, gammas)               # [T, L, K]
        self.weighted_c += joint.sum(axis=0)
        # Both moments as one GEMM each over the flattened (label, codeword) or
        # (label, dimension) axis, rather than a loop over labels.
        if self.first is not None:
            self.first += (joint.reshape(T, -1).T @ features).reshape(self.first.shape)
        if self.second is not None:
            weighted = (gammas[:, :, np.newaxis] * features[:, np.newaxis, :]).reshape(T, -1)
            self.second += (weighted.T @ features).reshape(self.second.shape)

    def merge(self, other: "LabelCovarianceAccumulator") -> "LabelCovarianceAccumulator":
        for name in ("num_clusters", "update_means", "covariance"):
            if getattr(self, name) != getattr(other, name):
                raise ValueError(
                    f"cannot merge accumulators that disagree on {name}: "
                    f"{getattr(self, name)!r} vs {getattr(other, name)!r}"
                )
        if other.weighted_c is None:
            return self
        if self.weighted_c is None:
            self._allocate(other.weighted_c.shape[1], other.dim)
        self.weighted_c += other.weighted_c
        if self.first is not None:
            self.first += other.first
        if self.second is not None:
            self.second += other.second
        return self

    def scale(self, factor: float) -> "LabelCovarianceAccumulator":
        """Multiply every statistic by ``factor``, in place - the decay half of
        the statistics-level EMA, as for :class:`MixtureGaussianAccumulator`."""
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        for name in ("weighted_c", "first", "second"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name) * factor)
        return self

    def finalize(self, previous: ScoreModel) -> LabelCovarianceMixtureModel:
        if self.weighted_c is None:
            raise RuntimeError("nothing accumulated; cannot finalize")
        if not isinstance(previous, LabelCovarianceMixtureModel):
            raise TypeError(
                f"{type(self).__name__} needs a LabelCovarianceMixtureModel to fall back "
                f"on, got {type(previous).__name__}"
            )
        if previous.mixtures.shape != self.weighted_c.shape:
            raise ValueError(
                f"mixture shape mismatch: accumulated {self.weighted_c.shape} vs "
                f"{previous.mixtures.shape}"
            )

        # The mixture rule of MixtureGaussianAccumulator.finalize, unchanged.
        label_mass = self.weighted_c.sum(axis=-1)
        alive = alive_mask(label_mass, self.min_mass)
        weighted = self.weighted_c + self.mixture_floor
        mixtures = weighted / if_alive_else(label_mass, self.min_mass)[:, np.newaxis]
        if self.mixture_floor:
            mixtures /= mixtures.sum(axis=-1, keepdims=True)
        mixtures[~alive] = previous.mixtures[~alive]

        statistics: Dict[str, float] = {}
        # A frozen parameter is handed through as the previous model's own
        # array, so save() can hardlink it rather than write a copy.
        centroids = previous.centroids
        if self.update_means:
            centroids = self._update_means(previous, label_mass, statistics)
        covs = previous.covs
        if self.covariance != "fixed":
            covs = self._update_covariances(previous, centroids, label_mass, statistics)
        self.step_statistics = statistics or None
        return type(previous)(
            centroids=centroids,
            covs=covs,
            mixtures=mixtures,
            device=getattr(previous, "device", None),
        )

    def _update_means(self, previous, label_mass, statistics) -> np.ndarray:
        """The first ECM block: means under the previous covariances."""
        centroids = np.array(previous.centroids, dtype=np.float64)
        mass = self.weighted_c.sum(axis=0)                                  # [K]
        alive = alive_mask(mass, self.min_mass)
        covs = np.asarray(previous.covs, dtype=np.float64)
        precisions = np.linalg.inv(covs)
        idx = np.flatnonzero(alive)
        if (covs == covs[:1]).all():
            # One covariance for every label: the precision cancels.
            updated = self.first.sum(axis=0)[idx] / mass[idx, np.newaxis]
        else:
            updated = np.empty((len(idx), centroids.shape[1]))
            for start in range(0, len(idx), self.MEAN_BLOCK):
                block = idx[start:start + self.MEAN_BLOCK]
                lhs = np.einsum("lk,lde->kde", self.weighted_c[:, block], precisions, optimize=True)
                rhs = np.einsum("lde,lke->kd", precisions, self.first[:, block], optimize=True)
                updated[start:start + len(block)] = np.linalg.solve(
                    lhs, rhs[..., np.newaxis]
                )[..., 0]
        # How far the codewords moved, in the units the model scores in: the
        # Mahalanobis length under the mass-weighted average precision.
        share = label_mass / label_mass.sum() if label_mass.sum() > 0 else np.full(len(covs), 1 / len(covs))
        shift = updated - centroids[idx]
        average_precision = np.tensordot(share, precisions, axes=1)
        statistics["means_shift_mean"] = float(
            np.sqrt(np.einsum("kd,de,ke->k", shift, average_precision, shift)).mean()
        ) if len(idx) else 0.0
        statistics["means_kept_previous"] = int((~alive).sum())
        centroids[idx] = updated
        return centroids

    def _update_covariances(self, previous, centroids, label_mass, statistics) -> np.ndarray:
        """The second ECM block: covariances about the (possibly new) means."""
        mu = np.asarray(centroids, dtype=np.float64)
        dim = mu.shape[1]
        cross = np.einsum("lkd,ke->lde", self.first, mu, optimize=True)     # sum_k F_lk mu_k^T
        outer = np.einsum("lk,kd,ke->lde", self.weighted_c, mu, mu, optimize=True)
        scatter = self.second - cross - cross.transpose(0, 2, 1) + outer      # n_l W_l
        scatter = (scatter + scatter.transpose(0, 2, 1)) / 2

        alive = alive_mask(label_mass, self.min_mass)
        if not alive.any():
            print("WARNING: no label has enough mass to re-estimate a covariance", flush=True)
            return previous.covs
        pooled = scatter[alive].sum(axis=0) / label_mass[alive].sum()
        relevance = self.covariance_relevance
        # Dead labels get a dummy denominator; their rows are replaced below.
        denominator = np.where(alive, label_mass + (0.0 if np.isinf(relevance) else relevance), 1.0)

        if self.covariance == "full":
            if np.isinf(relevance):
                covs = np.repeat(pooled[np.newaxis], len(scatter), axis=0)
            else:
                covs = (scatter + relevance * pooled) / denominator[:, np.newaxis, np.newaxis]
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(pooled)
            transform = eigenvectors.T / np.sqrt(eigenvalues)[:, np.newaxis]   # A W A^T = I
            if self.covariance == "semi_tied":
                per_label = scatter[alive] / label_mass[alive, np.newaxis, np.newaxis]
                # Resume from the previous epoch's transform, recovered from its
                # covariances, when that scores higher than the pooled
                # eigenbasis: the iterations then accumulate across epochs, and
                # since the previous covariances lie in the family the new ones
                # are chosen from, the update cannot lower the likelihood.
                candidates = [transform]
                resumed = common_diagonalizer(previous.covs)
                if resumed is not None:
                    candidates.append(resumed)
                scores = [semi_tied_objective(c, per_label, label_mass[alive]) for c in candidates]
                best = int(np.argmax(scores))
                transform, history = semi_tied_transform(
                    per_label, label_mass[alive], candidates[best], self.semi_tied_iterations
                )
                statistics["semi_tied_resumed"] = best
                statistics["semi_tied_objective_start"] = history[0]
                statistics["semi_tied_objective_end"] = history[-1]
            variances = np.einsum("id,lde,ie->li", transform, scatter, transform, optimize=True)
            pooled_variances = np.einsum("id,de,ie->i", transform, pooled, transform, optimize=True)
            if np.isinf(relevance):
                variances = np.repeat(pooled_variances[np.newaxis], len(scatter), axis=0)
            else:
                variances = (variances + relevance * pooled_variances) / denominator[:, np.newaxis]
            inverse = np.linalg.inv(transform)
            covs = np.einsum("di,li,ei->lde", inverse, variances, inverse, optimize=True)
        covs = (covs + covs.transpose(0, 2, 1)) / 2

        # The same rank guard as SoftGaussianAccumulator: a label whose estimate
        # cannot be inverted keeps its previous covariance instead of failing
        # the next model's Cholesky an epoch later.
        kept = alive.copy()
        eigenvalues = np.linalg.eigvalsh(covs[kept])
        singular = eigenvalues.min(axis=1) <= eigenvalues.max(axis=1) * dim * np.finfo(np.float64).eps
        if singular.any():
            starved = np.flatnonzero(kept)[singular]
            print(
                f"WARNING: {len(starved)} label(s) produced a singular covariance and keep "
                f"the previous epoch's; mass "
                f"{np.array2string(label_mass[starved], precision=1, threshold=12)}. "
                f"Raise covariance_relevance.",
                flush=True,
            )
            kept[starved] = False
        covs[~kept] = np.asarray(previous.covs, dtype=np.float64)[~kept]

        log_dets = np.linalg.slogdet(covs)[1]
        previous_log_dets = np.linalg.slogdet(np.asarray(previous.covs, dtype=np.float64))[1]
        pooled_log_det = np.linalg.slogdet(pooled)[1]
        # Per dimension, so they read as log variance ratios: -1 is a label whose
        # covariance is e times tighter than the pool's, on geometric average.
        statistics["cov_logdet_vs_pooled_min"] = float((log_dets - pooled_log_det).min() / dim)
        statistics["cov_logdet_vs_pooled_max"] = float((log_dets - pooled_log_det).max() / dim)
        statistics["cov_logdet_vs_previous_mean"] = float(
            (log_dets - previous_log_dets)[kept].mean() / dim
        ) if kept.any() else 0.0
        statistics["cov_kept_previous"] = int((~kept).sum())
        return covs

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "dim": self.dim,
            "update_means": self.update_means,
            "covariance": self.covariance,
            "weighted_c": self.weighted_c,
            "first": self.first,
            "second": self.second,
        }

    def load_state_dict(self, state: dict) -> "LabelCovarianceAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"label count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        if (bool(state["update_means"]), state["covariance"]) != (self.update_means, self.covariance):
            raise ValueError(
                f"state was accumulated for update_means={state['update_means']}, "
                f"covariance={state['covariance']!r}; this accumulator is configured "
                f"for {self.update_means}, {self.covariance!r}"
            )
        self.dim = state["dim"]
        for name in ("weighted_c", "first", "second"):
            value = state[name]
            setattr(self, name, None if value is None else np.asarray(value, dtype=np.float64))
        return self


class VectorQuantizedAccumulator:
    """
    Sufficient statistics for a :class:`.models.VectorQuantizedModel`: how often
    each label emitted each codeword.

        N_lc = sum_t gamma_tl [ q(x_t) = c ]

    and ``finalize`` normalizes each row. That is the entire update - one pass,
    no iteration inside it, and no arithmetic that depends on the feature
    dimension. Supervised counting and unsupervised training differ only in
    where ``gamma`` comes from: a reference alignment makes it one-hot and this
    reduces to :class:`...setup.vq_baseline.SupervisedVQTableJob`.

    **The codebook is never re-estimated.** ``quantize`` is taken from the bound
    model and the model's ``centroids`` are copied into the next one unchanged,
    so the partition is an input to the whole run. This is what keeps the two
    artifacts' different first axes (``centroids`` by codeword, ``table`` by
    label) from colliding in the dead-entry rule - only the label-indexed one is
    ever updated, the same split :class:`MixtureGaussianAccumulator` makes.

    **Merge is exact.** Quantization is a function of the epoch's model, which
    is fixed across every chunk, so the counts are plain sums and partitioning
    the corpus cannot change them - the property ``num_chunks`` being excluded
    from the job hash rests on.

    State is ``[L, C]`` and nothing else: 160 kB at 40 labels and 512
    codewords, against ~1 GB for the full-covariance mixture accumulator at the
    same codebook size. That is the practical reason this scales to a 32M-vector
    corpus where the mixture did not.

    :param num_clusters: the label count L, injected by the epoch job
    :param num_codewords: codebook size, if it has to be fixed before a model is
        bound; otherwise taken from the model
    :param table_floor: added to every count before normalizing. A zero entry
        scores ``+inf``, and with the codebook frozen a label that loses a
        codeword can never regain it, so zero is absorbing exactly as
        ``mixture_floor`` is for the mixture models. Unlike there, it is also
        what stops a frame from having no viable label at all - see
        :meth:`.models.VectorQuantizedModel.scores`.
    :param min_mass: evidence a label needs before its row is re-estimated;
        below it the previous table's row is kept.
    """

    def __init__(
        self,
        num_clusters: int,
        num_codewords: Optional[int] = None,
        table_floor: float = 0.0,
        min_mass: float = 0.0,
        **runtime_args,
    ):
        self.num_clusters = num_clusters
        self.num_codewords = num_codewords
        self.table_floor = table_floor
        self.min_mass = min_mass
        self.model: Optional[ScoreModel] = None
        self.counts = None
        if num_codewords is not None:
            self._allocate(num_codewords)

    @property
    def num_labels(self) -> int:
        return self.num_clusters

    def _allocate(self, num_codewords: int) -> None:
        if self.num_codewords is not None and self.num_codewords != num_codewords:
            raise ValueError(
                f"codebook size changed: {self.num_codewords} -> {num_codewords}"
            )
        self.num_codewords = num_codewords
        self.counts = np.zeros((self.num_clusters, num_codewords), dtype=np.float64)

    def bind_model(self, model: ScoreModel) -> None:
        if not hasattr(model, "quantize"):
            raise TypeError(
                f"{type(self).__name__} needs a model that can quantize a frame to a "
                f"codeword (a quantize() method), got {type(model).__name__}"
            )
        if model.num_clusters != self.num_clusters:
            raise ValueError(
                f"label count mismatch: accumulator has {self.num_clusters}, "
                f"{type(model).__name__} scores {model.num_clusters}"
            )
        self.model = model
        if self.counts is None:
            self._allocate(model.num_codewords)
        elif self.counts.shape[1] != model.num_codewords:
            raise ValueError(
                f"codebook size mismatch: accumulator holds {self.counts.shape[1]}, "
                f"model has {model.num_codewords}"
            )

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        if self.model is None:
            raise RuntimeError(
                "bind_model() must be called before observe(): the codeword a frame "
                "counts towards is defined by the model this epoch recognized with"
            )
        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError(f"features must be 2-D [T, D], got shape {features.shape}")
        gammas = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(gammas) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs {len(gammas)} posteriors"
            )
        codewords = self.model.quantize(features)
        # Scattered into a [C, L] buffer and transposed once, rather than into a
        # transposed view: add.at on a view is correct but subtle, and this is
        # one small allocation per sequence against a statistic that is read for
        # the rest of the epoch.
        contribution = np.zeros((self.num_codewords, self.num_clusters), dtype=np.float64)
        np.add.at(contribution, codewords, gammas)
        self.counts += contribution.T

    def merge(self, other: "VectorQuantizedAccumulator") -> "VectorQuantizedAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"label count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if other.counts is None:
            return self
        if self.counts is None:
            self._allocate(other.num_codewords)
        self.counts += other.counts
        return self

    def scale(self, factor: float) -> "VectorQuantizedAccumulator":
        """
        Multiply the accumulated statistics by ``factor``, in place.

        The decay half of the statistics-level EMA ``S <- a*S + (1-a)*N``;
        :meth:`merge` is the addition half, so that recursion needs no
        arithmetic of its own (see :mod:`.ema`). Every statistic here is a sum
        over frames, so scaling it is the stochastic-approximation step of
        online EM: the result is still a (fractional) count and ``finalize``
        normalizes it exactly as it does a plain epoch's.
        """
        if factor < 0.0:
            raise ValueError(f"scale factor must be non-negative, got {factor}")
        if self.counts is not None:
            self.counts = self.counts * factor
        return self

    def finalize(self, previous: ScoreModel) -> ScoreModel:
        if self.counts is None:
            raise RuntimeError("nothing accumulated; cannot finalize")
        arrays = previous.artifacts()
        missing = {"centroids", "table"} - set(arrays)
        if missing:
            raise TypeError(
                f"{type(self).__name__} needs a model carrying {sorted(missing)}, got "
                f"{type(previous).__name__} with {sorted(arrays)}"
            )
        if arrays["table"].shape != self.counts.shape:
            raise ValueError(
                f"table shape mismatch: accumulated {self.counts.shape} vs "
                f"{type(previous).__name__}'s {arrays['table'].shape}"
            )

        mass = self.counts.sum(axis=1)
        alive = alive_mask(mass, self.min_mass)
        floored = self.counts + self.table_floor
        table = floored / if_alive_else(floored.sum(axis=1), 0.0)[:, np.newaxis]
        # A label nothing aligned to keeps its previous row: an unnormalized or
        # zeroed row would violate the model's own invariant and leave the label
        # with no emission probability at all.
        table[~alive] = arrays["table"][~alive]
        return type(previous)(
            centroids=arrays["centroids"],
            table=table,
            device=getattr(previous, "device", None),
        )

    def state_dict(self) -> dict:
        return {
            "num_clusters": self.num_clusters,
            "num_codewords": self.num_codewords,
            "counts": self.counts,
            "table_floor": self.table_floor,
        }

    def load_state_dict(self, state: dict) -> "VectorQuantizedAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"label count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        num_codewords = state["num_codewords"]
        if num_codewords is not None and self.counts is None:
            self._allocate(int(num_codewords))
        self.counts = (
            None if state["counts"] is None else np.asarray(state["counts"], dtype=np.float64)
        )
        return self


class NullAccumulator:
    """
    Accumulates nothing, for a pass that recognizes in order to *observe* it
    rather than to update a model - see
    :class:`...setup.diagnostics.ClusteringDiagnosticsJob`.

    ``run_chunk`` requires an accumulator, and satisfying it with this instead
    of making the argument optional keeps the loop's contract single-shaped:
    every pass through the corpus scores, recognizes and accumulates, and a
    diagnostics pass is the degenerate case where accumulating is a no-op.
    ``finalize`` raises rather than returning the previous model unchanged: a
    caller that reaches it wanted a new model out of a pass that by
    construction cannot produce one.
    """

    def __init__(self, num_clusters: Optional[int] = None, **_kwargs):
        self.num_clusters = num_clusters

    def bind_model(self, model: ScoreModel) -> None:
        """No-op: this accumulator records nothing at all."""

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        pass

    def merge(self, other: "NullAccumulator") -> "NullAccumulator":
        return self

    def finalize(self, previous: ScoreModel) -> ScoreModel:
        raise NotImplementedError(
            "NullAccumulator records no statistics and cannot produce a model; "
            "it is for diagnostics passes, which have no reduce step"
        )

    def state_dict(self) -> dict:
        return {}

    def load_state_dict(self, state: dict) -> "NullAccumulator":
        return self


class GaussianAccumulator:
    """
    Per-cluster mean and covariance, via the Welford/M2 accumulators in
    :class:`PCAUpdater`; finalizes to a :class:`GaussianModel`.

    Only hard assignments are supported - ``StreamingPCA`` has no weighted
    update - so an n-best or full-sum recognizer needs either a weighted
    variant here or pairing with :class:`MeanAccumulator`.

    :param shrinkage: as on :class:`SoftGaussianAccumulator`, and applied to
        the same effect - see :func:`shrink_covariances`. Kept on both rather
        than on one, because which of the two runs is decided by the search,
        and the covariance question is the same either way.
    """

    def __init__(
        self,
        num_clusters: int,
        dim: Optional[int] = None,
        shrinkage: float = 0.0,
        **runtime_args,
    ):
        self.num_clusters = num_clusters
        self.dim = dim
        self.shrinkage = shrinkage
        self._updater = PCAUpdater(num_clusters)

    def bind_model(self, model: ScoreModel) -> None:
        """No-op: Welford's accumulators need only the frames and their labels."""

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None:
        labels = as_hard_labels(posteriors)
        features = np.asarray(features)
        if len(labels) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs {len(labels)} labels"
            )
        self._updater.update(features, labels)

    def merge(self, other: "GaussianAccumulator") -> "GaussianAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        self._updater.merge(other._updater)
        return self

    def finalize(self, previous: ScoreModel) -> GaussianModel:
        arrays = previous.artifacts()
        missing = {"centroids", "covs"} - set(arrays)
        if missing:
            raise TypeError(
                f"GaussianAccumulator needs a model carrying {sorted(missing)} to fall "
                f"back on, got {type(previous).__name__} with {sorted(arrays)}"
            )
        # PCAUpdater.get_model applies the dead-cluster rule itself: clusters
        # with too few samples to form a covariance raise NotEnoughSamplesError
        # and keep the previous mean/cov. Hence no keep_previous_where_dead()
        # here - it would be a redundant second pass over the same rule.
        model = self._updater.get_model(_PreviousArrays(arrays["centroids"], arrays["covs"]))
        means = model.means
        means = means.detach().cpu().numpy() if hasattr(means, "detach") else np.asarray(means)
        covs = np.asarray(model.covs)
        if self.shrinkage:
            # get_model() has already applied the dead-cluster rule, so a cluster
            # that kept the previous epoch's covariance would otherwise be pulled
            # towards this epoch's pool. Only the re-estimated ones are shrunk.
            counts = np.asarray(self._updater.state_dict()["n_samples"], dtype=np.float64)
            covs = shrink_covariances(covs, counts, self.shrinkage, counts > 0)
        return GaussianModel(means, covs, device=getattr(previous, "device", None))

    def state_dict(self) -> dict:
        state = self._updater.state_dict()
        state["num_clusters"] = self.num_clusters
        state["shrinkage"] = self.shrinkage
        return state

    def load_state_dict(self, state: dict) -> "GaussianAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self.shrinkage = float(state.get("shrinkage", self.shrinkage))
        self._updater.load_state_dict(state)
        return self
