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
    "MeanAccumulator",
    "GaussianAccumulator",
    "NullAccumulator",
    "SoftGaussianAccumulator",
    "keep_previous_where_dead",
]

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np

from ..pca import PCAUpdater
from .interfaces import (
    Posteriors,
    ScoreModel,
    as_dense_responsibilities,
    as_hard_labels,
)
from .models import EuclideanModel, GaussianModel


@dataclass
class _PreviousArrays:
    """
    Minimal stand-in exposing the ``.means``/``.covs`` attributes
    :meth:`PCAUpdater.get_model` reads off the previous model, so this does not
    have to reach into a GaussianModel's wrapped GaussianModelNumpy.
    """

    means: np.ndarray
    covs: np.ndarray


def keep_previous_where_dead(
    updated: Mapping[str, np.ndarray], previous: ScoreModel, dead: np.ndarray
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
    if not dead.any():
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


class MeanAccumulator:
    """
    Per-cluster frame counts and feature sums; finalizes to centroids.

    Equivalent to what ``RunningAverageUpdater`` accumulates in the
    single-process callback, but keeping raw sums instead of a running mean -
    same result, one division instead of one per sequence.

    :param num_clusters: size of the label inventory
    :param dim: feature dimension; inferred from the first observation if None
    """

    def __init__(self, num_clusters: int, dim: Optional[int] = None):
        self.num_clusters = num_clusters
        self.counts = np.zeros(num_clusters, dtype=np.float64)
        self.sums = None if dim is None else np.zeros((num_clusters, dim), dtype=np.float64)

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
    """

    def __init__(self, num_clusters: int, dim: Optional[int] = None):
        self.num_clusters = num_clusters
        self.dim = dim
        self.counts = np.zeros(num_clusters, dtype=np.float64)
        if dim is not None:
            self.weighted_sums = np.zeros((num_clusters, dim), dtype=np.float64)
            self.weighted_sq = np.zeros((num_clusters, dim, dim), dtype=np.float64)
        else:
            self.weighted_sums = None
            self.weighted_sq = None

    def _ensure(self, dim: int) -> None:
        if self.weighted_sums is None:
            self.dim = dim
            self.weighted_sums = np.zeros((self.num_clusters, dim), dtype=np.float64)
            self.weighted_sq = np.zeros((self.num_clusters, dim, dim), dtype=np.float64)
        elif self.dim != dim:
            raise ValueError(f"feature dim changed: {self.dim} -> {dim}")

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
        if isinstance(posteriors, tuple) or np.asarray(posteriors).ndim != 2:
            raise ValueError(
                "SoftGaussianAccumulator requires dense 2-D gamma posteriors [T, K]; "
                f"got {type(posteriors).__name__} with ndim={getattr(posteriors, 'ndim', '?')}. "
                "Use GaussianAccumulator for hard-label input."
            )
        gammas = as_dense_responsibilities(posteriors, self.num_clusters)
        if len(gammas) != T:
            raise ValueError(f"frame count mismatch: {T} features vs {len(gammas)} gammas")

        self.counts += gammas.sum(0)
        self.weighted_sums += gammas.T @ features  # [K, D]

        # Weighted outer-product sum: S2[k] += Σ_t γ_tk · x_t x_t^T
        # Loop over K to keep the per-call memory footprint at O(T·D) rather
        # than O(T·K·D); at K=40 the loop overhead is negligible.
        for k in range(self.num_clusters):
            wf = features * gammas[:, k : k + 1]  # [T, D], broadcast weight
            self.weighted_sq[k] += wf.T @ features  # [D, T] @ [T, D] = [D, D]

    def merge(self, other: "SoftGaussianAccumulator") -> "SoftGaussianAccumulator":
        if self.num_clusters != other.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {self.num_clusters} vs {other.num_clusters}"
            )
        if other.weighted_sums is None:
            return self
        self._ensure(other.dim)
        self.counts += other.counts
        self.weighted_sums += other.weighted_sums
        self.weighted_sq += other.weighted_sq
        return self

    def finalize(self, previous: ScoreModel) -> GaussianModel:
        if self.weighted_sums is None:
            raise RuntimeError("nothing accumulated; cannot finalize")

        missing = {"centroids", "covs"} - set(previous.artifacts())
        if missing:
            raise TypeError(
                f"SoftGaussianAccumulator needs a model with {sorted(missing)}, "
                f"got {type(previous).__name__} with {sorted(previous.artifacts())}"
            )

        alive = self.counts > 0
        n = np.where(alive, self.counts, 1.0)  # avoid division by zero; dead rows replaced below

        means = self.weighted_sums / n[:, np.newaxis]  # [K, D]
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

        merged = keep_previous_where_dead(
            {"centroids": means, "covs": covs}, previous, ~alive
        )
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
        }

    def load_state_dict(self, state: dict) -> "SoftGaussianAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self.dim = state["dim"]
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
    """

    def __init__(self, num_clusters: int, dim: Optional[int] = None):
        self.num_clusters = num_clusters
        self.dim = dim
        self._updater = PCAUpdater(num_clusters)

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
        return GaussianModel(means, np.asarray(model.covs), device=getattr(previous, "device", None))

    def state_dict(self) -> dict:
        state = self._updater.state_dict()
        state["num_clusters"] = self.num_clusters
        return state

    def load_state_dict(self, state: dict) -> "GaussianAccumulator":
        if int(state["num_clusters"]) != self.num_clusters:
            raise ValueError(
                f"cluster count mismatch: {state['num_clusters']} vs {self.num_clusters}"
            )
        self._updater.load_state_dict(state)
        return self
