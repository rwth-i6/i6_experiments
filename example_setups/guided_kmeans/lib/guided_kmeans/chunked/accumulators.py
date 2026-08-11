"""
Accumulators: the updating mechanism, in mergeable form.

Every accumulator here holds a *sufficient statistic* of the frames it has
seen, and ``merge`` is associative. That is what allows a clustering epoch to
be split across independent chunk tasks and reduced afterwards, and it is why
``num_chunks`` can be excluded from the job hash: the partition provably
cannot change the result (up to float summation order).
"""

from __future__ import annotations

__all__ = ["MeanAccumulator", "GaussianAccumulator", "keep_previous_where_dead"]

from dataclasses import dataclass
from typing import Optional, TypeVar

import numpy as np

from ..pca import PCAUpdater
from .interfaces import Posteriors, ScoreModel, as_hard_labels, as_responsibilities
from .models import ArtifactModel, EuclideanModel, GaussianModel

_ModelT = TypeVar("_ModelT", bound=ArtifactModel)


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
    updated: _ModelT, previous: ScoreModel, dead: np.ndarray
) -> _ModelT:
    """
    Restore the previous epoch's parameters for clusters that received no data.

    Without this they would come out as zeros and stop being reachable
    candidates in the next recognition pass - the same rule the single-process
    callback applies to centroids, here generalized over a model's artifacts so
    it holds for any parameter set.

    Relies on every artifact being indexed by cluster on its first axis, which
    is asserted rather than assumed: a model carrying a genuinely global
    parameter needs this rule extended, and should fail loudly here instead of
    having that parameter silently overwritten.
    """
    if not dead.any():
        return updated

    new_arrays = updated.artifacts()
    old_arrays = previous.artifacts()
    if set(new_arrays) != set(old_arrays):
        raise ValueError(
            f"cannot carry parameters over from {type(previous).__name__} to "
            f"{type(updated).__name__}: artifacts {sorted(old_arrays)} vs {sorted(new_arrays)}"
        )

    merged = {}
    for name, array in new_arrays.items():
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

    return type(updated).from_artifacts(merged, updated.meta())


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
        idx, weights = as_responsibilities(posteriors)
        if len(idx) != len(features):
            raise ValueError(
                f"frame count mismatch: {len(features)} features vs {len(idx)} labels"
            )
        # Build the [T, K] responsibility matrix and contract. For hard labels
        # this is exactly the one-hot matmul the single-process callback does
        # (`idx_matrix.T @ hidden_states`), down to the summation order, which
        # keeps the two pipelines bit-comparable; soft posteriors just put
        # non-binary weights in the same matrix.
        responsibilities = np.zeros((len(features), self.num_clusters), dtype=np.float64)
        np.add.at(responsibilities, (np.arange(len(features))[:, None], idx), weights)
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
        return keep_previous_where_dead(
            EuclideanModel(centroids), previous, self.counts == 0
        )

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
