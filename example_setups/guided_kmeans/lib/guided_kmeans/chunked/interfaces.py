"""
The protocols the chunked clustering loop is written against.

Each of the four is an injection point (see :mod:`.spec`); the loop in
:mod:`.runner` never names a concrete implementation.
"""

from __future__ import annotations

__all__ = [
    "Posteriors",
    "FeatureSource",
    "ScoreModel",
    "Recognizer",
    "Accumulator",
    "Probe",
    "as_hard_labels",
    "as_responsibilities",
]

from typing import Any, Callable, Dict, Iterator, List, Protocol, Tuple, Union, runtime_checkable

import numpy as np

#: What a recognizer hands back for one sequence, in one of two forms:
#:
#: * ``[T]`` integer array - one label per frame (Viterbi / forced alignment)
#: * ``([T, n] int, [T, n] float)`` - the n best labels per frame with their
#:   weights, which is how n-best lists and full-sum posteriors will arrive
#:
#: Accumulators consume this single type, so moving from Viterbi to n-best or
#: full-sum is a recognizer swap and touches no accumulator code.
Posteriors = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


def as_hard_labels(posteriors: Posteriors) -> np.ndarray:
    """
    ``[T]`` label array, for accumulators that cannot use soft assignments.
    Raises on genuinely soft input rather than silently taking the argmax.
    """
    if isinstance(posteriors, tuple):
        idx, weights = posteriors
        if idx.shape[1] != 1:
            raise NotImplementedError(
                "this accumulator requires hard assignments, got soft posteriors "
                f"with {idx.shape[1]} candidates per frame"
            )
        return idx[:, 0]
    return posteriors


def as_responsibilities(posteriors: Posteriors) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize either form to ``([T, n] idx, [T, n] weights)``. Hard labels
    become one candidate per frame with weight 1.0.
    """
    if isinstance(posteriors, tuple):
        return posteriors
    return posteriors[:, None], np.ones((len(posteriors), 1), dtype=np.float64)


@runtime_checkable
class FeatureSource(Protocol):
    """Iterates the sequences assigned to one chunk."""

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Yields ``(seq_tag, features [T, D])``, post-pooling."""
        ...

    def __len__(self) -> int:
        """Number of sequences in this chunk (for progress reporting)."""
        ...


@runtime_checkable
class ScoreModel(Protocol):
    """
    Maps encoder features to per-frame emission costs, one per cluster.

    ``artifacts`` is the whole persistence and parameter-carry-over contract:
    it is what gets written to the model directory, what a new model is rebuilt
    from, and what the dead-cluster rule operates on. Implementations normally
    inherit the mechanics from :class:`.models.ArtifactModel`; loading from
    disk is a module-level function, not part of this protocol.
    """

    num_clusters: int
    dim: int

    def scores(self, features: np.ndarray) -> np.ndarray:
        """``[T, D] -> [T, K]``, lower is better (a cost, not a likelihood)."""
        ...

    def artifacts(self) -> Dict[str, np.ndarray]:
        """Everything needed to reconstruct this model, as name -> array."""
        ...

    def save(self, directory: str) -> None: ...


@runtime_checkable
class Recognizer(Protocol):
    """
    Turns per-frame scores into per-frame labels, asynchronously.

    ``on_result`` receives ``(seq_tag, posteriors, traceback)``; the traceback
    is passed through untouched for the statistics counters and may be an
    empty list for recognizers that do not produce one.
    """

    def start(self, on_result: Callable[[str, Posteriors, List[Any]], None]) -> None: ...

    def submit(self, seq_tag: str, scores: np.ndarray) -> None: ...

    def drain(self) -> None:
        """Block until every submitted sequence has been passed to on_result."""
        ...

    def shutdown(self) -> None: ...


@runtime_checkable
class Probe(Protocol):
    """
    Read-only observer of one recognized sequence, for diagnostics.

    The fifth injection point, and the only one that may not influence the
    epoch's result: :func:`.runner.run_chunk` calls it after the accumulator
    has already seen the sequence, and never reads anything back. It exists
    because the statistics counters take a traceback alone - enough for corpus
    aggregates, not enough to relate a score to the sequence it came from or to
    the model that scored it.

    ``scores`` is the model's output *before* the recognizer applies its
    ``distance_scale``, i.e. the plain distance from each frame to each
    cluster; multiply by that scale to reconstruct what the search saw.
    """

    def observe(
        self,
        *,
        seq_tag: str,
        features: np.ndarray,
        scores: np.ndarray,
        posteriors: "Posteriors",
        traceback: List[Any],
    ) -> None: ...


@runtime_checkable
class Accumulator(Protocol):
    """
    Sufficient statistics for one clustering epoch.

    ``merge`` MUST be associative: the chunked pipeline treats the number of
    chunks as a scheduling parameter (it is excluded from the job hash), which
    is only sound if partitioning cannot change the result. Equivalently, the
    accumulator state has to be a sufficient statistic of the observations,
    never something order-dependent.
    """

    def observe(self, features: np.ndarray, posteriors: Posteriors) -> None: ...

    def merge(self, other: "Accumulator") -> "Accumulator": ...

    def finalize(self, previous: ScoreModel) -> ScoreModel:
        """
        Build the next epoch's model. ``previous`` supplies the fallback for
        clusters that received no data this epoch, so they stay viable
        candidates instead of collapsing.
        """
        ...

    def state_dict(self) -> dict: ...

    def load_state_dict(self, state: dict) -> "Accumulator": ...
