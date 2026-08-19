"""
The protocols the chunked clustering loop is written against.

Each of the four is an injection point (see :mod:`.spec`); the loop in
:mod:`.runner` never names a concrete implementation.
"""

from __future__ import annotations

__all__ = [
    "Posteriors",
    "RecognitionResult",
    "FeatureSource",
    "ScoreModel",
    "Recognizer",
    "Accumulator",
    "Probe",
    "as_hard_labels",
    "as_responsibilities",
    "as_dense_responsibilities",
]

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    Union,
    runtime_checkable,
)

import numpy as np

#: What a recognizer hands back for one sequence, in one of three forms:
#:
#: * ``[T]`` integer array - one label per frame (Viterbi / forced alignment)
#: * ``([T, n] int, [T, n] float)`` - the n best labels per frame with their
#:   weights, which is how an n-best recognizer reports
#: * ``[T, K]`` float array - a dense posterior (gamma) matrix over all
#:   clusters, which is how forward-backward search reports
#:
#: The forms are told apart by structure: a tuple is the sparse form, a 1-D
#: array is hard labels, a 2-D array is dense gammas. Accumulators consume the
#: single type through the converters below, so swapping Viterbi for n-best or
#: full-sum is a recognizer change and touches no accumulator arithmetic.
Posteriors = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]


def _is_dense(posteriors: Posteriors) -> bool:
    return not isinstance(posteriors, tuple) and np.asarray(posteriors).ndim == 2


def as_hard_labels(posteriors: Posteriors) -> np.ndarray:
    """
    ``[T]`` label array, for accumulators that cannot use soft assignments.
    Raises on genuinely soft input rather than silently taking the argmax.
    """
    if _is_dense(posteriors):
        raise NotImplementedError(
            "this accumulator requires hard assignments, got a dense "
            f"{np.asarray(posteriors).shape} posterior matrix; pair a "
            "forward-backward recognizer with an accumulator that takes soft "
            "assignments (MeanAccumulator, SoftGaussianAccumulator)"
        )
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
    Normalize any form to ``([T, n] idx, [T, n] weights)``. Hard labels become
    one candidate per frame with weight 1.0; a dense ``[T, K]`` matrix becomes
    K candidates per frame.

    Prefer :func:`as_dense_responsibilities` for accumulators that contract
    against all clusters anyway - it avoids materializing an index array the
    dense form does not need.
    """
    if isinstance(posteriors, tuple):
        return posteriors
    posteriors = np.asarray(posteriors)
    if posteriors.ndim == 2:
        num_frames, num_clusters = posteriors.shape
        idx = np.broadcast_to(np.arange(num_clusters), (num_frames, num_clusters))
        return idx, posteriors
    return posteriors[:, None], np.ones((len(posteriors), 1), dtype=np.float64)


def as_dense_responsibilities(posteriors: Posteriors, num_clusters: int) -> np.ndarray:
    """
    ``[T, K]`` responsibility matrix, whichever form the recognizer used.

    This is the single place the three forms are told apart. Hard labels give a
    one-hot matrix, so an accumulator written against this sees exactly the
    arithmetic it saw before dense posteriors existed.
    """
    if isinstance(posteriors, tuple):
        idx, weights = posteriors
        dense = np.zeros((len(idx), num_clusters), dtype=np.float64)
        np.add.at(dense, (np.arange(len(idx))[:, None], idx), weights)
        return dense

    posteriors = np.asarray(posteriors)
    if posteriors.ndim == 2:
        if posteriors.shape[1] != num_clusters:
            raise ValueError(
                f"dense posteriors have {posteriors.shape[1]} columns, "
                f"expected {num_clusters}"
            )
        return posteriors.astype(np.float64, copy=False)

    dense = np.zeros((len(posteriors), num_clusters), dtype=np.float64)
    dense[np.arange(len(posteriors)), posteriors] = 1.0
    return dense


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


@dataclass
class RecognitionResult:
    """
    Everything a recognizer reports about one sequence.

    A single object rather than a widening argument list, so a recognizer with
    something extra to say has a named field to say it in. The alternative -
    threading per-sequence metadata through the ``traceback`` slot - worked for
    one value but made the slot mean different things for different
    recognizers, and forced consumers to identify the producer by inspecting
    the items inside it.

    :param seq_tag: the sequence this result belongs to
    :param posteriors: per-frame assignments, in any of the :data:`Posteriors`
        forms
    :param traceback: the search's own path representation, passed through
        untouched for the statistics counters. Empty for recognizers that
        produce no discrete path, such as forward-backward.
    :param sequence_score: the search's score for the whole sequence, in the
        recognizer's natural units - log-likelihood for forward-backward. Left
        as ``None`` by recognizers whose per-sequence score is already
        recoverable from ``traceback`` (Viterbi puts it in the final item).
    """

    seq_tag: str
    posteriors: "Posteriors"
    traceback: List[Any] = field(default_factory=list)
    sequence_score: Optional[float] = None


@runtime_checkable
class Recognizer(Protocol):
    """
    Turns per-frame scores into per-frame labels, asynchronously.

    ``on_result`` receives one :class:`RecognitionResult` per sequence.
    """

    def start(self, on_result: Callable[["RecognitionResult"], None]) -> None: ...

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
