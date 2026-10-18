"""
Transition models: per-frame transition scores for the forward-backward search.

The ``backoff_fb`` op adds a ``[T, 3]`` array of log-scores per sequence to the
static transition weights of its graph (SPEC.md section 14 in the op's
checkout). Row ``t`` weighs the transition *leaving* frame ``t``, and the three
columns are what that transition can be:

* column 0, **loop**: stay in the current HMM state
* column 1, **forward**: step to the next state of the same phoneme
* column 2, **exit**: leave the phoneme, through the LM, to the next one; on
  the last frame this is the final transition

Only a phoneme's last state exits, so the exit column scores phoneme
*boundaries* and nothing inside a phoneme; a boundary after frame ``t`` is an
exit out of frame ``t``. The scores are shared by every phoneme and every
state, -inf forbids a transition on that frame, and they need not be
normalized.

A :class:`.interfaces.TransitionModel` produces them, and
:func:`.runner.run_chunk` hands them to a recognizer that takes them (so far
only :class:`.recognizers.BackoffFBRecognizer`). The models here are fixed. A
trainable one, re-estimated each epoch from the transition posteriors, has its
interfaces in place (:class:`.interfaces.TrainableTransitionModel`,
:class:`.interfaces.TransitionAccumulator`) but cannot run yet: the op does not
compute those posteriors before its milestone T3.
"""

from __future__ import annotations

__all__ = [
    "LOOP",
    "FORWARD",
    "EXIT",
    "segment_ends",
    "jitter_boundaries",
    "boundary_probability",
    "boundary_transitions",
    "OracleBoundaryTransitions",
]

import zlib
from typing import Optional, Tuple

import numpy as np

#: Column indices of a transition score array.
LOOP, FORWARD, EXIT = 0, 1, 2


def segment_ends(labels: np.ndarray) -> np.ndarray:
    """``[T]`` bool: whether frame ``t`` is the last frame of a run of equal
    labels. The last frame of the sequence always is."""
    labels = np.asarray(labels).reshape(-1)
    return np.r_[labels[1:] != labels[:-1], True]


def jitter_boundaries(ends: np.ndarray, jitter: float, rng: np.random.Generator) -> np.ndarray:
    """Move every boundary but the last by ``round(jitter * N(0, 1))`` frames.

    ``ends`` is the ``[T]`` bool mask of :func:`segment_ends`, and so is the
    result. The number of segments stays the same and every segment keeps at
    least one frame: boundaries are placed left to right, each clipped to lie
    after the one before it and to leave a frame for each one still to come.
    The last frame stays a boundary, because the sequence ends there.

    A shift moves the boundary between two segments, so the frames it crosses
    change segment. It never changes which phoneme a segment is, and it adds or
    drops no boundary.
    """
    ends = np.asarray(ends, dtype=bool)
    T = len(ends)
    inner = np.flatnonzero(ends[:-1])
    K = len(inner)
    shifted = inner + np.rint(jitter * rng.standard_normal(K)).astype(np.int64)
    out = np.zeros(T, dtype=bool)
    out[-1] = True
    previous = -1
    for k in range(K):
        # Room for the K - 1 - k boundaries still to come, each with a frame of
        # its own, and for the last segment: at most T - 2 - (K - 1 - k).
        previous = int(min(max(shifted[k], previous + 1), T - 2 - (K - 1 - k)))
        out[previous] = True
    return out


def boundary_probability(
    ends: np.ndarray, boundary_error: float = 0.0, blur: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """``(p, 1 - p)``: the probability of a boundary after each frame, and its
    complement, computed separately so that neither loses digits to
    cancellation.

    ``p_t = boundary_error + (1 - 2 boundary_error) k(d_t)``, where ``d_t`` is
    the distance from frame ``t`` to the nearest boundary in ``ends`` and ``k``
    is 1 at distance 0 and 0 elsewhere, or, with ``blur > 0``, the Gaussian
    ``exp(-d^2 / (2 blur^2))``. ``p`` therefore peaks at ``1 - boundary_error``
    on a boundary and falls to ``boundary_error`` far from one.
    """
    if not 0.0 <= boundary_error < 0.5:
        raise ValueError(f"boundary_error must be in [0, 0.5), got {boundary_error}")
    if blur < 0.0:
        raise ValueError(f"blur must be >= 0, got {blur}")
    if blur > 0.0 and boundary_error == 0.0:
        raise ValueError(
            "blur needs boundary_error > 0: at 0 the boundary frame keeps p = 1, "
            "an exit forced there as before, and the blur would only add "
            "optional extra boundaries next to it")
    ends = np.asarray(ends, dtype=bool)
    if ends.size == 0:
        raise ValueError("cannot score an empty sequence")
    if blur == 0.0:
        k, not_k = ends.astype(np.float64), (~ends).astype(np.float64)
    else:
        positions = np.flatnonzero(ends)
        t = np.arange(len(ends))
        right = np.searchsorted(positions, t).clip(max=len(positions) - 1)
        left = (right - 1).clip(min=0)
        d = np.minimum(np.abs(positions[right] - t), np.abs(positions[left] - t))
        z = -(d.astype(np.float64) ** 2) / (2.0 * blur ** 2)
        k, not_k = np.exp(z), -np.expm1(z)
    scale = 1.0 - 2.0 * boundary_error
    return boundary_error + scale * k, boundary_error + scale * not_k


def boundary_transitions(
    labels: np.ndarray, boundary_error: float = 0.0, blur: float = 0.0,
    ends: Optional[np.ndarray] = None,
) -> np.ndarray:
    """``[T, 3]`` transition scores that place phoneme boundaries where the
    labels change.

    This is the boundary model of the op's SPEC.md 14.1 with the boundary
    probability ``p_t`` of :func:`boundary_probability`:

        theta_t = (log(1 - p_t), log(1 - p_t), log p_t)

    Read as a likelihood, it treats the alignment's boundaries ``o_t`` as a
    noisy observation of the path's boundaries ``b_t`` (``b_t = 1`` when the
    path exits after frame ``t``): ``P(o_t | b_t) = 1 - boundary_error`` when
    they agree, ``boundary_error`` when not, independently per frame. Every
    disagreement then costs ``lambda = log((1 - eps) / eps)`` nats against the
    alignment. Dropping or inserting a boundary costs one lambda, and moving
    one costs two, by however many frames it moves.

    **boundary_error = 0** (the default) gives hard boundaries: an exit is
    forbidden inside a segment and forced at its last frame, so each segment
    holds exactly one phoneme. With one HMM state per phoneme that is exactly
    the segment-level search of :class:`.recognizers.OracleSegmentFBRecognizer`
    (SPEC.md 14.1, consequence 4). With S states a segment shorter than S
    frames has no path, so the whole sequence comes back with log Z = -inf.

    **boundary_error -> 0.5** makes the scores the same on every frame, which
    leaves the posteriors to the graph's own transition weights (consequence
    1): a free segmentation.

    **blur > 0** widens where a boundary is plausible: ``p`` keeps its peak of
    ``1 - boundary_error`` on the alignment's boundary and falls off as a
    Gaussian of width ``blur`` frames. Moving a boundary by ``d`` frames then
    costs ``logit p_0 - logit p_d`` instead of two lambdas. Per-frame scores
    cannot say "exactly one boundary somewhere near here": a frame whose ``p``
    exceeds 1/2 *favours* a boundary of its own, so within about ``1.18 blur``
    frames of an alignment boundary an inserted second boundary is rewarded by
    these scores, and only the LM's price for one more phoneme and the
    acoustics hold it back. Narrow blurs (``blur <= 1``) barely reach that
    region. :func:`jitter_boundaries` is the counterpart that keeps the count:
    a boundary moved at random, exactly one per alignment boundary.

    :param labels: ``[T]`` per-frame labels; a boundary is where they change.
        Two realizations of one phoneme with nothing between them form one run
        and therefore one segment.
    :param boundary_error: the probability of a boundary error per frame, in
        ``[0, 0.5)``
    :param blur: Gaussian width in frames; needs ``boundary_error > 0``
    :param ends: use these boundaries instead of the labels' own, e.g. after
        :func:`jitter_boundaries`
    """
    if ends is None:
        ends = segment_ends(labels)
    p, not_p = boundary_probability(ends, boundary_error, blur)
    with np.errstate(divide="ignore"):
        stay, leave = np.log(not_p), np.log(p)
    return np.stack([stay, stay, leave], axis=1)


class OracleBoundaryTransitions:
    """
    Phoneme boundaries taken from a reference alignment, optionally degraded.

    Fixed, not trained, and the first transition model: with every knob at 0
    and one HMM state it reproduces
    :class:`.recognizers.OracleSegmentFBRecognizer` frame for frame, which is
    the check that the per-frame route is wired correctly. Unlike that
    recognizer it works at any number of states, tied or untied, and it can
    make the oracle worse in three ways, which compose:

    * ``jitter``: move each boundary by a random number of frames
      (:func:`jitter_boundaries`), drawn once per sequence from ``seed`` and
      the sequence tag. The same wrong segmentation is used in every epoch and
      in every chunk, like a worse aligner.
    * ``boundary_error``: soft boundaries, a per-frame error rate
      (:func:`boundary_transitions`).
    * ``blur``: a Gaussian-widened boundary (also there).

    With ``zero_topology=True`` in the search these scores are the whole
    transition model, and log Z at S = 1 with hard boundaries equals the
    segment-level one. With a loop probability strictly between 0 and 1 they
    are evidence on top of that duration prior instead: at hard boundaries the
    prior adds a constant per segment and the posteriors are the same, and as
    ``boundary_error`` approaches 0.5 the prior is all that is left. A loop
    probability of 0 forbids the loops a segment needs.

    :param frame_labels: pickle of ``{seq_tag: [T] int}``, one label per frame
        and aligned 1:1 with the features. This is the ``out_labels`` of
        ``SegmentedFeaturesFromAlignmentJob(pooling="none")``, the file
        :class:`.recognizers.OracleSegmentFBRecognizer` reads.
    :param boundary_error: see :func:`boundary_transitions`
    :param jitter: standard deviation in frames of each boundary's shift
    :param blur: see :func:`boundary_transitions`
    :param seed: for the jitter; with the sequence tag it fixes the draw, so it
        does not depend on chunking, process or epoch
    """

    trainable = False

    def __init__(self, frame_labels: str, boundary_error: float = 0.0,
                 jitter: float = 0.0, blur: float = 0.0, seed: int = 0):
        # Validated here, so that a bad value fails when the spec is built and
        # not in the first chunk task.
        boundary_probability(np.ones(1, dtype=bool), boundary_error, blur)
        if jitter < 0.0:
            raise ValueError(f"jitter must be >= 0, got {jitter}")
        self.frame_labels = frame_labels
        self.boundary_error = boundary_error
        self.jitter = jitter
        self.blur = blur
        self.seed = seed
        self._labels: Optional[dict] = None

    def _load(self) -> dict:
        # Lazily, so building the spec costs nothing until a chunk runs.
        if self._labels is None:
            import pickle
            import time

            t0 = time.perf_counter()
            with open(str(self.frame_labels), "rb") as f:
                self._labels = pickle.load(f)
            print(f"[TIMING] OracleBoundaryTransitions: {len(self._labels)} frame "
                  f"alignments loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        return self._labels

    def boundaries(self, seq_tag: str, num_frames: int) -> np.ndarray:
        """``[T]`` bool: the boundaries this model scores, after the jitter."""
        labels = self._load().get(seq_tag)
        if labels is None:
            raise KeyError(
                f"no frame alignment for {seq_tag!r} in {self.frame_labels}. "
                f"The alignment and the features have to come from the same "
                f"SegmentedFeaturesFromAlignmentJob.")
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != num_frames:
            raise ValueError(
                f"{seq_tag}: the alignment has {len(labels)} frames but the "
                f"features have {num_frames}. These features are probably the "
                f"pooled ones; the oracle needs pooling='none'.")
        ends = segment_ends(labels)
        if self.jitter > 0.0:
            # crc32 rather than hash(): Python salts str hashes per process.
            rng = np.random.default_rng([self.seed, zlib.crc32(seq_tag.encode())])
            ends = jitter_boundaries(ends, self.jitter, rng)
        return ends

    def transition_scores(self, seq_tag: str, features: np.ndarray) -> np.ndarray:
        ends = self.boundaries(seq_tag, len(features))
        return boundary_transitions(None, self.boundary_error, self.blur, ends=ends)
