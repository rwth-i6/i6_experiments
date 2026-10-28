"""
The clustering loop itself, free of sisyphus so it can be run and debugged
standalone (see ``run_chunk`` / ``reduce_chunks``).

Two differences to the single-process callback, both deliberate:

* Recognition and accumulation happen in **one** pass over the data. The old
  pipeline used separate RECOGNITION and CLUSTERING phases with a traceback
  database in between, but recognition uses centroids held fixed for the whole
  epoch and accumulation is order independent, so fusing them is
  mathematically identical - and removes the database, the phase state machine
  and the MultiEpochDataset entirely. (It saves little time: the clustering
  pass was measured at 0.4% of an epoch.)
* An epoch is split across chunks that are reduced afterwards, rather than
  streamed through one process.
"""

from __future__ import annotations

__all__ = [
    "ChunkResult",
    "MergedChunks",
    "run_chunk",
    "merge_chunks",
    "reduce_chunks",
    "save_chunk",
    "load_chunk",
]

import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..statistics import CounterProtocol
from .interfaces import (
    Accumulator,
    FeatureSource,
    Probe,
    Recognizer,
    RecognitionResult,
    ScoreModel,
    TransitionAccumulator,
    TransitionModel,
)
from .stats import merge_counters

#: Seconds between run_chunk's ``[PROGRESS]`` lines. Unlike the progress bar
#: they replace, they are flushed, so a running chunk's log shows how far it
#: has got. ``scripts/chunk_watchdog.py`` in the experiment root reads them to
#: tell a chunk on a degraded card from a slow but healthy one, so the line's
#: format is that script's interface (``PROGRESS_RE`` there): keep the two in
#: step.
PROGRESS_INTERVAL_S = 300


@dataclass
class ChunkResult:
    """What one chunk task contributes to the epoch."""

    accumulator_state: dict
    counter: Optional[CounterProtocol]
    num_seqs: int
    num_frames: int
    num_recognized: int
    #: ``{seq_tag: hypothesis}`` if run_chunk was given a ``transcribe``, else
    #: None. Small next to the accumulator state (~600 B per sequence against
    #: up to 84 MB of full-covariance statistics), so it travels in the same
    #: pickle rather than in a file of its own.
    hypotheses: Optional[Dict[str, str]] = None
    #: The :class:`.topology.TopologyAccumulator` state if run_chunk was given
    #: one, else None. A pickle written before this field existed reads as
    #: None too, through the class default.
    topology_state: Optional[dict] = None


def run_chunk(
    *,
    features: FeatureSource,
    model: ScoreModel,
    recognizer: Recognizer,
    accumulator: Accumulator,
    counter: Optional[CounterProtocol] = None,
    transcribe: Optional[Callable[[List[Any]], str]] = None,
    transcribe_posteriors: Optional[Callable[[np.ndarray], str]] = None,
    probe: Optional[Probe] = None,
    transitions: Optional[TransitionModel] = None,
    transition_accumulator: Optional[TransitionAccumulator] = None,
    topology_accumulator: Optional[Any] = None,
    verbosity: int = 1,
) -> ChunkResult:
    """
    Score, recognize and accumulate one chunk of the corpus.

    The accumulator is handed the model once via ``bind_model`` before the
    first sequence; the model is fixed for the whole epoch, so this does not
    make accumulation order-dependent.

    Recognition is asynchronous, so features are parked in ``pending`` until
    their result arrives. That dict is bounded by the recognizer's in-flight
    limit (``ParallelSegmentRecognizer`` drains its oldest task once more than
    ``max_pending_tasks`` are outstanding), not by the size of the chunk.

    :param transcribe: turns a traceback into a hypothesis string; when given,
        the hypothesis of every recognized sequence is kept in the result. This
        is a separate hook rather than another statistics counter because a
        counter only ever sees the traceback, not the sequence tag - and it
        takes the lemma-exclusion policy from the caller, keeping the loop
        itself free of any notion of what a label means.
    :param transcribe_posteriors: the same for a search that reports label
        posteriors instead of a path (a forward-backward recognizer, recognized
        by its ``sequence_score``). Used in place of ``transcribe`` for those
        results; without it they get ``transcribe``'s empty hypothesis, as
        before this existed.
    :param probe: optional diagnostics observer (:class:`.interfaces.Probe`).
        Given one, the ``[T, K]`` score matrix is parked alongside the features
        rather than handed to the recognizer and dropped, so the probe can
        relate the alignment back to the model that produced it. That is the
        only cost of this hook and it is paid only when it is used: the parked
        matrices are bounded by the same in-flight limit as ``pending`` (32
        sequences at the pipeline's default worker count, ~15 MB at K=40).
        The probe never feeds anything back into the loop - what it records
        cannot change the epoch's result.
    :param transitions: per-frame transition scores
        (:class:`.interfaces.TransitionModel`), computed from each sequence's
        features and submitted to the recognizer alongside its scores. The
        recognizer has to take them (``accepts_transitions``).
    :param transition_accumulator: statistics for re-estimating a trainable
        transition model from the transition posteriors. Not implemented: the
        op does not compute those posteriors before its milestone T3, so this
        raises rather than accumulating nothing.
    :param topology_accumulator: a :class:`.topology.TopologyAccumulator`,
        which observes every sequence's expected transition counts
        (:attr:`.interfaces.RecognitionResult.transition_counts`) so that the
        reduce step can re-estimate the loop probabilities. The recognizer has
        to report them (``BackoffFBRecognizer(transition_counts=True)``).
    """
    if transition_accumulator is not None:
        raise NotImplementedError(
            "re-estimating a transition model needs the transition posteriors "
            "(the gradient of log Z with respect to the transition scores), "
            "which the backoff_fb op computes from milestone T3 on (SPEC.md "
            "14.8); until then a transition model can only be fixed")
    if transitions is not None and not getattr(recognizer, "accepts_transitions", False):
        raise ValueError(
            f"{type(recognizer).__name__} does not take per-frame transition "
            f"scores; only the backoff_fb recognizer does")
    if topology_accumulator is not None and not getattr(recognizer, "transition_counts", False):
        raise ValueError(
            f"{type(recognizer).__name__} does not report transition counts, which "
            f"the topology accumulator needs; only BackoffFBRecognizer("
            f"transition_counts=True) does")
    pending: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    hypotheses: Optional[Dict[str, str]] = {} if transcribe is not None else None
    stats = {"recognized": 0, "frames": 0}
    progress = {"total": 0, "started": 0.0, "last": 0.0}

    def report_progress(force: bool = False) -> None:
        now = time.time()
        if not force and now - progress["last"] < PROGRESS_INTERVAL_S:
            return
        progress["last"] = now
        print(
            f"[PROGRESS] {time.strftime('%Y-%m-%dT%H:%M:%S%z', time.localtime(now))} "
            f"recognized {stats['recognized']}/{progress['total']} seqs "
            f"in {now - progress['started']:.0f} s",
            flush=True,
        )

    def on_result(result: RecognitionResult) -> None:
        seq_tag, posteriors, traceback = result.seq_tag, result.posteriors, result.traceback
        try:
            seq_features, seq_scores = pending.pop(seq_tag)
        except KeyError:
            raise KeyError(
                f"recognition result for unknown sequence {seq_tag!r}"
            ) from None
        if len(posteriors) != len(seq_features):
            raise ValueError(
                f"{seq_tag}: recognizer returned {len(posteriors)} labels for "
                f"{len(seq_features)} frames"
            )
        accumulator.observe(seq_features, posteriors)
        if topology_accumulator is not None:
            if result.transition_counts is None:
                raise ValueError(f"{seq_tag}: the recognizer reported no transition counts")
            topology_accumulator.observe(result.transition_counts)
        if counter is not None:
            # Which reader applies follows from what the recognizer reported,
            # not from inspecting the items inside the traceback: a search that
            # produces no discrete path reports a sequence_score instead.
            if result.sequence_score is not None and hasattr(counter, "read_gammas"):
                counter.read_gammas(posteriors, result.sequence_score)
            elif traceback:
                counter.read(traceback)
        if transcribe is not None and hypotheses is not None:
            # Unconditionally, empty traceback included: a sequence the search
            # emitted nothing for is a legitimate empty hypothesis and scoring
            # has to see it as a deletion, not as a missing segment. A
            # pathless search leaves traceback empty, which yields exactly that.
            if transcribe_posteriors is not None and result.sequence_score is not None and not traceback:
                hypotheses[seq_tag] = transcribe_posteriors(posteriors)
            else:
                hypotheses[seq_tag] = transcribe(traceback)
        if probe is not None:
            assert seq_scores is not None
            probe.observe(
                seq_tag=seq_tag,
                features=seq_features,
                scores=seq_scores,
                posteriors=posteriors,
                traceback=traceback,
            )
        stats["recognized"] += 1
        report_progress()

    # Before the first observe(), and before any result can come back: an
    # accumulator whose E-step is defined against the current parameters (a
    # mixture model's, say) needs the epoch's model, and asking for it here
    # keeps it out of every other accumulator's constructor - and out of the
    # zero-argument factory reduce_chunks() builds empty ones with.
    accumulator.bind_model(model)

    recognizer.start(on_result)
    progress["total"] = len(features)
    started = progress["started"] = time.time()
    # After start(), which builds the search graph: the first line marks when
    # recognition begins, and the rate is measured from there.
    report_progress(force=True)

    num_seqs = 0
    try:
        for seq_tag, seq_features in features:
            if seq_tag in pending:
                raise ValueError(f"duplicate sequence tag in chunk: {seq_tag!r}")
            seq_scores = model.scores(seq_features)
            # Park before submitting, not after: submit() may deliver a result
            # synchronously (SerialRasrRecognizer) or drain an older task to
            # stay under its in-flight limit, and on_result reads this dict.
            pending[seq_tag] = (seq_features, seq_scores if probe is not None else None)
            stats["frames"] += len(seq_features)
            num_seqs += 1
            if verbosity >= 2:
                print(f"Submitting {seq_tag} ({len(seq_features)} frames)")
            if transitions is None:
                recognizer.submit(seq_tag, seq_scores)
            else:
                recognizer.submit(
                    seq_tag, seq_scores,
                    transitions=transitions.transition_scores(seq_tag, seq_features),
                )

        recognizer.drain()
        report_progress(force=True)
    finally:
        recognizer.shutdown()

    if pending:
        raise RuntimeError(
            f"{len(pending)} sequence(s) never got a recognition result, "
            f"e.g. {sorted(pending)[:3]}"
        )

    print(
        f"[TIMING] chunk: {num_seqs} seqs, {stats['frames']} frames, "
        f"{time.time() - started:.1f}s",
        flush=True,
    )
    return ChunkResult(
        accumulator_state=accumulator.state_dict(),
        counter=counter,
        num_seqs=num_seqs,
        num_frames=stats["frames"],
        num_recognized=stats["recognized"],
        hypotheses=hypotheses,
        topology_state=(topology_accumulator.state_dict()
                        if topology_accumulator is not None else None),
    )


def save_chunk(result: ChunkResult, path: str) -> None:
    """
    Persist a chunk result for the reduce step.

    Pickle rather than npz because the payload mixes arrays with a statistics
    counter object. These files live in the job's ``work/`` directory, which
    sisyphus deletes on job completion (``JOB_CLEANUP_KEEP_WORK = False``), so
    nothing here accumulates across epochs.
    """
    with open(path, "wb") as fp:
        pickle.dump(result, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_chunk(path: str) -> ChunkResult:
    with open(path, "rb") as fp:
        return pickle.load(fp)


@dataclass
class MergedChunks:
    """Everything an epoch's chunks add up to, before the model is re-estimated."""

    #: The chunks' statistics summed into one accumulator, ready to finalize.
    accumulator: Accumulator
    statistics: Dict[str, Any]
    totals: Dict[str, int]
    #: ``{seq_tag: hypothesis}`` over all chunks; empty unless the chunks were
    #: run with a ``transcribe``.
    hypotheses: Dict[str, str]
    #: The chunks' transition counts summed into one
    #: :class:`.topology.TopologyAccumulator`, if merge_chunks was given its
    #: factory; else None.
    topology_accumulator: Optional[Any] = None


def merge_chunks(
    *,
    chunk_paths: Sequence[str],
    accumulator_factory: Callable[[], Accumulator],
    topology_accumulator_factory: Optional[Callable[[], Any]] = None,
) -> MergedChunks:
    """
    Sum every chunk's sufficient statistics, without re-estimating anything.

    Split out of :func:`reduce_chunks` so that an update rule can sit between
    the sum and the M-step: batched runs damp the merged statistics against the
    previous step's before finalizing (see :mod:`.ema`). Everything up to that
    point - loading, merging, counting, checking the chunks really partitioned
    the corpus - is identical either way, and there is only one copy of it.

    :param accumulator_factory: builds a fresh, empty accumulator to load each
        chunk's state into - the same spec the chunk tasks were built from
    :param topology_accumulator_factory: the same for the transition counts.
        Given one, every chunk must carry them.
    """
    if not chunk_paths:
        raise ValueError("no chunk results to reduce")

    counters: List[Optional[CounterProtocol]] = []
    totals = {"num_seqs": 0, "num_frames": 0, "num_recognized": 0}
    hypotheses: Dict[str, str] = {}

    merged: Optional[Accumulator] = None
    merged_topology = None
    for path in chunk_paths:
        result = load_chunk(path)
        incoming = accumulator_factory().load_state_dict(result.accumulator_state)
        merged = incoming if merged is None else merged.merge(incoming)
        if topology_accumulator_factory is not None:
            if result.topology_state is None:
                raise RuntimeError(
                    f"{path} carries no transition counts; it was written by a chunk "
                    f"task that had no topology accumulator")
            topology = topology_accumulator_factory().load_state_dict(result.topology_state)
            merged_topology = topology if merged_topology is None else merged_topology.merge(topology)
        counters.append(result.counter)
        totals["num_seqs"] += result.num_seqs
        totals["num_frames"] += result.num_frames
        totals["num_recognized"] += result.num_recognized
        if result.hypotheses:
            hypotheses.update(result.hypotheses)

    assert merged is not None

    if hypotheses and len(hypotheses) != totals["num_recognized"]:
        # Chunks partition the corpus, so tags cannot repeat across them; if
        # they do, update() silently dropped hypotheses and the accumulators
        # saw those sequences more than once too.
        raise RuntimeError(
            f"{totals['num_recognized']} sequences recognized but only "
            f"{len(hypotheses)} distinct tags - the chunks overlap"
        )

    counter = merge_counters(counters)
    statistics = counter.finalize() if counter is not None else {}
    return MergedChunks(
        accumulator=merged,
        statistics=statistics,
        totals=totals,
        hypotheses=hypotheses,
        topology_accumulator=merged_topology,
    )


def reduce_chunks(
    *,
    chunk_paths: Sequence[str],
    accumulator_factory: Callable[[], Accumulator],
    previous_model: ScoreModel,
) -> tuple:
    """
    Merge every chunk's sufficient statistics and finalize the next model.

    The whole-epoch update: :func:`merge_chunks` followed immediately by the
    M-step. A run that updates more than once per epoch calls the two halves
    separately so it can damp the statistics in between.

    :param accumulator_factory: builds a fresh, empty accumulator to load each
        chunk's state into - the same spec the chunk tasks were built from
    :return: ``(model, statistics_dict, totals_dict, hypotheses)``, the last
        being ``{seq_tag: hypothesis}`` over all chunks and empty unless the
        chunks were run with a ``transcribe``
    """
    merged = merge_chunks(
        chunk_paths=chunk_paths, accumulator_factory=accumulator_factory
    )
    model = merged.accumulator.finalize(previous_model)
    return model, merged.statistics, merged.totals, merged.hypotheses
