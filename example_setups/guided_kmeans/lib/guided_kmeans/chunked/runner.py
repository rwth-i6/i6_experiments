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

__all__ = ["ChunkResult", "run_chunk", "reduce_chunks", "save_chunk", "load_chunk"]

import pickle
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..statistics import CounterProtocol
from ..util import ProgressLogger
from .interfaces import Accumulator, FeatureSource, Probe, Recognizer, ScoreModel
from .stats import merge_counters


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


def run_chunk(
    *,
    features: FeatureSource,
    model: ScoreModel,
    recognizer: Recognizer,
    accumulator: Accumulator,
    counter: Optional[CounterProtocol] = None,
    transcribe: Optional[Callable[[List[Any]], str]] = None,
    probe: Optional[Probe] = None,
    verbosity: int = 1,
) -> ChunkResult:
    """
    Score, recognize and accumulate one chunk of the corpus.

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
    :param probe: optional diagnostics observer (:class:`.interfaces.Probe`).
        Given one, the ``[T, K]`` score matrix is parked alongside the features
        rather than handed to the recognizer and dropped, so the probe can
        relate the alignment back to the model that produced it. That is the
        only cost of this hook and it is paid only when it is used: the parked
        matrices are bounded by the same in-flight limit as ``pending`` (32
        sequences at the pipeline's default worker count, ~15 MB at K=40).
        The probe never feeds anything back into the loop - what it records
        cannot change the epoch's result.
    """
    pending: Dict[str, Tuple[np.ndarray, Optional[np.ndarray]]] = {}
    hypotheses: Optional[Dict[str, str]] = {} if transcribe is not None else None
    stats = {"recognized": 0, "frames": 0}

    def on_result(seq_tag: str, posteriors, traceback: List[Any]) -> None:
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
        if counter is not None and traceback:
            counter.read(traceback)
        if transcribe is not None and hypotheses is not None:
            # Unconditionally, empty traceback included: a sequence the search
            # emitted nothing for is a legitimate empty hypothesis and scoring
            # has to see it as a deletion, not as a missing segment.
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

    recognizer.start(on_result)
    progress = ProgressLogger(max(len(features), 1), bar_length=40, logging_step=32)
    progress.start()
    started = time.time()

    num_seqs = 0
    try:
        for seq_idx, (seq_tag, seq_features) in enumerate(features):
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
            recognizer.submit(seq_tag, seq_scores)
            progress.progress(seq_idx)

        recognizer.drain()
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


def reduce_chunks(
    *,
    chunk_paths: Sequence[str],
    accumulator_factory: Callable[[], Accumulator],
    previous_model: ScoreModel,
) -> tuple:
    """
    Merge every chunk's sufficient statistics and finalize the next model.

    :param accumulator_factory: builds a fresh, empty accumulator to load each
        chunk's state into - the same spec the chunk tasks were built from
    :return: ``(model, statistics_dict, totals_dict, hypotheses)``, the last
        being ``{seq_tag: hypothesis}`` over all chunks and empty unless the
        chunks were run with a ``transcribe``
    """
    if not chunk_paths:
        raise ValueError("no chunk results to reduce")

    counters: List[Optional[CounterProtocol]] = []
    totals = {"num_seqs": 0, "num_frames": 0, "num_recognized": 0}
    hypotheses: Dict[str, str] = {}

    merged: Optional[Accumulator] = None
    for path in chunk_paths:
        result = load_chunk(path)
        incoming = accumulator_factory().load_state_dict(result.accumulator_state)
        merged = incoming if merged is None else merged.merge(incoming)
        counters.append(result.counter)
        totals["num_seqs"] += result.num_seqs
        totals["num_frames"] += result.num_frames
        totals["num_recognized"] += result.num_recognized
        if result.hypotheses:
            hypotheses.update(result.hypotheses)

    assert merged is not None
    model = merged.finalize(previous_model)

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
    return model, statistics, totals, hypotheses
