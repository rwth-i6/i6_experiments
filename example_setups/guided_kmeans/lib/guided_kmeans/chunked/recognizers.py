"""
Recognizers: per-frame cluster scores in, per-frame labels out.

The RASR search is the dominant cost of a clustering epoch (measured: ~17 s of
CPU per sequence, ~99.6% of epoch wall time), which is the whole reason the
epoch is chunked across cluster tasks in the first place.

Adding an n-best or full-sum recognizer means implementing
:class:`.interfaces.Recognizer` and returning the soft form of
:data:`.interfaces.Posteriors`; no accumulator or loop code has to change.
"""

from __future__ import annotations

__all__ = ["PhonemeIdxMap", "RasrViterbiRecognizer", "SerialRasrRecognizer", "RasrFBRecognizer"]

from collections import UserDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional

import numpy as np

from i6_core.lib.lexicon import Lexicon

from ..parallel_recognizer import ParallelFBRecognizer, ParallelSegmentRecognizer, PlainTracebackItem
from ..util import segments_to_array
from .interfaces import Posteriors, RecognitionResult


class PhonemeIdxMap(UserDict):
    """
    Lexicon phoneme -> index, the label inventory shared by the recognizer and
    the model's cluster axis.

    Deliberately a copy of the class in ``..clustering`` rather than an import:
    that module pulls in RETURNN and torch at import time, which a CPU chunk
    task running only RASR search has no reason to pay for.
    """

    def __init__(self, lexicon_path: str):
        self.data = self.load_lexicon_map(lexicon_path)

    @staticmethod
    def load_lexicon_map(lexicon_path: str) -> dict:
        lex = Lexicon()
        lex.load(lexicon_path)
        return {phon: i for i, phon in enumerate(lex.phonemes)}

    def apply(self, it: Iterable[str]) -> List[int]:
        return [self[phon] for phon in it]

    def inverse(self) -> dict:
        return {idx: phon for phon, idx in self.data.items()}


def traceback_to_labels(traceback: List[Any], phoneme_map: PhonemeIdxMap) -> np.ndarray:
    """
    Expand a RASR traceback into one label per frame - the same conversion
    ``GuidedKMeansClusteringCallback._apply_recognition_result`` performs.
    """
    segments = np.asarray(
        [(phoneme_map[item.lemma], item.start_time, item.end_time) for item in traceback]
    )
    if segments.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return segments_to_array(segments).astype(np.int64)


class RasrViterbiRecognizer:
    """
    Wraps :class:`ParallelSegmentRecognizer` (a pool of librasr
    ``SearchAlgorithm`` worker processes) behind the Recognizer protocol.

    :param recognition_config: path to the RASR config
    :param lexicon_path: lexicon defining the label inventory
    :param distance_scale: acoustic scale applied to the model's scores before
        search, matching ``scaled_distances = distances * self.distance_scale``
        in the single-process callback
    :param num_workers: worker processes *within one chunk task*. Total
        parallelism is num_chunks x num_workers; this is a scheduling knob and
        is excluded from the job hash.
    """

    def __init__(
        self,
        recognition_config: str,
        lexicon_path: str,
        distance_scale: float = 1.0,
        num_workers: Optional[int] = 8,
        task_timeout: Optional[float] = 1800.0,
    ):
        self.recognition_config = recognition_config
        self.phoneme_map = PhonemeIdxMap(lexicon_path)
        self.distance_scale = distance_scale
        self._recognizer = ParallelSegmentRecognizer(
            recognition_config, num_workers=num_workers, task_timeout=task_timeout
        )
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None

    @property
    def num_labels(self) -> int:
        return len(self.phoneme_map)

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        self._on_result = on_result
        self._recognizer.start(on_result=self._handle)

    def _handle(self, seq_tag: str, traceback: List[PlainTracebackItem]) -> None:
        assert self._on_result is not None
        self._on_result(
            RecognitionResult(
                seq_tag=seq_tag,
                posteriors=traceback_to_labels(traceback, self.phoneme_map),
                traceback=traceback,
            )
        )

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        self._recognizer.submit(seq_tag, scores * self.distance_scale)

    def drain(self) -> None:
        self._recognizer.drain()

    def shutdown(self) -> None:
        self._recognizer.shutdown()


class RasrFBRecognizer:
    """
    Parallel forward-backward recognizer behind the Recognizer protocol.

    Runs recognize_segment_forward_backward() in a worker pool (same spawn
    infrastructure as RasrViterbiRecognizer) and delivers the soft gamma
    matrix as the Posteriors value — a plain 2-D numpy array [T, num_clusters].

    The paired accumulator must handle dense gammas; MeanAccumulator does.
    GaussianAccumulator does not (it requires hard assignments).

    :param recognition_config: RASR config with the FB language model topology
    :param num_clusters: label inventory size (gamma columns to keep)
    :param distance_scale: acoustic scale applied to model scores before search
    :param num_workers: worker processes within this chunk task
    """

    def __init__(
        self,
        recognition_config: str,
        num_clusters: int,
        distance_scale: float = 1.0,
        num_workers: int | None = 8,
        task_timeout: float | None = 1800.0,
        per_task_timeout: float | None = None,
    ):
        self.recognition_config = recognition_config
        self.num_clusters = num_clusters
        self.distance_scale = distance_scale
        self._recognizer = ParallelFBRecognizer(
            recognition_config,
            num_workers=num_workers,
            task_timeout=task_timeout,
            per_task_timeout=per_task_timeout,
        )
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None

    @property
    def num_labels(self) -> int:
        return self.num_clusters

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        self._on_result = on_result
        self._recognizer.start(on_result=self._handle)

    def _handle(self, seq_tag: str, gammas: np.ndarray, log_likelihood: float) -> None:
        assert self._on_result is not None
        if gammas.shape[0] == 0:
            # Broken worker — propagate so run_chunk's length check raises cleanly.
            self._on_result(RecognitionResult(seq_tag=seq_tag, posteriors=gammas))
            return
        # RASR accumulates alpha/beta in float32; per-frame normalization recovers
        # the correct relative posteriors (same fix as the single-process FB path).
        phoneme_gammas = gammas[:, :self.num_clusters]
        row_sums = phoneme_gammas.sum(axis=1, keepdims=True)
        phoneme_gammas = np.where(
            row_sums > 1e-30,
            phoneme_gammas / np.maximum(row_sums, 1e-300),
            np.zeros_like(phoneme_gammas),
        )
        self._on_result(
            RecognitionResult(
                seq_tag=seq_tag,
                posteriors=phoneme_gammas,
                sequence_score=log_likelihood,
            )
        )

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        self._recognizer.submit(seq_tag, scores * self.distance_scale)

    def drain(self) -> None:
        self._recognizer.drain()

    def shutdown(self) -> None:
        self._recognizer.shutdown()


class SerialRasrRecognizer:
    """
    Single-process variant: same search, no worker pool, results delivered
    synchronously from submit(). Slow by design - for debugging a chunk
    interactively, where a process pool obscures tracebacks.
    """

    def __init__(
        self,
        recognition_config: str,
        lexicon_path: str,
        distance_scale: float = 1.0,
    ):
        self.recognition_config = recognition_config
        self.phoneme_map = PhonemeIdxMap(lexicon_path)
        self.distance_scale = distance_scale
        self._search = None
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None

    @property
    def num_labels(self) -> int:
        return len(self.phoneme_map)

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        from librasr import Configuration, SearchAlgorithm

        config = Configuration()
        config.set_from_file(self.recognition_config)
        self._search = SearchAlgorithm(config=config)
        self._on_result = on_result

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        assert self._search is not None and self._on_result is not None
        traceback = self._search.recognize_segment(scores * self.distance_scale, seq_tag)
        self._on_result(
            RecognitionResult(
                seq_tag=seq_tag,
                posteriors=traceback_to_labels(traceback, self.phoneme_map),
                traceback=traceback,
            )
        )

    def drain(self) -> None:
        pass  # submit() is synchronous

    def shutdown(self) -> None:
        self._search = None
