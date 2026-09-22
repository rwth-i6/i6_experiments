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

__all__ = [
    "ArgmaxRecognizer",
    "PhonemeIdxMap",
    "RasrViterbiRecognizer",
    "SerialRasrRecognizer",
    "RasrFBRecognizer",
    "BackoffFBRecognizer",
]

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


class ArgmaxRecognizer:
    """
    Unguided assignment: every frame takes its own lowest-cost cluster.

    This is what turns the guided pipeline into plain Lloyd's k-means. No
    lexicon, no language model, no transition costs and no search - so nothing
    couples neighbouring frames, and the "recognition" is a per-frame argmin
    over the score matrix the model already computed.

    Two consequences worth being explicit about, because they are what make an
    unguided epoch a different kind of object from a guided one:

    * **No traceback.** There is no discrete path to report, so
      :attr:`.interfaces.RecognitionResult.traceback` stays empty. The Viterbi
      statistics counters read tracebacks, so they record nothing here and the
      flavor leaves ``statistics`` unset rather than wiring up counters that
      would report zeros. ``out_hypotheses`` comes out as empty lines for the
      same reason, which also makes ``score_reference`` meaningless for an
      unguided run - there is no label inventory to score against.
    * **No distance scale.** ``argmin`` is invariant under multiplication by a
      positive constant, so a scale would be a parameter that provably cannot
      change the result. The RASR recognizers take one because a search weighs
      acoustic costs against LM and transition costs, and only the *ratio*
      matters there; with nothing to weigh against, the knob disappears.

    It also removes essentially all of an epoch's cost: the RASR search is
    ~99.6% of guided epoch wall time (see the module docstring), so an unguided
    epoch is bounded by reading features and computing scores. ``num_chunks``
    stops being about parallelizing search and becomes about parallelizing I/O.

    :param num_clusters: label inventory size. Optional and only used to check
        the score matrix has the width the pipeline thinks it does - the model
        decides that, and a mismatch here means the flavor is inconsistent.
    """

    def __init__(self, num_clusters: Optional[int] = None):
        self.num_clusters = num_clusters
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None

    @property
    def num_labels(self) -> Optional[int]:
        return self.num_clusters

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        self._on_result = on_result

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        assert self._on_result is not None, "start() must be called before submit()"
        scores = np.asarray(scores)
        if scores.ndim != 2:
            raise ValueError(f"expected a [T, K] score matrix, got {scores.shape}")
        if self.num_clusters is not None and scores.shape[1] != self.num_clusters:
            raise ValueError(
                f"{seq_tag}: model scored {scores.shape[1]} clusters, recognizer was "
                f"configured for {self.num_clusters}"
            )
        # Synchronous, like SerialRasrRecognizer: there is no work to hand to a
        # worker pool, and run_chunk parks the features before calling submit()
        # precisely so a result may come back from inside it.
        self._on_result(
            RecognitionResult(
                seq_tag=seq_tag,
                posteriors=scores.argmin(axis=1).astype(np.int64),
            )
        )

    def drain(self) -> None:
        pass  # submit() is synchronous

    def shutdown(self) -> None:
        self._on_result = None


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


class BackoffFBRecognizer:
    """
    Forward-backward on the GPU, via the ``backoff_fb`` CUDA op.

    Computes the same quantity as :class:`RasrFBRecognizer` -- per-frame label
    posteriors under an HMM whose transitions are a backoff n-gram phoneme LM --
    without a RASR worker pool and without pruning: the op is exact over the
    whole graph, so there is no beam to tune and no variational gap.

    Three differences from the RASR path are worth stating, because each is a
    place a mismatch could hide:

    * **Sign.** ``ScoreModel.scores`` yields costs; the op takes emission
      log-likelihoods. The conversion is ``o = -distance_scale * scores``,
      applied here on the GPU.
    * **Batching.** RASR handles one sequence per worker call; the op is
      batched, so submissions are buffered and length-bucketed before a call.
      This changes throughput, never a result -- E5 in the op's suite asserts
      each utterance equals its own B = 1 run bitwise.
    * **No pruning.** RASR runs a beam (``max_beam_size``); this is exact. If
      the two disagree, the beam is the first thing to suspect, not the kernel.

    :param backoff_fb_root: where the op is checked out; pass
        ``tools.BACKOFF_FB``. Must be a path the job container binds --
        /work/asr4 is, /u/mann is not.
    :param arpa_path: the phoneme LM, plain or gzipped ARPA
    :param lexicon_path: the *forward-backward* lexicon, which fixes the score
        column order (see :mod:`..backoff_fb_graph`)
    :param num_clusters: label inventory size; must match the lexicon's
    :param distance_scale: AM weight applied to the costs before the search,
        as :class:`RasrFBRecognizer` does in ``submit``
    :param emission_scale: the second AM weight, which RASR applies inside its
        label scorer on top of ``distance_scale``. The effective weight is the
        product, so it is folded in here rather than ignored -- at the default
        of 1.0 nothing changes, but a config that sets it would otherwise train
        at a silently different AM/LM balance from the RASR path.
    :param lm_scale: baked into the compiled graph
    :param loop_probability: HMM self-loop probability; 0 for segment-pooled
        features, where one observation is one phoneme
    :param apply_sentence_end: whether a path pays ``ln P(</s> | g)`` to
        terminate. True is the correct model; False reproduces RASR, which
        omits it (see :mod:`..backoff_fb_graph`)
    :param batch_size: sequences per op call (unhashed: throughput only)
    :param max_frames: cap on ``B * T_max`` per call, which is what bounds
        memory (unhashed)
    :param checkpoint_interval: ``-1`` picks the spec-8.1 optimum per batch,
        ``0`` stores every frame, ``K > 0`` fixes it (unhashed: K changes peak
        memory and nothing else -- the op's K1 test asserts the result is
        bitwise identical either way)
    """

    def __init__(
        self,
        backoff_fb_root: str,
        arpa_path: str,
        lexicon_path: str,
        num_clusters: int,
        distance_scale: float = 1.0,
        emission_scale: float = 1.0,
        lm_scale: float = 1.0,
        loop_probability: float = 0.0,
        silence_loop_probability: Optional[float] = None,
        transition_scale: float = 1.0,
        apply_sentence_end: bool = True,
        batch_size: int = 16,
        max_frames: int = 0,
        checkpoint_interval: int = -1,
        storage_dtype: str = "float32",
        device: str = "cuda",
    ):
        self.backoff_fb_root = backoff_fb_root
        self.arpa_path = arpa_path
        self.lexicon_path = lexicon_path
        self.num_clusters = num_clusters
        self.distance_scale = distance_scale
        self.emission_scale = emission_scale
        self.lm_scale = lm_scale
        self.loop_probability = loop_probability
        self.silence_loop_probability = silence_loop_probability
        self.transition_scale = transition_scale
        self.apply_sentence_end = apply_sentence_end
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.checkpoint_interval = checkpoint_interval
        self.storage_dtype = storage_dtype
        self.device = device

        self._graph = None
        self._batcher = None
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None
        self._n_dead = 0
        self._n_seqs = 0

    @property
    def num_labels(self) -> int:
        return self.num_clusters

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        import time

        import torch

        from ..backoff_fb_graph import GraphSpec, build_graph, ensure_importable

        ensure_importable(self.backoff_fb_root)
        from backoff_fb.batching import SequenceBatcher

        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "BackoffFBRecognizer needs a GPU, and torch.cuda.is_available() "
                "is False. The chunk job must request one: pass "
                "rqmt={'gpu': 1} to chunked_clustering (settings.py adds --nv "
                "to the container call when it sees a GPU in the rqmt).")

        self._on_result = on_result
        t0 = time.perf_counter()
        self._graph = build_graph(
            GraphSpec(
                backoff_fb_root=str(self.backoff_fb_root),
                arpa_path=str(self.arpa_path),
                lexicon_path=str(self.lexicon_path),
                lm_scale=self.lm_scale,
                loop_probability=self.loop_probability,
                silence_loop_probability=self.silence_loop_probability,
                transition_scale=self.transition_scale,
                apply_sentence_end=self.apply_sentence_end,
            ),
            device=self.device,
        )
        if self._graph.F != self.num_clusters:
            raise ValueError(
                f"the lexicon gives {self._graph.F} labels but num_clusters is "
                f"{self.num_clusters}; the score matrix and the graph disagree")
        print(
            f"[TIMING] BackoffFBRecognizer: graph ready in "
            f"{time.perf_counter() - t0:.1f}s -- {self._graph}",
            flush=True,
        )
        self._batcher = SequenceBatcher(
            self._run_batch, batch_size=self.batch_size, max_frames=self.max_frames
        )
        self._n_dead = 0
        self._n_seqs = 0

    def _resolved_k(self, T: int) -> int:
        if self.checkpoint_interval >= 0:
            return min(self.checkpoint_interval, T)
        # The spec 8.1 optimum: minimize ceil(T/K) + K.
        return min(range(1, T + 1), key=lambda k: (T + k - 1) // k + k)

    def _run_batch(self, tags: List[str], arrays: List[np.ndarray]) -> None:
        import torch

        from ..backoff_fb_graph import ensure_importable

        ensure_importable(self.backoff_fb_root)
        from backoff_fb import _C
        from backoff_fb.batching import pad_batch

        assert self._on_result is not None and self._graph is not None
        storage = getattr(torch, self.storage_dtype)
        costs, seq_lens = pad_batch(arrays, dtype=torch.float64)
        T = int(seq_lens.max())

        with torch.no_grad():
            # Costs to log-likelihoods. Gradients are never needed here -- the
            # pipeline consumes Gamma itself, not d logZ / d scores -- so this
            # goes straight to the component entry point rather than through
            # the autograd Function.
            am_scale = self.distance_scale * self.emission_scale
            log_probs = (costs * -am_scale).to(self.device, storage)
            logz, gamma, _ = _C.forward_backward(
                self._graph.handle, log_probs, seq_lens.to(self.device),
                storage, self._resolved_k(T),
            )
            logz = logz.double().cpu().numpy()
            gamma = gamma.double().cpu().numpy()

        for i, tag in enumerate(tags):
            length = int(seq_lens[i])
            self._n_seqs += 1
            g = gamma[i, :length, : self.num_clusters]
            if not np.isfinite(logz[i]):
                # No path survives (spec 4.4): Gamma is already all-zero. Pass
                # it through so run_chunk's own checks see it rather than
                # silently training on zeros.
                self._n_dead += 1
                self._on_result(
                    RecognitionResult(seq_tag=tag, posteriors=g,
                                      sequence_score=float(logz[i]))
                )
                continue
            # The op normalizes already (its E3 test asserts sum_k Gamma = 1
            # per frame); this is the same guard the RASR path applies and
            # costs one pass over a small array.
            row_sums = g.sum(axis=1, keepdims=True)
            g = np.where(row_sums > 1e-30, g / np.maximum(row_sums, 1e-300),
                         np.zeros_like(g))
            self._on_result(
                RecognitionResult(seq_tag=tag, posteriors=g,
                                  sequence_score=float(logz[i]))
            )

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        assert self._batcher is not None, "call start() first"
        self._batcher.submit(seq_tag, scores)

    def drain(self) -> None:
        assert self._batcher is not None, "call start() first"
        self._batcher.drain()
        if self._n_dead:
            print(
                f"[WARNING] BackoffFBRecognizer: {self._n_dead}/{self._n_seqs} "
                f"sequences had log Z = -inf (no path through the graph) and "
                f"contributed zero posteriors.",
                flush=True,
            )

    def shutdown(self) -> None:
        self._batcher = None
        # The graph stays in the module-level cache: another epoch in the same
        # task reuses it instead of paying the compile again.
