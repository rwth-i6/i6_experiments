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
    "OracleSegmentFBRecognizer",
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
    :param num_states: HMM states per phoneme
    :param tie_states: whether a phoneme's states share one emission label
        (the default) or each has its own; untied, ``num_clusters`` must be
        ``V * num_states`` and the columns are laid out as
        :class:`..backoff_fb_graph.LabelLayout` says
    :param apply_sentence_end: whether a path pays ``ln P(</s> | g)`` to
        terminate. True is the correct model; False reproduces RASR, which
        omits it (see :mod:`..backoff_fb_graph`)
    :param zero_topology: every static transition weight 0, so that per-frame
        transition scores passed to :meth:`submit` are the whole transition
        model (see :func:`..backoff_fb_graph.build_topology`)
    :param transition_posteriors: report
        :attr:`.interfaces.RecognitionResult.transition_posteriors`. Not
        implemented before the op's milestone T3, and raises until then.
    :param exit_offset: a fixed offset added to every phoneme's exit weight,
        in nats after the transition scale: a float for all phonemes, or
        ``{symbol: offset}`` with 0 for the symbols it leaves out. Negative is
        a price per phoneme, positive a bonus. See
        :func:`..backoff_fb_graph.build_topology`.
    :param loop_table: ``[V, S]`` loop probabilities to run with in place of
        ``loop_probability`` and ``silence_loop_probability``, rows in the
        lexicon's symbol order (:attr:`symbols`). This is how an epoch hands a
        trained :class:`.topology.StationaryTopology` to the search; the scalars
        then only record where training started. A table filled from the
        scalars gives bitwise the graph the scalars give.
    :param transition_counts: report
        :attr:`.interfaces.RecognitionResult.transition_counts`, the expected
        ``(stay, move)`` count of every (phoneme, state) per sequence (the op's
        SPEC.md 15.2). Costs ~15% of the search and changes no posterior: with
        the flag on, log Z and Gamma are bitwise what they are with it off.
    :param batch_size: sequences per op call (unhashed: throughput only)
    :param max_frames: cap on ``B * T_max`` per call, which is what bounds
        memory (unhashed)
    :param checkpoint_interval: ``-1`` picks the spec-8.1 optimum per batch,
        ``0`` stores every frame, ``K > 0`` fixes it (unhashed: K changes peak
        memory and nothing else -- the op's K1 test asserts the result is
        bitwise identical either way)
    :param device: ``"cuda"`` runs the op. ``"cpu"`` runs the op's exact
        float64 reference instead, which is orders of magnitude slower and is
        there for tests on a machine without a GPU.

    **Transition scores.** ``submit`` takes an optional ``[T, 3]`` array of
    per-frame transition log-scores (:mod:`.transitions`), which the op adds to
    the graph's static weights. A batch passes them for all of its sequences or
    for none, and the padding rows are never read.
    """

    #: ``submit`` takes ``transitions=``; see :class:`.interfaces.Recognizer`.
    accepts_transitions = True

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
        num_states: int = 1,
        tie_states: bool = True,
        apply_sentence_end: bool = True,
        zero_topology: bool = False,
        transition_posteriors: bool = False,
        exit_offset=None,
        loop_table=None,
        transition_counts: bool = False,
        batch_size: int = 16,
        max_frames: int = 0,
        checkpoint_interval: int = -1,
        storage_dtype: str = "float32",
        device: str = "cuda",
    ):
        if transition_posteriors:
            raise NotImplementedError(
                "transition posteriors (the gradient of log Z with respect to "
                "the per-frame transition scores) come from backoff_fb milestone "
                "T3 (SPEC.md 14.8); the op returns log Z and Gamma only")
        if zero_topology and (loop_table is not None or exit_offset is not None):
            raise ValueError(
                "zero_topology=True sets every static transition weight to 0; a "
                "loop table or an exit offset would be ignored")
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
        self.num_states = num_states
        self.tie_states = tie_states
        self.apply_sentence_end = apply_sentence_end
        self.zero_topology = zero_topology
        self.exit_offset = exit_offset
        self.loop_table = None if loop_table is None else np.array(loop_table, dtype=np.float64)
        self.transition_counts = transition_counts
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.checkpoint_interval = checkpoint_interval
        self.storage_dtype = storage_dtype
        self.device = device

        self._graph = None
        self._reference_graph = None
        self._batcher = None
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None
        self._transitions: dict = {}
        self._symbols: Optional[List[str]] = None
        self._n_dead = 0
        self._n_seqs = 0
        self._n_with_transitions = 0

    @property
    def num_labels(self) -> int:
        return self.num_clusters

    @property
    def symbols(self) -> List[str]:
        """The LM symbols in score-column order, read off the lexicon: the row
        order of :attr:`loop_table`, of the exit offsets and of the counts."""
        if self._symbols is None:
            from ..backoff_fb_graph import lexicon_symbols

            self._symbols = lexicon_symbols(str(self.lexicon_path))
        return self._symbols

    def topology(self):
        """The :class:`.topology.StationaryTopology` this search runs with: the
        loop table if one was given, else the table its scalars fill."""
        from .topology import StationaryTopology

        if self.zero_topology:
            raise ValueError("a zero topology has no loop probabilities")
        if self.loop_table is not None:
            return StationaryTopology(self.loop_table, self.symbols)
        return StationaryTopology.from_scalars(
            self.symbols, self.num_states, self.loop_probability, self.silence_loop_probability)

    def _exit_offsets(self) -> Optional[tuple]:
        """``exit_offset`` as one float per symbol, or None if none is set."""
        if self.exit_offset is None:
            return None
        symbols = self.symbols
        if isinstance(self.exit_offset, dict):
            unknown = sorted(set(self.exit_offset) - set(symbols))
            if unknown:
                raise ValueError(f"exit offsets for symbols the lexicon does not have: {unknown}")
            return tuple(float(self.exit_offset.get(sym, 0.0)) for sym in symbols)
        return tuple(float(self.exit_offset) for _ in symbols)

    @property
    def label_layout(self):
        """What the posterior columns mean; see
        :class:`..backoff_fb_graph.LabelLayout`."""
        from ..backoff_fb_graph import LabelLayout

        untied = not self.tie_states and self.num_states > 1
        per_phoneme = self.num_states if untied else 1
        if self.num_clusters % per_phoneme:
            raise ValueError(
                f"num_clusters={self.num_clusters} is not a multiple of "
                f"num_states={self.num_states}, which untied states require")
        return LabelLayout(self.num_clusters // per_phoneme, self.num_states, self.tie_states)

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
        loop_table = None
        if self.loop_table is not None:
            if self.loop_table.shape != (len(self.symbols), self.num_states):
                raise ValueError(
                    f"loop table has shape {self.loop_table.shape}, but the lexicon "
                    f"has {len(self.symbols)} symbols and num_states is {self.num_states}")
            loop_table = tuple(tuple(float(p) for p in row) for row in self.loop_table)
        self._graph = build_graph(
            GraphSpec(
                backoff_fb_root=str(self.backoff_fb_root),
                arpa_path=str(self.arpa_path),
                lexicon_path=str(self.lexicon_path),
                lm_scale=self.lm_scale,
                loop_probability=self.loop_probability,
                silence_loop_probability=self.silence_loop_probability,
                transition_scale=self.transition_scale,
                num_states=self.num_states,
                tie_states=self.tie_states,
                apply_sentence_end=self.apply_sentence_end,
                zero_topology=self.zero_topology,
                loop_table=loop_table,
                exit_offset=self._exit_offsets(),
            ),
            device=self.device,
        )
        if self._graph.F != self.num_clusters:
            raise ValueError(
                f"the lexicon gives {self._graph.F} labels but num_clusters is "
                f"{self.num_clusters}; the score matrix and the graph disagree")
        if self._graph.device.type != "cuda":
            from backoff_fb_ref import TorchGraph

            self._reference_graph = TorchGraph(self._graph.compiled, device="cpu")
        print(
            f"[TIMING] BackoffFBRecognizer: graph ready in "
            f"{time.perf_counter() - t0:.1f}s -- {self._graph}",
            flush=True,
        )
        self._batcher = SequenceBatcher(
            self._run_batch, batch_size=self.batch_size, max_frames=self.max_frames
        )
        self._transitions = {}
        self._n_dead = 0
        self._n_seqs = 0
        self._n_with_transitions = 0

    def _resolved_k(self, T: int) -> int:
        if self.checkpoint_interval >= 0:
            return min(self.checkpoint_interval, T)
        # The spec 8.1 optimum: minimize ceil(T/K) + K.
        return min(range(1, T + 1), key=lambda k: (T + k - 1) // k + k)

    def _run_batch(self, tags: List[str], arrays: List[np.ndarray]) -> None:
        import torch

        from ..backoff_fb_graph import ensure_importable

        ensure_importable(self.backoff_fb_root)
        from backoff_fb.batching import pad_batch

        assert self._on_result is not None and self._graph is not None
        storage = getattr(torch, self.storage_dtype)
        costs, seq_lens = pad_batch(arrays, dtype=torch.float64)
        T = int(seq_lens.max())

        # The transition scores travel beside the batcher rather than through
        # it, keyed by tag, so the bucketing stays the op's own code. A batch
        # either has them for every sequence or for none: mixing would mean a
        # run whose transition model covers some sequences and silently not
        # the others.
        thetas = [self._transitions.pop(tag, None) for tag in tags]
        given = sum(theta is not None for theta in thetas)
        if given not in (0, len(tags)):
            missing = [tag for tag, theta in zip(tags, thetas) if theta is None]
            raise ValueError(
                f"{len(missing)} of {len(tags)} sequences in a batch came without "
                f"transition scores, e.g. {missing[:3]}; either every sequence "
                f"has them or none does")
        transitions = None
        if given:
            # Padding rows are never read (SPEC.md 14.1, consequence 5), so
            # pad_batch's zeros are as good as anything.
            transitions, _ = pad_batch(thetas, dtype=torch.float64)
            self._n_with_transitions += given

        with torch.no_grad():
            # Costs to log-likelihoods. Gradients are never needed here -- the
            # pipeline consumes Gamma itself, not d logZ / d scores -- so this
            # goes straight to the component entry point rather than through
            # the autograd Function.
            am_scale = self.distance_scale * self.emission_scale
            counts = None
            if self._reference_graph is not None:
                from backoff_fb_ref import structured_fb

                log_probs = costs * -am_scale
                logz, gamma, debug = structured_fb(
                    self._reference_graph, log_probs, seq_lens, trans=transitions)
                if self.transition_counts:
                    from backoff_fb.counts import transition_counts

                    counts = transition_counts(self._reference_graph, log_probs, seq_lens,
                                               logz, debug, transitions)
            else:
                from backoff_fb import _C

                log_probs = (costs * -am_scale).to(self.device, storage)
                args = (self._graph.handle, log_probs, seq_lens.to(self.device),
                        storage, self._resolved_k(T),
                        None if transitions is None else transitions.to(self.device))
                if self.transition_counts:
                    logz, gamma, info = _C.forward_backward(*args, transition_counts=True)
                    counts = info["transition_counts"]
                else:
                    # The call every run before the counts made, unchanged.
                    logz, gamma, _ = _C.forward_backward(*args)
            logz = logz.double().cpu().numpy()
            gamma = gamma.double().cpu().numpy()
            if counts is not None:
                counts = counts.double().cpu().numpy()

        for i, tag in enumerate(tags):
            length = int(seq_lens[i])
            self._n_seqs += 1
            g = gamma[i, :length, : self.num_clusters]
            # [V, S, 2]; all zero for a dead sequence, as its Gamma is.
            c = None if counts is None else counts[i].copy()
            if not np.isfinite(logz[i]):
                # No path survives (spec 4.4): Gamma is already all-zero. Pass
                # it through so run_chunk's own checks see it rather than
                # silently training on zeros.
                self._n_dead += 1
                self._on_result(
                    RecognitionResult(seq_tag=tag, posteriors=g,
                                      sequence_score=float(logz[i]),
                                      transition_counts=c)
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
                                  sequence_score=float(logz[i]),
                                  transition_counts=c)
            )

    def submit(self, seq_tag: str, scores: np.ndarray,
               transitions: Optional[np.ndarray] = None) -> None:
        assert self._batcher is not None, "call start() first"
        if transitions is not None:
            transitions = np.asarray(transitions, dtype=np.float64)
            if transitions.shape != (len(scores), 3):
                raise ValueError(
                    f"{seq_tag}: transition scores have shape {transitions.shape}, "
                    f"expected ({len(scores)}, 3) - one (loop, forward, exit) row "
                    f"per frame")
            if seq_tag in self._transitions:
                raise ValueError(f"{seq_tag} submitted twice")
            self._transitions[seq_tag] = transitions
        self._batcher.submit(seq_tag, scores)

    def drain(self) -> None:
        assert self._batcher is not None, "call start() first"
        self._batcher.drain()
        if self._transitions:
            raise RuntimeError(
                f"transition scores for {len(self._transitions)} sequences were "
                f"never used: {sorted(self._transitions)[:3]}")
        if self._n_dead:
            why = ""
            if self._n_with_transitions:
                # The likely cause with an oracle, and otherwise a puzzle.
                why = (f" With hard boundaries in the transition scores, a "
                       f"segment shorter than num_states={self.num_states} frames "
                       f"has no path.")
            print(
                f"[WARNING] BackoffFBRecognizer: {self._n_dead}/{self._n_seqs} "
                f"sequences had log Z = -inf (no path through the graph) and "
                f"contributed zero posteriors.{why}",
                flush=True,
            )
        if self._n_with_transitions:
            print(f"[INFO] BackoffFBRecognizer: {self._n_with_transitions}/"
                  f"{self._n_seqs} sequences ran with per-frame transition scores",
                  flush=True)

    def shutdown(self) -> None:
        self._batcher = None
        self._transitions = {}
        # The graph stays in the module-level cache: another epoch in the same
        # task reuses it instead of paying the compile again.


class OracleSegmentFBRecognizer:
    """
    Frame-level forward-backward with the segment boundaries given.

    A reference alignment says where each phoneme ends. With one HMM state per
    phoneme that pins the segmentation, and a path is then only free to choose
    *which* phoneme each segment holds -- so its score over a segment [b, e) is

        sum_{t in [b, e)} o_t(a)  +  log P(a | history)

    which is exactly what :class:`BackoffFBRecognizer` computes from one
    observation whose score is that sum. So this needs no new kernel: sum the
    per-frame costs within each oracle segment, run the op on the segments, and
    broadcast each segment's posterior back over its frames.

    Verified rather than assumed -- brute-forcing the frame-level trellis
    against the oracle on summed scores agrees to 2e-16 (see
    ``bench/rasr_comparison.md`` in the op repo for where the measurements
    live).

    Two things this does *not* do:

    * a **finite** boundary penalty. The equivalence holds only for a hard
      oracle: inside a segment looping is free and leaving is impossible.
    * **S > 1**. A segment would then span several states with distinct
      emissions, which summing collapses.

    Both need per-frame transition scores, which the op takes since its
    milestone T2: :class:`.transitions.OracleBoundaryTransitions` with
    :class:`BackoffFBRecognizer` on frames does both, and reduces to this
    recognizer at S = 1 with hard boundaries.

    The caller sees frames throughout -- frame features in, frame posteriors
    out -- so ``run_chunk``'s length check passes and ``accumulator.observe``
    weights every frame by its segment's posterior. That per-frame M-step,
    rather than one mean-pooled vector per segment, is the point of the
    exercise.

    :param frame_labels: pickle of ``{seq_tag: [T] int}``, one label per frame,
        aligned 1:1 with the features. This is the ``out_labels`` of
        ``SegmentedFeaturesFromAlignmentJob(pooling="none")``; boundaries are
        where the label changes. Note such labels merge two realizations of one
        phoneme separated by a removed pause (``AA SIL AA``) into one run --
        measured at 0.09% of segments on ls-100, so the oracle is a hair
        coarser than the alignment it came from.
    :param kwargs: everything :class:`BackoffFBRecognizer` takes, forwarded
        unchanged. ``loop_probability`` must stay 0: at segment level a loop
        would let two consecutive segments share a phoneme with *no* LM
        transition, which is a different model.
    """

    def __init__(self, frame_labels: str, **kwargs):
        loop = kwargs.get("loop_probability", 0.0)
        if loop:
            raise ValueError(
                f"loop_probability must be 0 for the oracle-segment recognizer, "
                f"got {loop}. One observation per segment means every one of "
                f"them arrives through an LM transition; a nonzero loop lets "
                f"two segments share a phoneme without one.")
        if kwargs.get("zero_topology", False):
            raise ValueError(
                "zero_topology must stay off for the oracle-segment recognizer: "
                "it makes the loop free, and at segment level a loop is the "
                "model loop_probability=0 exists to exclude. For the oracle "
                "with transition scores, use OracleBoundaryTransitions.")
        states = kwargs.get("num_states", 1)
        if states != 1:
            raise ValueError(
                f"num_states must be 1 for the oracle-segment recognizer, got "
                f"{states}. The equivalence it rests on holds because a "
                f"segment's score is the plain sum of its frames' scores; with "
                f"S > 1 the frames of a segment are split across states with "
                f"their own emissions, and summing collapses exactly the "
                f"distinction S > 1 exists to make. Use the unsegmented "
                f"recognizer (no frame_labels) for a multi-state topology.")
        self.frame_labels = frame_labels
        self._inner = BackoffFBRecognizer(**kwargs)
        self._on_result: Optional[Callable[[RecognitionResult], None]] = None
        self._labels: Optional[dict] = None
        self._runs: dict = {}
        self._n_seqs = 0
        self._n_frames = 0
        self._n_segments = 0

    @property
    def num_labels(self) -> int:
        return self._inner.num_labels

    @property
    def label_layout(self):
        return self._inner.label_layout

    def start(self, on_result: Callable[[RecognitionResult], None]) -> None:
        import pickle
        import time

        self._on_result = on_result
        t0 = time.perf_counter()
        with open(str(self.frame_labels), "rb") as f:
            self._labels = pickle.load(f)
        print(f"[TIMING] OracleSegmentFBRecognizer: {len(self._labels)} frame "
              f"alignments loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        self._runs = {}
        self._n_seqs = self._n_frames = self._n_segments = 0
        self._inner.start(self._handle)

    def _handle(self, result: RecognitionResult) -> None:
        assert self._on_result is not None
        lengths = self._runs.pop(result.seq_tag)
        gamma = result.posteriors
        if gamma.shape[0] != len(lengths):
            # A dead utterance comes back with an empty Gamma; pass it through
            # so run_chunk's own length check reports it rather than np.repeat
            # failing with something obscure.
            self._on_result(result)
            return
        self._on_result(
            RecognitionResult(
                seq_tag=result.seq_tag,
                posteriors=np.repeat(gamma, lengths, axis=0),
                sequence_score=result.sequence_score,
            )
        )

    def submit(self, seq_tag: str, scores: np.ndarray) -> None:
        assert self._labels is not None, "call start() first"
        labels = self._labels.get(seq_tag)
        if labels is None:
            raise KeyError(
                f"no frame alignment for {seq_tag!r} in {self.frame_labels}. "
                f"The alignment and the features have to come from the same "
                f"SegmentedFeaturesFromAlignmentJob.")
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(scores):
            raise ValueError(
                f"{seq_tag}: the alignment has {len(labels)} frames but the "
                f"scores have {len(scores)}. These features are probably the "
                f"pooled ones; this recognizer needs pooling='none'.")

        # Run starts, then the sums over each run. reduceat is exactly this:
        # segment k gets rows [starts[k], starts[k+1]).
        starts = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
        lengths = np.diff(np.r_[starts, len(labels)])
        self._runs[seq_tag] = lengths
        self._n_seqs += 1
        self._n_frames += len(labels)
        self._n_segments += len(starts)
        self._inner.submit(seq_tag, np.add.reduceat(scores, starts, axis=0))

    def drain(self) -> None:
        self._inner.drain()
        if self._runs:
            raise RuntimeError(
                f"{len(self._runs)} sequences were submitted but never came "
                f"back: {sorted(self._runs)[:5]}")
        if self._n_seqs:
            print(f"[INFO] OracleSegmentFBRecognizer: {self._n_seqs} sequences, "
                  f"{self._n_frames} frames -> {self._n_segments} oracle segments "
                  f"({self._n_frames / self._n_segments:.2f} frames each)",
                  flush=True)

    def shutdown(self) -> None:
        self._inner.shutdown()
        self._labels = None
