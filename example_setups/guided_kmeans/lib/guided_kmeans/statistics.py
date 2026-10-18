__all__ = [
    "LoopFrequencyCounter",
    "PhoenemeFrequencyCounter",
    "ScoreStatisticsCounter",
    "FBStatisticsCounter",
    "CombinedStatisticsCounter",
    "CounterBuilder",
    "EpochwiseStatisticsLogger",
    "get_default_counter_builder",
    "get_default_logger",
    "get_fb_counter_builder",
    "get_fb_logger",
]

import os
import json
import random
from dataclasses import dataclass
from functools import cached_property
from collections import Counter
from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence

import numpy as np

from .traceback import TracebackItemProtocol
from .running_update import RunningAverageUpdater

@dataclass(frozen=True)
class StatisticsConfig:
    phonemes: set

    @cached_property
    def num_phonemes(self) -> int:
        return len(self.phonemes)

class CounterProtocol(ABC):
    def __init__(self, config: StatisticsConfig, **kwargs):
        self.config = config

    @abstractmethod
    def read(self, traceback: list[TracebackItemProtocol]): ...
    @abstractmethod
    def finalize(self) -> dict: ...

    def merge(self, other: "CounterProtocol") -> "CounterProtocol":
        """
        Combine another counter of the same type into this one, in place.

        Only needed when an epoch is split across independent chunk tasks and
        the per-chunk counters have to be reduced before finalize(); the
        single-process path (EpochwiseStatisticsLogger) never calls this.
        Deliberately not abstract, so existing subclasses stay valid.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support merge()")

    @staticmethod
    def _check_mergeable(own, other):
        if type(own) is not type(other):
            raise TypeError(f"cannot merge {type(other).__name__} into {type(own).__name__}")

class LoopFrequencyCounter(CounterProtocol):
    def __init__(self, **_kwargs):
        self.count_has_loop = 0
        self.segment_count = 0

    def read(self, traceback: list[TracebackItemProtocol]):
        for item in traceback:
            if item.end_time - item.start_time > 1.0:
                self.count_has_loop += 1
            self.segment_count += 1

    def finalize(self):
        return {
            "relative_loop_frequency": self.count_has_loop / self.segment_count if self.segment_count > 0 else 0.0,
            "absolute_loop_count": self.count_has_loop,
        }

    def merge(self, other: "LoopFrequencyCounter") -> "LoopFrequencyCounter":
        self._check_mergeable(self, other)
        self.count_has_loop += other.count_has_loop
        self.segment_count += other.segment_count
        return self

class AverageSegmentDurationCounter(CounterProtocol):
    def __init__(self, **_kwargs):
        self.duration_updater = RunningAverageUpdater(shape=())

    def read(self, traceback: list[TracebackItemProtocol]):
        for item in traceback:
            self.duration_updater.update_single(item.end_time - item.start_time)

    def finalize(self):
        return {
            "average_segment_duration": self.duration_updater.value.item(),
        }

    def merge(self, other: "AverageSegmentDurationCounter") -> "AverageSegmentDurationCounter":
        self._check_mergeable(self, other)
        self.duration_updater.merge(other.duration_updater)
        return self

class PhoenemeFrequencyCounter(CounterProtocol):
    def __init__(self, config: StatisticsConfig, **_kwargs):
        super().__init__(config)
        self.counts = Counter()
        self.counts.update({phoneme: 0 for phoneme in self.config.phonemes})
    
    def read(self, traceback: list[TracebackItemProtocol]):
        self.counts.update(item.lemma for item in traceback)

    def finalize(self):
        fraction_non_zero = sum(1 for count in self.counts.values() if count > 0) / len(self.counts) if self.counts else 0.0
        if sum(self.counts.values()) == 0:
            return {
                "relative_phoneme_frequencies": self.counts,
                "absolute_phoneme_counts": self.counts,
                "fraction_visited_phonemes": fraction_non_zero
            }
        total_count = self.counts.total()
        relative_frequencies = {phoneme: count / total_count for phoneme, count in self.counts.items()}
        return {
            "relative_phoneme_frequencies": relative_frequencies,
            "absolute_phoneme_counts": self.counts,
            "fraction_visited_phonemes": fraction_non_zero
        }

    def merge(self, other: "PhoenemeFrequencyCounter") -> "PhoenemeFrequencyCounter":
        self._check_mergeable(self, other)
        self.counts.update(other.counts)
        return self

class ScoreStatisticsCounter(CounterProtocol):
    def __init__(self, **_kwargs):
        self.lm_score_updater = RunningAverageUpdater(shape=())
        self.am_score_updater = RunningAverageUpdater(shape=())
        self.transition_score_updater = RunningAverageUpdater(shape=())
        self.normed_total_score_updater = RunningAverageUpdater(shape=())

    def read(self, traceback: list[TracebackItemProtocol]):
        item = traceback[-1]  # only consider the last item in the traceback for scoring
        num_frames = traceback[-1].end_time
        transition_score = getattr(item, "transition_score", 0.0)
        total = item.am_score + transition_score + item.lm_score
        self.lm_score_updater.update_single(item.lm_score)
        self.am_score_updater.update_single(item.am_score)
        self.transition_score_updater.update_single(transition_score)
        self.normed_total_score_updater.update_single(total / num_frames)

    def finalize(self):
        total = self.lm_score_updater.value + self.am_score_updater.value + self.transition_score_updater.value
        return {
            "average_lm_score": self.lm_score_updater.value.item(),
            "average_am_score": self.am_score_updater.value.item(),
            "average_transition_score": self.transition_score_updater.value.item(),
            "average_total_score": total.item(),
            "average_total_normed_score": self.normed_total_score_updater.value.item(),
        }

    def merge(self, other: "ScoreStatisticsCounter") -> "ScoreStatisticsCounter":
        self._check_mergeable(self, other)
        self.lm_score_updater.merge(other.lm_score_updater)
        self.am_score_updater.merge(other.am_score_updater)
        self.transition_score_updater.merge(other.transition_score_updater)
        self.normed_total_score_updater.merge(other.normed_total_score_updater)
        return self

class SampledTracebackPrinter(CounterProtocol):
    """
    Sample `num_tracebacks` at random (but in each epoch, the same tracebacks will be sampled, due to the fixed random seed).
    It doesn't do any actual counting, but just stores the tracebacks.
    """
    def __init__(
        self,
        num_tracebacks: int = 5,
        random_seed: int = 0,
        **_kwargs
    ):
        self.num_tracebacks = num_tracebacks
        self.stored_tracebacks: list[str] = []
        self.random_seed = random_seed
        self.rng = random.Random(random_seed)
    
    def read(self, traceback: list[TracebackItemProtocol]):
        if len(self.stored_tracebacks) < self.num_tracebacks:
            # with fixed random seed, this will always sample the same tracebacks in each epoch
            if self.rng.random() < self.num_tracebacks / (len(self.stored_tracebacks) + 1):
                phon_seq = " ".join([item.lemma for item in traceback])
                self.stored_tracebacks.append(phon_seq)

    def finalize(self):
        return {
            "sampled_tracebacks": self.stored_tracebacks
        }

    def merge(self, other: "SampledTracebackPrinter") -> "SampledTracebackPrinter":
        """
        Concatenate up to num_tracebacks examples, this counter's first.

        Note this is *not* the same sample a single-process pass would draw:
        the reservoir in read() is keyed on call order, so it cannot be
        reproduced exactly once the corpus is split across chunks. The result
        is still deterministic (chunks are merged in index order) and this is
        a debug printer, not a statistic anything downstream depends on.
        """
        self._check_mergeable(self, other)
        room = self.num_tracebacks - len(self.stored_tracebacks)
        if room > 0:
            self.stored_tracebacks.extend(other.stored_tracebacks[:room])
        return self

class FBStatisticsCounter(CounterProtocol):
    """
    Statistics accumulated from soft posterior distributions produced by
    forward-backward search, as a drop-in replacement for the default
    Viterbi-traceback counters in FB clustering epochs.

    Driven by :meth:`read_gammas` (called once per sequence from the
    recognition loop or the chunked ``on_result`` hook); the inherited
    :meth:`read` is a no-op because FB produces no Viterbi path.

    Statistics logged:

    * ``mean_log_likelihood_per_frame`` — normalized FB objective; the primary
      convergence signal when guided PER is not available.
    * ``mean_posterior_entropy`` — average ``-Σ_k γ_tk log γ_tk`` over all
      frames; decreases as posteriors sharpen around individual clusters.
    * ``soft_cluster_frequencies`` — expected fraction of frames per cluster;
      the FB analogue of ``PhoenemeFrequencyCounter``.
    * ``dead_cluster_count`` — clusters whose expected frame rate falls below
      ``1e-4``; a sudden increase flags mode collapse.

    **Untied HMM states** (``tie_states=False``, ``num_states > 1``): the
    posterior columns are states, ``num_clusters = V * num_states``, laid out
    as :class:`..backoff_fb_graph.LabelLayout` says. Everything that describes
    *phonemes* is then computed on the posteriors folded back to phonemes, so
    the columns keep their meaning next to a tied run: the frequencies, the
    posterior entropy and both duration estimates (the argmax one also starts
    a new segment where a phoneme's state index goes back down, so two
    consecutive realizations of a phoneme count twice). What describes the
    model's *rows* stays per state: ``dead_cluster_count`` counts dead states,
    and two keys are added - ``soft_state_frequencies`` and
    ``mean_state_posterior_entropy``.
    """

    def __init__(self, num_clusters: int, num_states: int = 1, tie_states: bool = True, **_kwargs):
        self.num_clusters = num_clusters
        self.num_states = num_states
        self.tie_states = tie_states
        self.layout  # validates num_clusters against the layout up front
        self.state_entropy_sum = 0.0
        self.total_log_likelihood = 0.0
        self.total_frames = 0
        self.total_seqs = 0
        self.entropy_sum = 0.0
        self.cluster_counts = np.zeros(num_clusters, dtype=np.float64)
        # Duration bookkeeping. A forward-backward pass produces no path, so
        # there is nothing to count loops on the way the Viterbi counters do;
        # these are read off the posteriors instead. See read_gammas.
        #
        # Counted with their own frame and sequence totals rather than reusing
        # total_frames/total_seqs, so that a partial merge stays *correct*
        # instead of merely plausible. Chunk counters are pickled and merged, so
        # an epoch can mix chunks that carry duration data with chunks that do
        # not (see __setstate__); dividing all frames by boundaries counted on
        # some of them would silently inflate the duration.
        self.stay_mass = 0.0        # expected same-label adjacent frame pairs
        self.adjacent_pairs = 0     # adjacent frame pairs seen
        self.argmax_runs = 0        # runs in the argmax label sequence
        self.duration_frames = 0    # frames those three were measured over
        self.duration_seqs = 0      # sequences likewise

    #: Fields added after runs were already in flight. Chunk counters travel as
    #: pickles, so a reduce task can be handed a counter written by an older
    #: process; without this it dies on the first attribute that did not exist
    #: yet. Keep appending here rather than renaming anything.
    _LATE_FIELDS = (
        ("stay_mass", 0.0),
        ("adjacent_pairs", 0),
        ("argmax_runs", 0),
        ("duration_frames", 0),
        ("duration_seqs", 0),
    )

    def __setstate__(self, state):
        self.__dict__.update(state)
        for name, default in self._LATE_FIELDS:
            self.__dict__.setdefault(name, default)
        # The layout predates nothing that was pickled with a different one:
        # every counter written before it existed was tied. Not in _LATE_FIELDS,
        # whose entries merge() adds up.
        self.__dict__.setdefault("num_states", 1)
        self.__dict__.setdefault("tie_states", True)
        self.__dict__.setdefault("state_entropy_sum", 0.0)
        # Three pickle generations exist, not two: the first version of the
        # duration counters shipped without their own frame/sequence totals and
        # divided the shared ones instead. Such a pickle arrives with boundary
        # counts but zero frames to divide by. Its counts *were* taken over
        # everything the counter saw, so the shared totals are exactly the right
        # values to restore - and restoring them is what keeps a merge with a
        # current chunk correct rather than merely non-crashing.
        if self.__dict__["adjacent_pairs"] and not self.__dict__["duration_frames"]:
            self.__dict__["duration_frames"] = self.__dict__.get("total_frames", 0)
            self.__dict__["duration_seqs"] = self.__dict__.get("total_seqs", 0)

    @property
    def layout(self):
        """The column layout; see :class:`..backoff_fb_graph.LabelLayout`."""
        from .backoff_fb_graph import LabelLayout

        untied = not self.tie_states and self.num_states > 1
        if untied and self.num_clusters % self.num_states:
            raise ValueError(
                f"num_clusters={self.num_clusters} is not a multiple of "
                f"num_states={self.num_states}, which untied states require")
        V = self.num_clusters // self.num_states if untied else self.num_clusters
        return LabelLayout(V, self.num_states, self.tie_states)

    def read(self, traceback):
        pass

    def read_gammas(self, gammas: np.ndarray, log_likelihood: float) -> None:
        gammas = np.asarray(gammas, dtype=np.float64)
        T = len(gammas)
        if T == 0:
            return
        layout = self.layout
        self.total_log_likelihood += log_likelihood
        self.total_frames += T
        self.total_seqs += 1
        self.cluster_counts += gammas.sum(0)
        if layout.untied:
            p = np.clip(gammas, 1e-12, 1.0)
            self.state_entropy_sum += float(-np.sum(p * np.log(p)))
            # From here on, phonemes: the duration and entropy below then mean
            # what they mean for a tied run.
            states, gammas = gammas, layout.phoneme_posteriors(gammas)
        p = np.clip(gammas, 1e-12, 1.0)
        self.entropy_sum += float(-np.sum(p * np.log(p)))

        if T > 1:
            # Two duration estimates, because neither is the real thing and they
            # fail in opposite directions.
            #
            # The soft one: treating adjacent frames as independent, the chance
            # that t and t+1 carry different labels is 1 - sum_k g_t(k) g_t+1(k).
            # Independence is false -- adjacent frames are coupled through the
            # HMM, which is exactly what the pairwise posterior xi would capture
            # and what the op does not return (SPEC.md section 13) - and the
            # error has a known sign: two draws from the same smeared
            # distribution still disagree often, so this *over*-counts
            # boundaries and under-states duration whenever the posteriors are
            # flat. It is therefore an upper bound on the boundary rate.
            self.stay_mass += float(np.sum(gammas[:-1] * gammas[1:]))
            self.adjacent_pairs += T - 1
            self.duration_frames += T
            self.duration_seqs += 1
            # The hard one: run lengths of the argmax label sequence, which is
            # the segmentation a reader would take off the posteriors. Exact for
            # that sequence, and it under-counts boundaries where two labels are
            # nearly tied. Between them the truth is bracketed.
            if layout.untied:
                self.argmax_runs += len(layout.argmax_segments(states))
            else:
                labels = gammas.argmax(axis=1)
                self.argmax_runs += 1 + int(np.count_nonzero(labels[1:] != labels[:-1]))

    def finalize(self) -> dict:
        layout = self.layout
        if self.total_frames == 0:
            return {
                "mean_log_likelihood_per_frame": 0.0,
                "mean_posterior_entropy": 0.0,
                "soft_cluster_frequencies": [0.0] * layout.num_phonemes,
                "dead_cluster_count": self.num_clusters,
                "total_sequences": 0,
                "total_frames": 0,
            }
        total_mass = self.cluster_counts.sum()
        state_freqs = (
            self.cluster_counts / total_mass
            if total_mass > 0
            else np.zeros(self.num_clusters)
        )
        # Per phoneme, so a lexicon-indexed reader (the silence share, the
        # distance to the unigram prior) reads an untied run as it reads a tied
        # one. The identity when tied.
        freqs = layout.phoneme_posteriors(state_freqs[None, :])[0].tolist()
        dead = int((self.cluster_counts / self.total_frames < 1e-4).sum())
        stats = {
            "mean_log_likelihood_per_frame": self.total_log_likelihood / self.total_frames,
            "mean_posterior_entropy": self.entropy_sum / self.total_frames,
            "soft_cluster_frequencies": freqs,
            "dead_cluster_count": dead,
            "total_sequences": self.total_seqs,
            "total_frames": self.total_frames,
        }
        if layout.untied:
            stats["soft_state_frequencies"] = state_freqs.tolist()
            stats["mean_state_posterior_entropy"] = self.state_entropy_sum / self.total_frames
        if self.adjacent_pairs and self.duration_frames:
            # Deliberately the same keys the Viterbi counters use, so the
            # existing "loop_frequency", "loop_count" and "segment_duration"
            # columns read either path and a forward-backward run can be put in
            # the same table as a Viterbi one. The quantity is not identical -
            # an expectation over posteriors rather than a count along a path -
            # which is what the argmax variant beside it is for.
            stay_rate = self.stay_mass / self.adjacent_pairs
            expected_segments = self.duration_seqs + (self.adjacent_pairs - self.stay_mass)
            stats.update(
                {
                    "relative_loop_frequency": stay_rate,
                    "absolute_loop_count": int(round(self.stay_mass)),
                    "average_segment_duration": self.duration_frames / max(expected_segments, 1e-9),
                    "average_segment_duration_argmax": (
                        self.duration_frames / self.argmax_runs if self.argmax_runs else 0.0
                    ),
                    "boundary_rate_argmax": (
                        self.argmax_runs / self.duration_frames
                        if self.duration_frames else 0.0
                    ),
                    # Fraction of the epoch's frames these durations were measured
                    # over. 1.0 normally; below that for the one epoch a code
                    # change landed in, which is worth seeing rather than
                    # averaging away.
                    "duration_coverage": (
                        self.duration_frames / self.total_frames
                        if self.total_frames else 0.0
                    ),
                }
            )
        return stats

    def merge(self, other: "FBStatisticsCounter") -> "FBStatisticsCounter":
        self._check_mergeable(self, other)
        if self.layout != other.layout:
            raise ValueError(f"cannot merge counters over {self.layout} and {other.layout}")
        self.state_entropy_sum += getattr(other, "state_entropy_sum", 0.0)
        self.total_log_likelihood += other.total_log_likelihood
        self.total_frames += other.total_frames
        self.total_seqs += other.total_seqs
        self.entropy_sum += other.entropy_sum
        self.cluster_counts += other.cluster_counts
        # getattr, not attribute access: `other` may be an older pickle that
        # predates these fields and was not routed through __setstate__.
        for name, default in self._LATE_FIELDS:
            setattr(self, name,
                    getattr(self, name, default) + getattr(other, name, default))
        return self


class CombinedStatisticsCounter(CounterProtocol):
    """
    Accepts a list of different CounterProtocol subclass instances
    and combines their results.
    """
    def __init__(self, config: StatisticsConfig, counters: Sequence[CounterProtocol], **_kwargs):
        self.config = config
        self.counters = counters

    def read(self, traceback: list[TracebackItemProtocol]):
        for counter in self.counters:
            counter.read(traceback)

    def finalize(self):
        result = {}
        for counter in self.counters:
            result.update(counter.finalize())
        return result

    def merge(self, other: "CombinedStatisticsCounter") -> "CombinedStatisticsCounter":
        self._check_mergeable(self, other)
        if len(self.counters) != len(other.counters):
            raise ValueError(
                f"counter count mismatch: {len(self.counters)} vs {len(other.counters)}"
            )
        for own, incoming in zip(self.counters, other.counters):
            own.merge(incoming)
        return self

class CounterBuilder(Protocol):
    def __call__(self) -> CounterProtocol: ...

class EpochwiseStatisticsLogger:
    def __init__(
        self,
        counter_builder: CounterBuilder,
        log_file: str = "epoch_statistics.json",
    ):
        self.counter_builder = counter_builder
        self.counter = None
        self.epoch_statistics: dict[int, dict[str, Any]] = {}

        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                self.epoch_statistics = json.load(f)
        self.log_file = log_file
    
    def start_epoch(self):
        self.counter = self.counter_builder()
    
    def read_traceback(self, traceback: list[TracebackItemProtocol]):
        if self.counter is None:
            raise RuntimeError("Epoch not started. Call start_epoch() before reading tracebacks.")
        self.counter.read(traceback)

    def read_soft(self, gammas: np.ndarray, log_likelihood: float) -> None:
        """Forward-backward path: accumulate per-sequence gamma statistics."""
        if self.counter is None:
            raise RuntimeError("Epoch not started. Call start_epoch() before reading gammas.")
        if hasattr(self.counter, "read_gammas"):
            self.counter.read_gammas(gammas, log_likelihood)
    
    def end_epoch(self, epoch: int, print_stats: bool = False):
        if self.counter is None:
            raise RuntimeError("Epoch not started. Call start_epoch() before ending epoch.")
        epoch_stats = self.counter.finalize()
        self.epoch_statistics[epoch] = epoch_stats
        with open(self.log_file, "w") as f:
            json.dump(self.epoch_statistics, f, indent=4)
        if print_stats:
            print(f"Epoch {epoch} statistics:")
            for key, value in epoch_stats.items():
                print(f"  {key}: {value}")
        self.counter = None

def get_default_counter_builder(phonemes) -> CounterBuilder:
    """
    The standard counter set, as a builder that can be called once per epoch
    (single-process path) or once per chunk (chunked path, whose per-chunk
    counters are then reduced via CounterProtocol.merge).
    """
    config = StatisticsConfig(phonemes=set(phonemes))
    def default_counter_builder():
        return CombinedStatisticsCounter(
            config=config,
            counters=[
                LoopFrequencyCounter(),
                AverageSegmentDurationCounter(),
                PhoenemeFrequencyCounter(config),
                ScoreStatisticsCounter(),
                SampledTracebackPrinter(num_tracebacks=5, random_seed=42),
            ]
        )
    return default_counter_builder

def get_default_logger(phonemes):
    return EpochwiseStatisticsLogger(counter_builder=get_default_counter_builder(phonemes))


def get_fb_counter_builder(num_clusters: int) -> CounterBuilder:
    """Builder for one FB-statistics counter instance (called once per epoch or chunk)."""
    def fb_counter_builder():
        return FBStatisticsCounter(num_clusters=num_clusters)
    return fb_counter_builder


def get_fb_logger(num_clusters: int) -> EpochwiseStatisticsLogger:
    return EpochwiseStatisticsLogger(counter_builder=get_fb_counter_builder(num_clusters))
