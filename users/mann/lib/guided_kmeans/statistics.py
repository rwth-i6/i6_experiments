__all__ = [
    "LoopFrequencyCounter",
    "PhoenemeFrequencyCounter",
    "ScoreStatisticsCounter",
    "CombinedStatisticsCounter",
    "CounterBuilder",
    "EpochwiseStatisticsLogger",
]

import os
import json
import random
from dataclasses import dataclass
from functools import cached_property
from collections import Counter
from abc import ABC, abstractmethod
from typing import Any, Protocol, Sequence

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

class ScoreStatisticsCounter(CounterProtocol):
    def __init__(self, **_kwargs):
        self.lm_score_updater = RunningAverageUpdater(shape=())
        self.am_score_updater = RunningAverageUpdater(shape=())
        self.normed_total_score_updater = RunningAverageUpdater(shape=())
    
    def read(self, traceback: list[TracebackItemProtocol]):
        item = traceback[-1]  # only consider the last item in the traceback for scoring
        num_frames = traceback[-1].end_time
        self.lm_score_updater.update_single(item.lm_score)
        self.am_score_updater.update_single(item.am_score)
        self.normed_total_score_updater.update_single((item.am_score + item.lm_score) / num_frames)

    def finalize(self):
        return {
            "average_lm_score": self.lm_score_updater.value.item(),
            "average_am_score": self.am_score_updater.value.item(),
            "average_total_score": (self.lm_score_updater.value + self.am_score_updater.value).item(),
            "average_total_normed_score": self.normed_total_score_updater.value.item(),
        }

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

def get_default_logger(phonemes):
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
    return EpochwiseStatisticsLogger(counter_builder=default_counter_builder)
