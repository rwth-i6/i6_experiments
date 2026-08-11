"""
Statistics hooks for the chunked pipeline.

These are unhashed by the epoch job on purpose: which diagnostics get recorded
must not influence a job's identity, otherwise adding a counter invalidates
every existing result. The corollary is that a hook should not carry a
``tk.Path`` - sisyphus still walks unhashed arguments when collecting job
inputs, so a path in here would create a dependency edge the hash does not
account for. None of the counters do, and they are not user-supplied.
"""

from __future__ import annotations

__all__ = ["default_stats_hooks", "merge_counters"]

from typing import Iterable, Optional, Sequence

from ..statistics import CounterProtocol, get_default_counter_builder


def default_stats_hooks(phonemes: Iterable[str]) -> CounterProtocol:
    """
    The standard counter set, identical to what ``get_default_logger`` builds
    for the single-process pipeline: loop frequency, average segment duration,
    phoneme frequencies, score statistics and a few sampled tracebacks.
    """
    return get_default_counter_builder(phonemes)()


def merge_counters(counters: Sequence[Optional[CounterProtocol]]) -> Optional[CounterProtocol]:
    """
    Reduce per-chunk counters in chunk order. Order matters only for
    ``SampledTracebackPrinter``, whose merge is documented as approximate.
    """
    merged = None
    for counter in counters:
        if counter is None:
            continue
        if merged is None:
            merged = counter
        else:
            merged.merge(counter)
    return merged
