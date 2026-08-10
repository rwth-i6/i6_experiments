__all__ = ["ParallelSegmentRecognizer", "PlainTracebackItem", "RecognizerAborted"]

import os

# Every BLAS/OpenMP backend we might link against (numpy/scipy in the calling
# process, and whatever librasr's search engine uses internally in the worker
# processes) defaults to spawning one thread per *visible* core, not per core
# actually granted by the cluster's cgroup. With `num_workers` processes
# already providing the parallelism, letting each of those processes
# additionally fan out into its own thread pool oversubscribes the allotted
# cpu_rqmt many times over and can make a "parallel" run slower than a serial
# one. Pin everything to 1 thread per process here, at import time, so it
# takes effect before this module's own numpy calls (or a caller's) get a
# chance to lazily size a BLAS thread pool.
#
# NB: RETURNN's launcher exports OMP_NUM_THREADS/MKL_NUM_THREADS = cpu_rqmt
# into the job's environment *before* any of this is imported, so a
# setdefault() here would be a no-op - these must be force-overridden.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_env_var] = "1"

import multiprocessing
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Callable, NoReturn

import numpy as np


class RecognizerAborted(RuntimeError):
    """
    Raised by :meth:`ParallelSegmentRecognizer._hard_abort` after the worker
    pool has been killed, i.e. when recognition cannot continue: a worker
    hung past ``task_timeout``, or the pool broke outright.

    Propagating an exception (rather than exiting the process) is what lets
    the caller's own error handling run - notably sisyphus', which records the
    task as failed instead of silently rescheduling it.
    """


@dataclass(frozen=True)
class PlainTracebackItem:
    """
    Plain, always-picklable stand-in for whatever traceback item type
    librasr's SearchAlgorithm.recognize_segment() actually returns, carrying
    exactly the fields callers use (see .traceback.TracebackItemProtocol) so
    results can cross the ProcessPoolExecutor boundary regardless of whether
    the native binding type itself supports pickling.
    """
    lemma: str
    start_time: float
    end_time: float
    lm_score: float
    am_score: float


def _init_worker(recognition_config: str):
    global _worker_search_algo
    # Re-assert thread pinning: this runs first in each freshly forked/spawned
    # worker, before librasr's search engine (and any BLAS library it links
    # against) gets a chance to size its thread pool from the environment.
    for env_var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[env_var] = "1"
    t0 = time.perf_counter()
    from librasr import Configuration, SearchAlgorithm
    config = Configuration()
    config.set_from_file(recognition_config)
    _worker_search_algo = SearchAlgorithm(config=config)
    print(f"[TIMING] _init_worker pid={os.getpid()} took {time.perf_counter() - t0:.3f}s", flush=True)


def _worker_recognize(seq_tag: str, scaled_distances: np.ndarray):
    global _worker_search_algo
    t_start = time.time()
    traceback = _worker_search_algo.recognize_segment(scaled_distances, seq_tag)
    t_end = time.time()
    items = [
        PlainTracebackItem(
            lemma=item.lemma,
            start_time=item.start_time,
            end_time=item.end_time,
            lm_score=item.lm_score,
            am_score=item.am_score,
        )
        for item in traceback
    ]
    print(
        f"[TIMING] _worker_recognize pid={os.getpid()} seq={seq_tag} "
        f"took {t_end - t_start:.3f}s (wall {t_start:.3f}-{t_end:.3f})",
        flush=True,
    )
    # print(items)
    return seq_tag, items, os.getpid(), t_start, t_end


class ParallelSegmentRecognizer:
    """
    Wraps a pool of librasr SearchAlgorithm worker processes for parallel
    recognize_segment() calls.

    Usage: start(on_result) once, then submit() sequences as they become
    available (non-blocking). Every result is delivered to on_result(seq_tag,
    traceback_items) in submission order - not completion order, since some
    callers' downstream state (e.g. a fixed reservoir sample keyed on call
    order) depends on that. Results are applied as soon as they're available
    rather than buffered: submit() itself drains and applies the oldest
    outstanding task whenever more than max_pending_tasks are in flight, so
    memory use stays bounded by max_pending_tasks regardless of how many
    sequences a phase/job submits in total - buffering results for an entire
    RECOGNITION phase (which can span the whole corpus) grew a job's RSS from
    tens of MB to double-digit GB before this was added. drain() flushes
    whatever's still outstanding and does not tear the pool down, so it can
    be called repeatedly - once at job end (see ClusteringDecodeCallback), or
    once per phase/epoch while reusing the same pool (see
    GuidedKMeansClusteringCallback) - call shutdown() only when the pool is
    no longer needed at all.
    """

    def __init__(
        self,
        recognition_config: str,
        num_workers: int | None = 7,
        task_timeout: float | None = 1800.0,
        max_pending_tasks: int | None = None,
    ):
        self.recognition_config = recognition_config
        self.num_workers = num_workers
        # A worker that dies outright (segfault, OOM-killed) is caught by
        # ProcessPoolExecutor itself (BrokenProcessPool). A worker that
        # *hangs* while staying alive - e.g. librasr's search getting stuck
        # on some pathological input - is invisible to it: the OS sees a
        # healthy process, so future.result() would otherwise block forever.
        # This bounds that wait; see _hard_abort() for what happens next.
        self.task_timeout = task_timeout
        # Keep enough queued that workers are never starved waiting for the
        # next submission, without letting outstanding tasks (and their
        # scaled_distances payloads) accumulate for an entire phase.
        self.max_pending_tasks = max_pending_tasks if max_pending_tasks is not None else 4 * (num_workers or 4)
        self.executor: ProcessPoolExecutor | None = None
        self.futures: deque[tuple[str, Future]] = deque()
        self._on_result: Callable[[str, list[PlainTracebackItem]], None] | None = None

        self._t_first_submit: float | None = None
        self._t_last_submit: float | None = None
        self._task_intervals: list[tuple[int, float, float]] = []  # (pid, start, end), for drain()'s stats

    def start(self, on_result: Callable[[str, list[PlainTracebackItem]], None]) -> None:
        assert self.executor is None, "already started"
        self._on_result = on_result
        # The calling process typically already holds an active CUDA context
        # (the encoder model loaded onto the GPU) by the time this runs. The
        # default "fork" start method is not safe/supported in that
        # situation (CUDA contexts don't survive a fork) and in practice
        # leaves the pool unable to run tasks in true parallel - "spawn"
        # starts each worker as a clean interpreter instead. RETURNN itself
        # only ever uses "spawn" for the same reason (see
        # returnn/util/watch_memory.py).
        t0 = time.perf_counter()
        ctx = multiprocessing.get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=self.num_workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(self.recognition_config,),
        )
        print(
            f"[TIMING] ProcessPoolExecutor constructed in {time.perf_counter() - t0:.3f}s "
            f"(workers are started lazily on first submit(), so this is expected to be fast)",
            flush=True,
        )

    def _hard_abort(self, reason: str) -> NoReturn:
        """
        Kill every worker process, then raise :class:`RecognizerAborted`.

        The kill has to come first, and that ordering is the whole subtlety.
        On interpreter shutdown concurrent.futures.process *joins* every
        worker, and joining a genuinely hung worker never returns - so simply
        raising would leave the job occupying its cluster allocation forever.
        SIGKILL-ing them first makes those joins return immediately, because
        the processes are already dead.

        Measured, spawn context, one hung worker per case:

            kill then raise  -> exits in 2.1s, traceback visible
            raise, no kill   -> still alive after 120s, never terminates
            kill then _exit  -> exits in 2.1s, no traceback

        So raising is as reliable as the os._exit() this used to do, and
        strictly better in two ways: the traceback reaches the log, and the
        exception propagates to whatever is driving the recognizer. That
        matters for sisyphus - os._exit() skips Task.run's exception handling,
        which is the only thing that records a task as failed, leaving a task
        that is neither finished nor failed for the manager to resubmit
        forever.
        """
        print(f"[FATAL] {reason} - killing worker pool and aborting so the job stops occupying cluster resources.", flush=True)
        if self.executor is not None:
            # No public API for this; `_processes` is the executor's own
            # pid -> multiprocessing.Process map.
            for proc in getattr(self.executor, "_processes", {}).values():
                proc.kill()
        raise RecognizerAborted(reason)

    def _drain_one(self) -> None:
        """Pop and apply the oldest outstanding future, blocking on it if needed."""
        assert self._on_result is not None, "call start(on_result) first"
        seq_tag, future = self.futures.popleft()
        try:
            result_seq_tag, items, pid, t_start, t_end = future.result(timeout=self.task_timeout)
        except Exception as e:
            self._hard_abort(
                f"recognize_segment for seq_tag={seq_tag!r} did not complete "
                f"within task_timeout={self.task_timeout}s: {e!r}"
            )
        assert result_seq_tag == seq_tag
        self._task_intervals.append((pid, t_start, t_end))
        self._on_result(seq_tag, items)

    def submit(self, seq_tag: str, scaled_distances: np.ndarray) -> None:
        assert self.executor is not None, "call start() first"
        t_submit = time.time()
        if self._t_first_submit is None:
            self._t_first_submit = t_submit
        self._t_last_submit = t_submit

        try:
            future = self.executor.submit(_worker_recognize, seq_tag, scaled_distances)
        except Exception as e:
            self._hard_abort(f"submit() for seq_tag={seq_tag!r} failed: {e!r} (worker pool likely already broken)")
        self.futures.append((seq_tag, future))

        # Bound memory: apply (and release) the oldest outstanding results
        # instead of letting them, and their queued scaled_distances
        # payloads, pile up for an entire phase - on a phase spanning
        # thousands of sequences this previously grew RSS from tens of MB to
        # double-digit GB.
        while len(self.futures) > self.max_pending_tasks:
            self._drain_one()

    def drain(self) -> None:
        """
        Block until every outstanding submission (including whatever
        submit() hasn't already drained incrementally) has been applied via
        on_result, then print this batch's submission/concurrency stats.
        """
        assert self.executor is not None, "call start() first"

        n_pending = len(self.futures)
        if self._t_first_submit is not None and self._t_last_submit is not None:
            print(
                f"[TIMING] submission phase (first->last submit): "
                f"{self._t_last_submit - self._t_first_submit:.3f}s, {n_pending} sequences still pending at drain()",
                flush=True,
            )

        t_drain_start = time.time()
        while self.futures:
            self._drain_one()
        t_drain_end = time.time()

        self._t_first_submit = None
        self._t_last_submit = None

        # Sanity-check whether tasks actually ran concurrently: sum of
        # per-task durations divided by the wall-clock span they occupy.
        # ~1x means effectively serial, ~num_workers means real parallelism.
        task_intervals, self._task_intervals = self._task_intervals, []
        if task_intervals:
            busy_time = sum(end - start for _, start, end in task_intervals)
            span_start = min(start for _, start, _ in task_intervals)
            span_end = max(end for _, _, end in task_intervals)
            span = span_end - span_start
            n_distinct_workers = len({pid for pid, _, _ in task_intervals})
            print(
                f"[TIMING] drain phase: {t_drain_end - t_drain_start:.3f}s wall to flush the last {n_pending} pending, "
                f"{busy_time:.3f}s summed worker-busy time over {len(task_intervals)} tasks since the last drain(), "
                f"span={span:.3f}s, distinct worker pids={n_distinct_workers}, "
                f"concurrency utilization={busy_time / span if span > 0 else float('nan'):.2f}x "
                f"(expected up to ~{self.num_workers}x if truly parallel, ~1x if effectively serial)",
                flush=True,
            )

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["executor"] = None
        d["futures"] = deque()
        # A bound method of the owning callback would otherwise pull that
        # callback (and transitively this recognizer again) into the pickle.
        # Not restorable via unpickling anyway - start(on_result) must be
        # called again regardless, same as the executor.
        d["_on_result"] = None
        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
