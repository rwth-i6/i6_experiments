__all__ = ["ParallelSegmentRecognizer", "PlainTracebackItem"]

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
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np


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
    traceback = _worker_search_algo.recognize_segment(scaled_distances)
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
    return seq_tag, items, os.getpid(), t_start, t_end


class ParallelSegmentRecognizer:
    """
    Wraps a pool of librasr SearchAlgorithm worker processes for parallel
    recognize_segment() calls.

    Usage: start() once, then submit() sequences as they become available
    (non-blocking) and drain() to block until every outstanding submission
    is done, returning results *in submission order*. drain() does not tear
    the pool down, so it can be called repeatedly - once at job end (see
    ClusteringDecodeCallback), or once per phase/epoch while reusing the same
    pool across many drains (see GuidedKMeansClusteringCallback) - call
    shutdown() only when the pool is no longer needed at all.
    """

    def __init__(
        self,
        recognition_config: str,
        num_workers: int | None = 7,
        task_timeout: float | None = 1800.0,
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
        self.executor: ProcessPoolExecutor | None = None
        self.futures: list[tuple[str, Future]] = []

        self._t_first_submit: float | None = None
        self._t_last_submit: float | None = None

    def start(self) -> None:
        assert self.executor is None, "already started"
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

    def _hard_abort(self, reason: str) -> None:
        """
        Kill every worker process and terminate this process immediately,
        bypassing the normal shutdown path.

        Just letting an exception propagate isn't enough: on interpreter
        shutdown, concurrent.futures.process runs an atexit handler that
        *joins* every worker process, and joining a genuinely hung worker
        blocks forever too - the job would keep occupying its cluster
        allocation even after "crashing". SIGKILL-ing the workers and
        calling os._exit() (which skips atexit handlers entirely) is the
        only way to guarantee the job process actually terminates.
        """
        print(f"[FATAL] {reason} - killing worker pool and aborting so the job stops occupying cluster resources.", flush=True)
        if self.executor is not None:
            # No public API for this; `_processes` is the executor's own
            # pid -> multiprocessing.Process map.
            for proc in getattr(self.executor, "_processes", {}).values():
                proc.kill()
        os._exit(1)

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

    def drain(self) -> list[tuple[str, list[PlainTracebackItem]]]:
        """
        Block until every future submitted since the last drain() is done.
        Returns (seq_tag, traceback_items) pairs in submission order - not
        completion order, since some callers' downstream state (e.g. a fixed
        reservoir sample keyed on call order) depends on that.
        """
        assert self.executor is not None, "call start() first"

        if self.futures and self._t_first_submit is not None:
            print(
                f"[TIMING] submission phase (first->last submit): "
                f"{self._t_last_submit - self._t_first_submit:.3f}s for {len(self.futures)} sequences",
                flush=True,
            )

        t_drain_start = time.time()
        results = []
        task_intervals = []  # (pid, start, end) as reported by the workers themselves
        for seq_tag, future in self.futures:
            try:
                result_seq_tag, items, pid, t_start, t_end = future.result(timeout=self.task_timeout)
            except Exception as e:
                self._hard_abort(
                    f"recognize_segment for seq_tag={seq_tag!r} did not complete "
                    f"within task_timeout={self.task_timeout}s: {e!r}"
                )
            assert result_seq_tag == seq_tag
            results.append((seq_tag, items))
            task_intervals.append((pid, t_start, t_end))
        t_drain_end = time.time()

        self.futures.clear()
        self._t_first_submit = None
        self._t_last_submit = None

        # Sanity-check whether tasks actually ran concurrently: sum of
        # per-task durations divided by the wall-clock span they occupy.
        # ~1x means effectively serial, ~num_workers means real parallelism.
        if task_intervals:
            busy_time = sum(end - start for _, start, end in task_intervals)
            span_start = min(start for _, start, _ in task_intervals)
            span_end = max(end for _, _, end in task_intervals)
            span = span_end - span_start
            n_distinct_workers = len({pid for pid, _, _ in task_intervals})
            print(
                f"[TIMING] drain phase: {t_drain_end - t_drain_start:.3f}s wall, "
                f"{busy_time:.3f}s summed worker-busy time over {len(task_intervals)} tasks, "
                f"span={span:.3f}s, distinct worker pids={n_distinct_workers}, "
                f"concurrency utilization={busy_time / span if span > 0 else float('nan'):.2f}x "
                f"(expected up to ~{self.num_workers}x if truly parallel, ~1x if effectively serial)",
                flush=True,
            )

        return results

    def shutdown(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True)
            self.executor = None

    def __getstate__(self) -> dict:
        d = dict(self.__dict__)
        d["executor"] = None
        d["futures"] = []
        return d

    def __setstate__(self, d) -> None:
        self.__dict__ = d
