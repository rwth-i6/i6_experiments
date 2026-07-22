"""
Memory watchdog for the Sisyphus manager.

Installs a background daemon thread that aborts the process (with diagnostics) if its
resident memory (RSS) exceeds a threshold. Safety net against a runaway / retention leak
in the manager: instead of silently bloating to ~100GB and GC-thrashing on a shared login
node, it dumps diagnostics and hard-exits so it can be restarted.

The dump (``mem_watchdog_dump.<pid>.txt`` in ``dump_dir``, default cwd) contains:
- all-thread Python stacktraces (faulthandler),
- a gc object-type histogram (what is on the heap),
- a source-location histogram of retained ``frame`` objects (which functions' frames are
  kept alive -- locates frame-keeping retention, e.g. via JOB_ADD_STACKTRACE_WITH_DEPTH +
  a frame-keeping traceback formatter),
- a referrer-type walk for the dominant types (what pins them).

Manager-only: guard the call with ``not tk.running_in_worker()`` in ``settings.py``::

    from i6_experiments.users.zeyer.utils.memory_watchdog import install_memory_watchdog

    def engine():
        from sisyphus import tk
        if not tk.running_in_worker():
            install_memory_watchdog(threshold_gb=20.0)
        ...
"""

from __future__ import annotations

import os
import sys
import datetime
from typing import Optional

_started = False


def install_memory_watchdog(
    threshold_gb: float = 20.0, interval_sec: float = 15.0, dump_dir: Optional[str] = None
):
    """
    Start a daemon thread that aborts this process (with diagnostics) if RSS exceeds ``threshold_gb``.

    :param threshold_gb: RSS limit in GiB. On crossing it, dump diagnostics and ``os._exit(3)``.
    :param interval_sec: polling interval in seconds.
    :param dump_dir: directory for the dump file. Defaults to the current working dir
        (for the sis manager that is the setup dir).
    """
    global _started
    if _started:
        return
    _started = True

    import threading
    import time

    if dump_dir is None:
        dump_dir = os.getcwd()
    page_size = os.sysconf("SC_PAGE_SIZE")

    def _rss_gb() -> float:
        with open("/proc/self/statm") as f:
            resident_pages = int(f.read().split()[1])
        return resident_pages * page_size / (1024**3)

    def _dump_and_exit(rss_gb: float):
        import gc
        import types
        import faulthandler
        import collections

        path = os.path.join(dump_dir, "mem_watchdog_dump.%d.txt" % os.getpid())
        try:
            with open(path, "w") as out:
                out.write(
                    "MEMORY WATCHDOG: RSS %.1f GB > %.1f GB threshold at %s (pid %d)\n\n"
                    % (rss_gb, threshold_gb, datetime.datetime.now(), os.getpid())
                )
                out.write("=== all-thread stacktraces ===\n")
                out.flush()
                faulthandler.dump_traceback(file=out, all_threads=True)

                out.write("\n=== gc object-type histogram (top 50 by count) ===\n")
                objs = gc.get_objects()
                counter = collections.Counter(type(o).__name__ for o in objs)
                for name, cnt in counter.most_common(50):
                    out.write("%12d  %s\n" % (cnt, name))
                out.write("\ntotal gc-tracked objects: %d\n" % len(objs))
                out.flush()

                # Which functions' frames are being retained? (locates frame-keeping retention)
                out.write("\n=== retained-frame source histogram (top 40 by count) ===\n")
                frame_src = collections.Counter()
                for o in objs:
                    if isinstance(o, types.FrameType):
                        co = o.f_code
                        frame_src["%s:%s" % (co.co_filename, co.co_name)] += 1
                for src, cnt in frame_src.most_common(40):
                    out.write("%12d  %s\n" % (cnt, src))
                out.flush()

                out.write("\n=== referrer-type analysis (what pins the dominant types) ===\n")
                skip = set()
                fr = sys._getframe()
                while fr is not None:
                    skip.add(id(fr))
                    fr = fr.f_back
                skip.add(id(objs))
                targets = ["ExtendedFrameSummary", "StackSummary", "frame", "_OutputLinesCollector"]
                samples = {t: [] for t in targets}
                skip.add(id(samples))
                for o in objs:
                    tn = type(o).__name__
                    if tn in samples and len(samples[tn]) < 8:
                        samples[tn].append(o)
                for t in targets:
                    cur = samples[t]
                    if not cur:
                        continue
                    out.write("\n-- %s (sample %d) --\n" % (t, len(cur)))
                    seen = set(id(x) for x in cur)
                    for lvl in range(3):
                        hist = collections.Counter()
                        nxt = []
                        skip.add(id(cur))
                        skip.add(id(nxt))
                        for x in cur:
                            for r in gc.get_referrers(x):
                                if id(r) in skip or id(r) in seen:
                                    continue
                                hist[type(r).__name__] += 1
                                if len(nxt) < 8:
                                    nxt.append(r)
                                    seen.add(id(r))
                        out.write("   L%d referrers: %s\n" % (lvl, dict(hist.most_common(8))))
                        cur = nxt
                        if not cur:
                            break
                out.flush()
        except Exception as exc:  # noqa: BLE001
            sys.stderr.write("MEMORY WATCHDOG: error while dumping: %r\n" % exc)
        sys.stderr.write(
            "MEMORY WATCHDOG: RSS %.1f GB exceeded %.1f GB; dumped to %s; exiting.\n"
            % (rss_gb, threshold_gb, path)
        )
        sys.stderr.flush()
        os._exit(3)

    def _watch():
        while True:
            time.sleep(interval_sec)
            try:
                rss = _rss_gb()
            except Exception:  # noqa: BLE001
                continue
            if rss > threshold_gb:
                _dump_and_exit(rss)

    threading.Thread(target=_watch, name="mem_watchdog", daemon=True).start()
