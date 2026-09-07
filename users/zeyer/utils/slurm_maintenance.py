"""
Upcoming SLURM maintenance reservations, for walltime capping in settings.py.

A reservation only matters when it covers (most of) the partition we submit to:
permanent small carve-outs (devel nodes, staff test reservations) must not cap anything,
so coverage is checked against the partition's node count.

Usage in ``check_engine_limits`` (see the setup's settings.py):
cap ``rqmt["time"]`` to the hours until the next covering reservation,
so jobs still backfill before the window;
once the start time has passed, there is no upcoming reservation anymore
and the cap deactivates itself.
"""

from __future__ import annotations

import datetime
import functools
import subprocess
import time
from typing import Optional


def get_next_maintenance_start(partition: str, *, min_coverage: float = 0.5) -> Optional[datetime.datetime]:
    """
    :param partition: SLURM partition the jobs go to (e.g. "booster")
    :param min_coverage: fraction of the partition's nodes a reservation must cover to count
    :return: start time of the next covering reservation, or None (none upcoming, or query failed)
    """
    # hour-granular cache key: a long-running manager refreshes, but we don't run scontrol per job
    return _next_maintenance_start_cached(partition, min_coverage, int(time.time() // 3600))


# bounded: the hour-bucket cache key changes forever, old buckets must get evicted.
# 1 suffices for the single (partition, coverage) call pattern; alternating queries
# for several partitions within one hour would thrash it and want a bigger size.
@functools.lru_cache(maxsize=1)
def _next_maintenance_start_cached(
    partition: str, min_coverage: float, cache_key: int
) -> Optional[datetime.datetime]:
    del cache_key  # only for the hourly cache granularity
    try:
        res_out = subprocess.run(
            ["scontrol", "-o", "show", "reservations"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
        sinfo_out = subprocess.run(
            ["sinfo", "-h", "-p", partition, "-o", "%D"], capture_output=True, text=True, check=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"slurm_maintenance: query failed ({exc}), not capping walltimes")
        return None

    partition_nodes = sum(int(ln) for ln in sinfo_out.split() if ln.isdigit())
    if not partition_nodes:
        print(f"slurm_maintenance: no node count for partition {partition!r}, not capping walltimes")
        return None

    now = datetime.datetime.now()
    starts = []
    for line in res_out.splitlines():
        fields = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        if "StartTime" not in fields or "NodeCnt" not in fields:
            continue
        # reservations pinned to another partition never block us;
        # PartitionName "(null)" means node-list based, judged by coverage below
        res_part = fields.get("PartitionName", "(null)")
        if res_part not in ("(null)", "", partition):
            continue
        try:
            start = datetime.datetime.strptime(fields["StartTime"], "%Y-%m-%dT%H:%M:%S")
            node_cnt = int(fields["NodeCnt"])
        except ValueError:
            continue
        if start <= now:
            continue  # already active or past; the walltime cap has nothing to protect
        if node_cnt < min_coverage * partition_nodes:
            continue  # small carve-out (devel/test reservation), plenty of nodes stay usable
        starts.append(start)
    return min(starts) if starts else None
