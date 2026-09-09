"""
Persistent, self-expiring registry of bad GPU nodes (e.g. psslurm doSpawn "black holes").

The registry file lives in the setup dir (the cwd of the Sisyphus manager):
``bad_gpu_nodes.json``, mapping hostname -> last-failure unix time.
Entries older than :data:`MAX_AGE_SECS` are ignored on read and dropped on the next write,
so the list cleans itself up once the operators fix the nodes -- no manual bookkeeping.

Writers: :class:`job_failure_handler.JobFailureHandler` (on a detected bad-host failure),
or a manual ``report_bad_node("jpbo-...")`` for a known wave.
Reader: ``settings.check_engine_limits``, so every submission excludes the current set.
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List

FILENAME = "bad_gpu_nodes.json"
# These problems are usually transient (operators fix/reboot the nodes),
# so entries expire like the failure handler's in-memory list does.
MAX_AGE_SECS = 24 * 3600


def get_bad_nodes() -> List[str]:
    """:return: sorted hostnames with a failure younger than :data:`MAX_AGE_SECS`"""
    now = time.time()
    return sorted(host for host, ts in _load().items() if now - ts <= MAX_AGE_SECS)


def report_bad_node(host: str):
    """Record a failure on ``host`` (short hostname; a FQDN is truncated) and expire old entries."""
    host = host.split(".")[0]
    d = _load()
    now = time.time()
    d[host] = now
    d = {h: ts for h, ts in d.items() if now - ts <= MAX_AGE_SECS}
    tmp_filename = FILENAME + ".tmp"
    with open(tmp_filename, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1, sort_keys=True)
        f.write("\n")
    os.rename(tmp_filename, FILENAME)


def _load() -> Dict[str, float]:
    if not os.path.exists(FILENAME):
        return {}
    with open(FILENAME, encoding="utf-8") as f:
        d = json.load(f)
    assert isinstance(d, dict)
    return d
