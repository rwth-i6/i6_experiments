"""
Global startup func.
All RETURNN jobs here can use this (add `startup_callback=global_startup_callback` to your ``post_config``).
We might do some extra checks, fixes, etc (depending on environment).
This should be optional and usually not needed if everything is set up correctly (thus ``post_config``).
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List
import os
import shutil
import sys
import textwrap
import traceback
from i6_experiments.common.setups import serialization


def global_startup_callback(*_args, **_kwargs):
    print("RETURNN global startup callback.")
    # noinspection PyBroadException
    try:
        _clean_alternative_file_cache_dirs()

    except Exception:
        # ignore all errors, it's not critical
        print("Warning: global_startup_callback failed:", file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()


def _clean_alternative_file_cache_dirs():
    """
    In some older setups, we left TMPDIR unchanged,
    and the RWTH IPC Slurm setup creates a new temp dir for each job,
    in the style /w0/tmp/slurm_az668407.61594861.

    Go through all those temp dirs, check if they contain a RETURNN file cache,
    and if so, potentially clean them up.
    """
    from returnn.util.basic import get_login_username, human_bytes_size
    from returnn.util import file_cache

    if not os.path.exists("/w0/tmp"):  # RWTH IPC
        return

    dirs = os.listdir("/w0/tmp")
    username = get_login_username()

    disk_usage = shutil.disk_usage("/w0/tmp")
    print(
        f"/w0/tmp disk usage: total {human_bytes_size(disk_usage.total)},"
        f" used {human_bytes_size(disk_usage.used)},"
        f" free {human_bytes_size(disk_usage.free)}"
    )

    freed = 0
    for d in dirs:
        if not d.startswith(f"slurm_{username}."):
            continue
        tmp_dir = "/w0/tmp/" + d
        cache_directory = f"{tmp_dir}/{username}/returnn/file_cache"
        if not os.path.exists(cache_directory):
            continue

        print(f"Found file cache directory: {cache_directory}")
        cache = file_cache.FileCache(cache_directory=cache_directory, cleanup_files_always_older_than_days=0.1)
        res = cache.cleanup()
        print(res)
        freed += res.freed

    print(f"Total freed space: {human_bytes_size(freed)}")
    if freed:
        disk_usage = shutil.disk_usage("/w0/tmp")
        print(
            f"/w0/tmp disk usage after cleanup: total {human_bytes_size(disk_usage.total)},"
            f" used {human_bytes_size(disk_usage.used)},"
            f" free {human_bytes_size(disk_usage.free)}"
        )


def serialize_global_startup_callback() -> serialization.NonhashedCode:
    """
    For old-style RETURNN config serialization.
    """
    return serialization.NonhashedCode(
        textwrap.dedent(f"""\
            def _global_startup_callback(*args, **kwargs):
                try:
                    from {__name__} import global_startup_callback
                except ImportError as exc:
                    print("Warning: could not import {__name__}.global_startup_callback: %s" % exc)
                else:
                    global_startup_callback()
                    
            startup_callback = _global_startup_callback
            """)
    )


def maybe_add_global_startup_callback_to_post_config(config: Dict[str, Any], post_config: Dict[str, Any]) -> None:
    """
    Maybe add the global startup callback to the given config, if no other startup_callback is already defined.
    """
    if "startup_callback" in config:
        return
    if "startup_callback" in post_config:
        return
    post_config["startup_callback"] = global_startup_callback


def maybe_serialize_global_startup_callback(*configs: Optional[Dict[str, Any]]) -> List[serialization.NonhashedCode]:
    """
    Maybe add the global startup callback serialization, if no other startup_callback is already defined.
    """
    for config in configs:
        if config and "startup_callback" in config:
            return []
    return [serialize_global_startup_callback()]
