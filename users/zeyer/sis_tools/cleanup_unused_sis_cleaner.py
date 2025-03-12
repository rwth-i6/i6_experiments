#!/usr/bin/env python3
"""
Cleanup unused jobs and directories in the work dir.
This calls Sisyphus ``tk.cleaner.cleanup_unused``.
"""

import sys
import os
import argparse
import logging
from functools import reduce
from typing import TypeVar


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


def main():
    import textwrap
    import sisyphus.logging_format
    from sisyphus.loader import config_manager
    import sisyphus.toolkit as tk

    # First line in cleanup_unused.__doc__ indentation is broken...
    cleanup_unused_doc_lines = tk.cleaner.cleanup_unused.__doc__.splitlines()
    cleanup_unused_doc = cleanup_unused_doc_lines[0] + "\n" + textwrap.dedent("\n".join(cleanup_unused_doc_lines[1:]))
    arg_parser = argparse.ArgumentParser(
        description=f"{__doc__}\n\ntk.cleaner.cleanup_unused:\n\n{cleanup_unused_doc}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument("config")
    arg_parser.add_argument("--log-level", type=int, default=20)
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), remove, move")
    args = arg_parser.parse_args()

    # See Sisyphus __main__ for reference.

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    config_manager.load_configs(args.config)

    job_dirs = tk.cleaner.list_all_graph_directories()

    # Add some directories we want to exclude
    exclude_list = [
        "work/i6_core/audio",
        "work/i6_core/corpus",
        "work/i6_core/datasets",
        "work/i6_core/text",
        "work/i6_core/tools",
        "work/i6_core/returnn/dataset",
        "work/i6_core/returnn/oggzip",
    ]
    for ex in exclude_list:
        job_dirs[ex] = 1

    tk.cleaner.cleanup_unused(job_dirs=job_dirs, mode=args.mode)


if __name__ == "__main__":
    main()
