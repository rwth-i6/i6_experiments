"""
Situation:
Some jobs were run with some bug.
You cannot just increase __sis_version__ of the job,
because many earlier jobs were run without that bug.
You can identify the broken jobs in some way
(e.g. ReturnnForwardJob with bad RETURNN version, which can be parsed from log).
It will not just delete the broken job,
but also all later jobs depending on it, because any such result is potentially invalid.
"""

import sys
import os
import argparse
import logging
import datetime
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
    from sisyphus import graph
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_core.returnn.forward import ReturnnForwardJobV2, ReturnnForwardJob
    from i6_experiments.users.zeyer.utils.job_log import get_recent_job_log_change_time

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
    arg_parser.add_argument("--job-type", default="returnn", help="Type of job to consider")
    arg_parser.add_argument(
        "--start-time",
        type=_parse_time,
        default=0,
        help="Only consider jobs modified after this time (absolute or relative time)",
    )
    args = arg_parser.parse_args()

    if args.job_type == "returnn":
        job_types = (ReturnnForwardJobV2, ReturnnForwardJob, ReturnnTrainingJob)
    elif args.job_type == "returnn-forward":
        job_types = (ReturnnForwardJobV2, ReturnnForwardJob)
    else:
        raise NotImplementedError(f"Unknown job type: {args.job_type}")

    # See Sisyphus __main__ for reference.

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    config_manager.load_configs(args.config)

    jobs = []
    for job in graph.graph.jobs():
        if not job._sis_finished():
            continue
        if not isinstance(job, job_types):
            continue

        mtime = get_recent_job_log_change_time(job)
        if mtime < args.start_time:
            continue
        print("***", job, ": mtime: ", datetime.datetime.fromtimestamp(mtime))
        jobs.append(job)

    tk.remove_job_and_descendants(jobs=jobs, mode=args.mode)


def _parse_time(s: str) -> float:
    """
    :param s: either absolute time in format YYYY-MM-DD_HH:MM:SS
      or relative time like -N (N hours ago)
    :return: timestamp
    """
    if s.startswith("-"):
        return datetime.datetime.now().timestamp() + float(s) * 60 * 60
    else:
        dt = datetime.datetime.strptime(s, "%Y-%m-%d_%H:%M:%S")
        return dt.timestamp()


if __name__ == "__main__":
    main()
