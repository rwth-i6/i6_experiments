"""
Given some Sisyphus config,
check any not-running job (given some conditions),
and run the provided actions.
"""

from __future__ import annotations
import sys
import os
import re
import argparse
import logging
import threading
from functools import reduce
from typing import TypeVar
import subprocess as sp
import textwrap
import better_exchook


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
    import sisyphus.logging_format
    from sisyphus.loader import config_manager
    import sisyphus.toolkit as tk
    from sisyphus import Job
    from sisyphus import gs
    from sisyphus.manager import Manager, create_aliases
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_core.returnn.forward import ReturnnForwardJobV2, ReturnnForwardJob

    # First line in cleanup_unused.__doc__ indentation is broken...
    cleanup_unused_doc_lines = tk.cleaner.cleanup_unused.__doc__.splitlines()
    cleanup_unused_doc = cleanup_unused_doc_lines[0] + "\n" + textwrap.dedent("\n".join(cleanup_unused_doc_lines[1:]))
    arg_parser = argparse.ArgumentParser(
        description=f"{__doc__}\n\ntk.cleaner.cleanup_unused:\n\n{cleanup_unused_doc}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    arg_parser.add_argument("config")
    arg_parser.add_argument("--log-level", type=int, default=20)
    arg_parser.add_argument("--job-type", default="returnn", help="Type of job to consider")
    arg_parser.add_argument("--sis-binary", default="./sis", help="Path to the sis binary to use")
    args = arg_parser.parse_args()

    if args.job_type == "returnn":
        job_types = (ReturnnForwardJobV2, ReturnnForwardJob, ReturnnTrainingJob)
        is_job_match = lambda job: isinstance(job, job_types)
    elif args.job_type == "returnn-forward":
        job_types = (ReturnnForwardJobV2, ReturnnForwardJob)
        is_job_match = lambda job: isinstance(job, job_types)
    else:
        is_job_match = lambda job: re.search(args.job_type, job._sis_id(), re.IGNORECASE)

    # See Sisyphus __main__ for reference.

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    better_exchook.install()
    better_exchook.replace_traceback_format_tb()

    config_manager.load_configs(args.config)

    sis_graph = tk.sis_graph
    job_engine = tk.cached_engine()
    job_engine.start_engine()
    threading._register_atexit(job_engine.stop_engine)

    # We are not really starting/using the manager thread,
    # but it comes with many useful utilities that we use here.
    manager = Manager(sis_graph=sis_graph, job_engine=job_engine)

    # Don't run manager.startup() directly but replicate the relevant parts here.
    manager.job_engine.reset_cache()
    manager.update_jobs()
    manager.update_state_overview()
    manager.print_state_overview()
    if not manager.jobs:
        logging.info("All calculations are done. Nothing to do.")
        return

    def clear_error():
        manager.clear_states(state=gs.STATE_ERROR)
        manager.clear_errors_once = False

    def clear_interrupted():
        manager.clear_states(state=gs.STATE_INTERRUPTED_NOT_RESUMABLE)
        manager.clear_interrupts_once = False

    def maybe_clear_state(state, always_clear, action):
        if state in manager.jobs:
            if always_clear:
                action()
            elif not manager.ignore_once:
                answer = manager.input(f"Clear jobs in {state} state? [y/N] ")

                if answer.lower() == "y":
                    action()
                    manager.print_state_overview()

    maybe_clear_state(gs.STATE_ERROR, manager.clear_errors_once, clear_error)
    maybe_clear_state(gs.STATE_INTERRUPTED_NOT_RESUMABLE, manager.clear_interrupts_once, clear_interrupted)

    gs.SKIP_IS_FINISHED_TIMEOUT = True

    logging.info("Runnable matching jobs:")
    for job in sorted(manager.jobs.get(gs.STATE_RUNNABLE, []), key=lambda j: str(j)):
        job: Job
        if is_job_match(job):
            manager.get_job_info_string(gs.STATE_RUNNABLE, job)

    manager.input("Press Enter to run those job, or Ctrl-C to cancel...")

    job_count = 0
    # Same order as the manager shows them in the overview.
    for job in sorted(manager.jobs.get(gs.STATE_RUNNABLE, []), key=lambda j: str(j)):
        job: Job
        if not is_job_match(job):
            continue

        # See Manager.run_jobs
        logging.info(f"Runnable matching job: {manager.get_job_info_string(gs.STATE_RUNNABLE, job)}")
        job_count += 1

        logging.info("Setup.")
        job._sis_setup_directory()
        logging.info("Create aliases.")
        create_aliases([job])

        for task in job._sis_tasks():
            finished = task.finished()
            logging.info(f"Task: {task.name()} {finished=}")
            if finished:
                continue
            for task_id in task.task_ids():
                logging.info(f"Run task {task.name()}.{task_id}")
                run(sys.executable, args.sis_binary, "worker", job._sis_path(), task.name(), str(task_id))
                assert task.finished(task_id)

        logging.info("All tasks finished. Check output.")
        manager.check_output(write_output=True)

    if job_count == 0:
        logging.error("No matching runnable job found.")
        sys.exit(1)

    logging.info(f"Done. {job_count} matching jobs were run.")


def run(*args):
    print("Running:", *args)
    return sp.check_call(args)


if __name__ == "__main__":
    try:
        main()
    finally:
        for thread in threading.enumerate():
            if thread.is_alive() and not thread.daemon and thread is not threading.current_thread():
                logging.info(f"Non-daemon thread still alive: {thread}")
