#!/usr/bin/env python3
"""
Cleanup unused train model checkpoints in the work dir.
"""

import os
import sys
import argparse
import logging
from functools import reduce
from typing import TypeVar

_my_dir = os.path.dirname(__file__)
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = _setup_base_dir + "/tools/sisyphus"
_returnn_dir = _setup_base_dir + "/tools/returnn"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)
        if _returnn_dir not in sys.path:
            sys.path.append(_returnn_dir)


_setup()


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config", nargs="+")
    arg_parser.add_argument("--log-level", type=int, default=20)
    arg_parser.add_argument("--mode", default="dryrun", help="dryrun (default), remove")
    args = arg_parser.parse_args()

    # See Sisyphus __main__ for reference.

    import sisyphus.logging_format
    from sisyphus.loader import config_manager
    from sisyphus import graph
    from sisyphus import gs
    from i6_experiments.users.zeyer.utils import job_aliases_from_log
    from i6_experiments.users.zeyer.utils.set_insert_order import SetInsertOrder
    from returnn.util.basic import human_bytes_size

    # HACK: Replace the set() by SetInsertOrder() to make the order deterministic.
    graph.graph._targets = SetInsertOrder()

    gs.WARNING_ABSPATH = False
    gs.GRAPH_WORKER = 1  # makes the order deterministic, easier to reason about

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    print("Loading Sisyphus configs...")
    config_manager.load_configs(args.config)

    print("Checking active train jobs of the Sisyphus graph...")
    active_train_job_paths = set()
    for job in graph.graph.jobs():
        job_path: str = job._sis_path()
        # Note: no isinstance(job, ReturnnTrainingJob) check here,
        # to also catch fake jobs (via dependency_boundary).
        if not job_path.startswith("work/i6_core/returnn/training/ReturnnTrainingJob."):
            continue
        # print("active train job:", job._sis_path())
        if os.path.isdir(job_path):
            active_train_job_paths.add(job_path)
            aliases = job_aliases_from_log.get_job_aliases(job_path)
            print("Active train job:", aliases[0] if aliases else job)
        else:
            print("Active train job not created yet:", job)
    print("Num active train jobs:", len(active_train_job_paths))

    print("Now checking all train jobs in work dir...")
    total_model_size_to_remove = 0
    total_train_job_count = 0
    train_job_with_models_to_remove = []
    unused_train_jobs = {}  # key: alias (or basename as fallback), value: job path filename
    model_fns_to_remove = []
    found_active_fns = set()  #  as a sanity check.
    for basename in os.listdir("work/i6_core/returnn/training"):
        if not basename.startswith("ReturnnTrainingJob."):
            continue
        fn = "work/i6_core/returnn/training/" + basename

        total_train_job_count += 1
        aliases = job_aliases_from_log.get_job_aliases(fn)
        alias = None
        if not aliases:
            print("No aliases found for train job:", fn)
        else:
            alias = aliases[0]
            alias_path = os.path.basename(os.readlink(alias))
            if alias_path != basename:
                # Can happen, e.g. when cleared by Sisyphus due to error (cleared.0001 etc),
                # or when I changed sth in the config due to some mistake.
                # print("Warning: Alias path mismatch:", alias_path, "actual:", basename)
                # But doesn't matter, clean up anyway, maybe even more so.
                pass

        if fn in active_train_job_paths:
            found_active_fns.add(fn)
            continue

        model_dir = fn + "/output/models"
        if not os.path.isdir(model_dir):
            continue  # can happen when there was an early error, e.g. at file creation
        # First collect all, and then go through them in sorted order below.
        # We do this because here the listdir order is totally arbitrary
        # (due to FS, but sorting by hash also would not help),
        # and to inspect the output, it's much more helpful when this is sorted in some way.
        unused_train_jobs[alias or basename] = fn

    print("Collecting model checkpoint files to remove...")
    # Now go sorted.
    for name, fn in sorted(unused_train_jobs.items()):
        model_dir = fn + "/output/models"
        model_count = 0
        model_size = 0
        with os.scandir(model_dir) as it:
            for model_base_fn in it:
                model_base_fn: os.DirEntry
                if not model_base_fn.name.endswith(".pt"):
                    print("Unexpected model file:", model_base_fn.name)
                    continue
                model_fns_to_remove.append(model_base_fn.path)
                model_size += model_base_fn.stat().st_size
                model_count += 1
        if model_count == 0:
            continue
        print("Unused train job:", name, "model size:", human_bytes_size(model_size))
        total_model_size_to_remove += model_size
        train_job_with_models_to_remove.append(name)

    print("Total train job count:", total_train_job_count)
    print("Total train job with models to remove count:", len(train_job_with_models_to_remove))
    print("List of train jobs with models to remove:")
    for alias in train_job_with_models_to_remove:
        print(f" {alias}")
    if not train_job_with_models_to_remove:
        print(" (none)")
    print("Can remove total model size:", human_bytes_size(total_model_size_to_remove))
    if len(found_active_fns) != len(active_train_job_paths):
        print("ERROR: Did not find some active jobs:")
        for fn in active_train_job_paths:
            if fn not in found_active_fns:
                print(" ", fn)
        raise Exception("Did not find some active jobs.")

    if args.mode == "remove":
        for fn in model_fns_to_remove:
            print("Remove model:", fn)
            os.remove(fn)
    elif args.mode == "dryrun":
        print("Dry-run mode, not removing.")
    else:
        raise ValueError("invalid mode: %r" % args.mode)


if __name__ == "__main__":
    main()
