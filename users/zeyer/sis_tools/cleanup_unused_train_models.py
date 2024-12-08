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
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"
_returnn_dir = os.path.dirname(_base_dir) + "/tools/returnn"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
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
    import sisyphus.toolkit as tk
    from sisyphus import graph
    from sisyphus import gs
    from i6_core.returnn.training import ReturnnTrainingJob
    from i6_experiments.users.zeyer.utils import job_aliases_from_log
    from returnn.util.basic import human_bytes_size

    gs.WARNING_ABSPATH = False

    sisyphus.logging_format.add_coloring_to_logging()
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=args.log_level)

    config_manager.load_configs(args.config)

    active_train_job_paths = set()
    for job in graph.graph.jobs():
        if not isinstance(job, ReturnnTrainingJob):
            continue
        # print("active train job:", job._sis_path())
        active_train_job_paths.add(job._sis_path())
    print("Num active train jobs:", len(active_train_job_paths))

    total_model_size_to_remove = 0
    total_train_job_count = 0
    total_train_job_with_models_to_remove_count = 0
    model_fns_to_remove = []
    found_active_count = 0  # as a sanity check
    for basename in os.listdir("../../../../../work/i6_core/returnn/training"):
        if not basename.startswith("ReturnnTrainingJob."):
            continue
        fn = "work/i6_core/returnn/training/" + basename
        if fn in active_train_job_paths:
            found_active_count += 1
            continue

        total_train_job_count += 1
        aliases = job_aliases_from_log.get_job_aliases(fn)
        alias = aliases[0]
        alias_path = os.path.basename(os.readlink(alias))
        if alias_path != basename:
            # Can happen, e.g. when cleared by Sisyphus due to error (cleared.0001 etc),
            # or when I changed sth in the config due to some mistake.
            # print("Warning: Alias path mismatch:", alias_path, "actual:", basename)
            # But doesn't matter, clean up anyway, maybe even more so.
            pass

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
        print("Unused train job:", alias, "model size:", human_bytes_size(model_size))
        total_model_size_to_remove += model_size
        total_train_job_with_models_to_remove_count += 1

    print("Total train job count:", total_train_job_count)
    print("Total train job with models to remove count:", total_train_job_with_models_to_remove_count)
    print("Can remove total model size:", human_bytes_size(total_model_size_to_remove))
    assert found_active_count == len(active_train_job_paths), (found_active_count, len(active_train_job_paths))

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
