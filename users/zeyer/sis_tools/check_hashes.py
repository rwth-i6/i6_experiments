#!/usr/bin/env python3

"""
Check hashes...
"""

from __future__ import annotations
import argparse
import os
import sys
import logging
import time
from functools import reduce


# It will take the dir of the checked out git repo.
# So you can also only use it there...
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)

        os.environ["SIS_GLOBAL_SETTINGS_FILE"] = f"{_setup_base_dir}/settings.py"

        try:
            import sisyphus  # noqa
            import i6_experiments  # noqa
        except ImportError:
            print("setup base dir:", _setup_base_dir)
            print("sys.path:")
            for path in sys.path:
                print(f"  {path}")
            raise


_setup()

from sisyphus.loader import config_manager
from sisyphus import toolkit, Path


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_files", nargs="*")
    arg_parser.add_argument("--target")
    args = arg_parser.parse_args()

    start = time.time()
    config_manager.load_configs(args.config_files)
    load_time = time.time() - start
    logging.info("Config loaded (time needed: %.2f)" % load_time)

    sis_graph = toolkit.sis_graph
    if not args.target:
        print("--target not specified, printing all targets:")
        for name, target in sis_graph.targets_dict.items():
            print(f"Target: {name} -> {target.required_full_list}")
        sys.exit(0)
    
    target = sis_graph.targets_dict[args.target]
    print(f"Target: {args.target} -> {target.required_full_list}")
    path, = target.required_full_list  # assume only one output path
    assert isinstance(path, Path)
    assert not path.hash_overwrite

    # if name == "2024-denoising-lm/error_correction_model/base-puttingItTogether(low)-nEp200/recog-ext/dlm_sum_score_results.txt":
    # path = target.required_full_list[0]
    # assert not path.hash_overwrite
    # print("Job id:", path.creator._sis_id())

    # job sis hash is the job sis_id, which is cached.
    # sis_id: via sis_hash = cls._sis_hash_static(parsed_args)

    # idea: use script to dump hash reconstruction
    # always call sis_hash_helper with settrace to detect recursive calls to sis_hash_helper
    # dump always path, object_type -> hash, starting from target, where path == "/"
    # then recursively for all dependencies, adding path as "/" + number + object_type or so when going down.


if __name__ == "__main__":
    main()
