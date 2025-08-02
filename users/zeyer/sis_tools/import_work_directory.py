#!/usr/bin/env python3

"""
Import some work directory by symlinking.

This is similar to ``tk.import_work_directory`` with the difference that it
symlinks any existing job directories, even when they are not finished.
"""

from __future__ import annotations
import sys
import os
from functools import reduce


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


import sisyphus.global_settings as gs  # noqa: E402
from sisyphus.graph import graph as sis_graph  # noqa: E402


def main():
    """When called directly"""
    import argparse
    import importlib
    from returnn.util import better_exchook

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_file", help="Sisyphus config file (entry point with `py` func)")
    arg_parser.add_argument("--import-work-dir", required=True, help="Directory to import from")
    arg_parser.add_argument("--no-dry-run", action="store_true", help="Actually create the symlinks")
    args = arg_parser.parse_args()
    better_exchook.install()

    # like sisyphus.loader.ConfigManager.load_config_file
    filename = args.config_file
    # maybe remove import path prefix such as "recipe/"
    for load_path in gs.IMPORT_PATHS:
        if load_path.endswith("/") and filename.startswith(load_path):
            filename = filename[len(load_path) :]
            break
    filename = filename.replace(os.path.sep, ".")  # allows to use tab completion for file selection
    assert all(part.isidentifier() for part in filename.split(".")), "Config name is invalid: %s" % filename
    module_name, function_name = filename.rsplit(".", 1)
    config = importlib.import_module(module_name)
    exp_func = getattr(config, function_name)  # the `py` func in the Sisyphus config file
    assert callable(exp_func)
    print(f"Calling {module_name}.{function_name} to build the Sisyphus job graph...")
    exp_func()

    gs.GRAPH_WORKER = 1  # for debugging? make deterministic...

    # See tk.import_work_directory.

    imported = set()

    def import_directory(job):
        # check for new inputs
        job._sis_runnable()
        if job in imported:
            return True
        # import work directory if job does not already exist
        local_path = job._sis_path()
        if not os.path.exists(local_path):
            import_path = os.path.join(args.import_work_dir, job._sis_id())
            if os.path.exists(import_path):
                print(f"existing import path: {import_path}")
                imported.add(job)

                # See Job._sis_import_from_dirs
                if args.no_dry_run:
                    os.makedirs(os.path.dirname(local_path), exist_ok=True)
                    os.symlink(import_path, local_path, target_is_directory=True)

        return True

    number_of_jobs = 0
    # run once before to unsure inputs are updated at least once
    sis_graph.for_all_nodes(import_directory, bottom_up=True)
    # run until no new jobs are added. This could be solved more efficient, but this is works...
    while number_of_jobs != len(sis_graph.jobs()):
        number_of_jobs = len(sis_graph.jobs())
        sis_graph.for_all_nodes(import_directory, bottom_up=True)

    if args.no_dry_run:
        print(f"Imported {len(imported)} jobs from {args.import_work_dir}.")
    else:
        print(f"Would import {len(imported)} jobs from {args.import_work_dir} (dry run).")


if __name__ == "__main__":
    main()
