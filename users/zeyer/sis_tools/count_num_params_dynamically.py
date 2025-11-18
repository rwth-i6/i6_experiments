"""
Take some Sisyphus config,
collect training jobs,
check the RETURNN config,
build model
(but not on a real device, using some dummy device, e.g. Torch Meta device, to not consume memory),
and count the number of parameters dynamically.
"""

from __future__ import annotations
import os
import sys
import logging
import argparse
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


from sisyphus import tk
from sisyphus.loader import config_manager
from i6_core.returnn.training import ReturnnTrainingJob


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("config_files", nargs="*")
    args = arg_parser.parse_args()

    start = time.time()
    config_manager.load_configs(args.config_files)
    load_time = time.time() - start
    logging.info("Config loaded (time needed: %.2f)" % load_time)

    for job in tk.sis_graph.jobs():
        if not isinstance(job, ReturnnTrainingJob):
            continue
        if not job.get_aliases():
            print("Skipping training job without alias:", job)
            continue
        print(f"Training job: {job}")


if __name__ == "__main__":
    main()
