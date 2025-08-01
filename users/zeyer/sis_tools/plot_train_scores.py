#!/usr/bin/env python3

from __future__ import annotations
import os
import sys
import numpy as np
from glob import glob
from argparse import ArgumentParser
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


# for parsing the LR (train score) file
# noinspection PyPep8Naming
def EpochData(learningRate, error):
    return {"learning_rate": learningRate, "error": error}


def main():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("exps", nargs="*")
    arg_parser.add_argument("--out", default="plot.svg")
    arg_parser.add_argument("--xlog", action="store_true")
    arg_parser.add_argument("--ylog", action="store_true")
    arg_parser.add_argument("--score-key")
    args = arg_parser.parse_args()

    data = {}  # name -> data
    max_epoch = -1
    score_key = args.score_key
    covered = set()

    for exps in args.exps:
        pattern = exps + "*"
        dirs = glob(pattern)
        if not dirs:
            print(f"WARN: no experiments found matching pattern: {pattern}")
            sys.exit(1)
        for d in dirs:
            name = os.path.basename(d)
            if not os.path.isdir(d):
                continue
            if os.path.isdir(d + "/train"):
                d += "/train"
            d_ = os.path.realpath(d)
            if d_ in covered:
                continue
            covered.add(d_)
            for postfix in ["work/learning_rates", "output/learning_rates"]:
                if os.path.exists(d + "/" + postfix):
                    print(f"found data for {name}")
                    with open(d + "/" + postfix, "rt") as f:
                        text = f.read()
                    data_ = eval(text, {"EpochData": EpochData, "nan": float("nan"), "inf": float("inf"), "np": np})
                    data[name] = data_
                    max_epoch = max(max_epoch, max(data_.keys()))
                    score_keys = [k for k in next(iter(data_.values()))["error"].keys()]
                    if not score_key:
                        score_keys = [k for k in score_keys if k.startswith("dev_")]
                        score_key = next(iter(score_keys))
                        print(f"using score key {score_key} for {name}")
                    else:
                        assert score_key in score_keys, f"score key {score_key} not found in {name} data: {score_keys}"

                    break

    if not data:
        print(f"ERR: no data found in any of the directories: {args.exps}")
        sys.exit(1)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    if args.xlog:
        ax1.set_xscale("log")
    if args.ylog:
        ax1.set_yscale("log")
    for name, data_ in data.items():
        epochs = sorted(data_.keys())
        scores = [data_[epoch]["error"][score_key] for epoch in epochs]
        ax1.plot(epochs, scores, label=name)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("scores")
    ax1.set_title(f"Training scores for {score_key}")
    ax1.legend(fontsize=8, loc="upper right")

    fig.savefig(fname=args.out)
    print(f"saved plot to {args.out}")


if __name__ == "__main__":
    main()
