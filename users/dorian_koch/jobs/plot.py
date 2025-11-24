import os
import shutil
import subprocess as sp
import tempfile
import collections
import math
import re
from typing import List, Dict, Tuple

from sisyphus import tk, Job, Task, gs
from sisyphus.delayed_ops import DelayedBase

# from i6_core.lib.corpus import *
from i6_core.util import instanciate_delayed, uopen
import i6_core.util as util
import numpy as np


def _plot_grid(
    evals: List[Tuple[float, List[float]]],
    *,
    scale_indices: List[int],
    title: str,
    cbar_label: str,
    y_axis_name: str,
    x_axis_name: str,
    out_plot_filename: str,
):
    # noinspection PyPackageRequirements
    import matplotlib.pyplot as plt

    # noinspection PyPackageRequirements
    import matplotlib.ticker as ticker

    results = {}  # (x,y) -> z
    for eval_, scales in evals:
        results[tuple([scales[i] for i in scale_indices])] = eval_
    xs = sorted(set(scales[scale_indices[0]] for _, scales in evals))
    ys = sorted(set(scales[scale_indices[1]] for _, scales in evals))

    plt.figure(figsize=(8, 8))

    zs = np.zeros((len(ys), len(xs)))
    for y_idx, y in enumerate(ys):
        for x_idx, x in enumerate(xs):
            zs[y_idx, x_idx] = results[(x, y)]

    best = np.min(zs.flatten())
    worst_limit = best * 1.3

    ax = plt.subplot(1, 1, 1)
    plt.contourf(xs, ys, zs, levels=np.geomspace(best, worst_limit, 30))

    ax.set_title(title)
    ax.set_ylabel(y_axis_name)
    ax.set_xlabel(x_axis_name)
    ax.xaxis.set_major_locator(ticker.AutoLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_major_locator(ticker.AutoLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    cbar = plt.colorbar()
    cbar.set_label(cbar_label)

    print("Saving plot to", out_plot_filename)
    plt.savefig(out_plot_filename)


class ScaleTuningPlotJob(Job):
    def __init__(
        self,
        metric_evals: list[tuple[float | DelayedBase, list[float | DelayedBase] | dict[str, float | DelayedBase]]],
        *,
        axis: list[tuple[int, str]] | None = None,
        metric_name: str = "Metric",
    ):
        if axis is None:
            axis = [(0, "Scale 0"), (1, "Scale 1")]
        self.metric_name = metric_name
        self.metric_evals = metric_evals
        self.out_plot = self.output_path("scale_tuning_plot.pdf")
        self.out_minimum = self.output_var("minimum")
        self.out_maximum = self.output_var("maximum")
        self.axis = axis

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        # deep resolve delayed
        nums = instanciate_delayed(self.metric_evals)
        _plot_grid(
            nums,
            scale_indices=[i for i, _ in self.axis],
            title=f"{self.metric_name} Tuning",
            cbar_label=self.metric_name,
            y_axis_name=self.axis[1][1],
            x_axis_name=self.axis[0][1],
            out_plot_filename=self.out_plot.get_path(),
        )

        nums = sorted(nums, key=lambda x: x[0])
        # print all
        for num in nums:
            print(num)

        minimum = min(nums, key=lambda x: x[0])
        maximum = max(nums, key=lambda x: x[0])
        with uopen(self.out_minimum.get_path(), "w") as f:
            f.write(str(minimum))
        with uopen(self.out_maximum.get_path(), "w") as f:
            f.write(str(maximum))

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)
