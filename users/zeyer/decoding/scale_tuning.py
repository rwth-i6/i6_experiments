"""
Job/code for scale tuning
"""


from __future__ import annotations
from typing import Optional, Dict, Set
import subprocess

import os
from sisyphus import Job, Task, tk
import i6_core.util as util


class ScaleTuningJob(Job):
    """
    Tunes some scales
    """

    __sis_version__ = 2
    __sis_hash_exclude__ = {"max_scales": None}

    def __init__(
        self,
        scores: Dict[str, tk.Path],
        *,
        evaluation: str,
        ref: Optional[tk.Path] = None,
        fixed_scales: Optional[Dict[str, float]] = None,
        negative_scales: Optional[Set[str]] = None,
        scale_relative_to: Optional[Dict[str, str]] = None,
        max_scales: Optional[Dict[str, float]] = None,
        returnn_python_exe: Optional[tk.Path] = None,
        returnn_root: Optional[tk.Path] = None,
    ):
        """
        :param scores: name -> scores. we expect same seq tags and same hyps for all.
            similar to what you would put into :class:`SearchCombineScoresJob` (but with scales instead of names).
            The names are what is being used in the output.
            The path contain hyps in dict format (like in :class:`RecogOutput`).
        :param evaluation: evaluation method. e.g. "edit_distance"
        :param ref: reference targets, needed for some (most) evaluation methods
        :param fixed_scales: name -> scale. if given, the scale will be fixed to this value.
        :param scale_relative_to: name -> other name. if given, the scale will be relative to the other scale.
            Example: scores={"am": ..., "prior": ..., "lm": ...}, scale_relative_to={"prior": "lm"},
            then we will return "prior_rel" as the scale, where prior_scale = lm_scale * prior_scale_rel.
            Also, the tuning itself will be done on the relative scale.
            This can make the tuning more stable.
        :param returnn_python_exe:
        :param returnn_root:
        """
        super().__init__()
        self.scores = scores
        self.evaluation = evaluation
        self.ref = ref
        self.fixed_scales = fixed_scales
        self.negative_scales = negative_scales
        self.scale_relative_to = scale_relative_to
        self.max_scales = max_scales
        self.returnn_python_exe = returnn_python_exe
        self.returnn_root = returnn_root

        self.out_scales = self.output_var("scales.txt")
        self.out_real_scales = self.output_var("real_scales.txt")
        self.out_real_scale_per_name = {name: self.output_var(f"real_scale_{name}.txt") for name in scores}
        self.out_grid_plot = self.output_path("grid_plot.0.pdf") if len(scores) - len(fixed_scales) == 2 else None

        self.rqmt = {"gpu": 0, "cpu": 4, "mem": 12, "time": 4}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        returnn_python_exe = util.get_returnn_python_exe(self.returnn_python_exe)
        returnn_root = util.get_returnn_root(self.returnn_root)
        cmd = [returnn_python_exe.get_path(), returnn_root.get_path() + "/tools/torch_scale_tuning.py"]
        cmd += sum([[name] for name, path in self.scores.items()], ["--names"])
        cmd += sum([[path.get_path()] for name, path in self.scores.items()], ["--scores"])
        cmd += ["--evaluation", self.evaluation]
        if self.ref is not None:
            cmd += ["--ref", self.ref.get_path()]
        if self.fixed_scales:
            cmd += sum(([name, str(scale)] for name, scale in self.fixed_scales.items()), ["--fixed-scales"])
        if self.negative_scales:
            cmd += ["--negative-scales"] + sorted(self.negative_scales)
        if self.scale_relative_to is not None:
            cmd += sum((list(item) for item in self.scale_relative_to.items()), ["--scale-relative-to"])
        if self.max_scales is not None:
            cmd += sum(([name, str(scale)] for name, scale in self.max_scales.items()), ["--max-scales"])
        if self.rqmt.get("gpu", 0) > 0:
            cmd += ["--device", "cuda"]
        else:
            cmd += ["--device", "cpu"]
        cmd += ["--output-scales", self.out_scales.get_path()]
        cmd += ["--output-real-scales", self.out_real_scales.get_path()]
        if self.out_grid_plot:
            cmd += ["--output-grid-plot", self.out_grid_plot.get_path()[: -len(".0.pdf")]]

        # For easier debugging, write the command to a shell script.
        with open("run.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("set -e\n")
            f.write(" ".join(cmd))
            f.write("\n")

        print("$", " ".join(cmd))
        subprocess.check_call(cmd)

        assert os.path.exists(self.out_scales.get_path())
        assert os.path.exists(self.out_real_scales.get_path())
        if self.out_grid_plot:
            assert os.path.exists(self.out_grid_plot.get_path())

        real_scales = eval(open(self.out_real_scales.get_path()).read())
        assert isinstance(real_scales, dict) and set(real_scales.keys()) == set(self.scores.keys()) == set(
            self.out_real_scale_per_name.keys()
        )
        for name, scale in real_scales.items():
            self.out_real_scale_per_name[name].set(scale)
