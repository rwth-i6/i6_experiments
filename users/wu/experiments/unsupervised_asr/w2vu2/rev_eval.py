"""SAE §4a step 4 -- the dev reverse-likelihood readout as a sisyphus job.

Same env split as ``W2vu2PerEvalJob``: the worker runs under the speech_llm python and spawns the
GPU forward under the ``w2vu`` python.  The numbers and their convention are ``rev_dev.py``'s.
"""

from __future__ import annotations

import json
import os
import subprocess as sp
from typing import Optional

from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.gan import REV_USER_DIR
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_WORKER = os.path.join(os.path.dirname(__file__), "rev_dev.py")


class W2vu2RevDevJob(Job):
    """Dev reverse log-likelihood per retained frame of one checkpoint, under its phi and a cold phi."""

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        checkpoint: tk.Path,
        data_dir: tk.Path,
        text_data: tk.Path,
        feats_dir: tk.Path,
        rev_units_dir: tk.Path,
        gold: tk.Path,
        cold_seed: Optional[int] = None,   # None = the arm's own model.rev_phi_seed
        batch: int = 8,
        python_exe: tk.Path = W2VU_PYTHON,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.text_data = text_data
        self.feats_dir = feats_dir
        self.rev_units_dir = rev_units_dir
        self.gold = gold
        self.cold_seed = cold_seed
        self.batch = batch
        self.python_exe = python_exe

        self.out_rev = self.output_path("rev_dev.json")
        self.rqmt = {"gpu": 1, "gpu_mem": 80, "mem": 32, "time": 4, "cpu": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        assert_w2vu_env(self.python_exe)
        recipe = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
        args = [
            os.fspath(self.python_exe), _WORKER,
            "--ckpt", self.checkpoint.get_path(),
            "--data", self.data_dir.get_path(),
            "--text-data", self.text_data.get_path(),
            "--feats", os.path.join(self.feats_dir.get_path(), "valid.npy"),
            "--rev-units", self.rev_units_dir.get_path(),
            "--gold", self.gold.get_path(),
            "--user-dir", REV_USER_DIR,
            "--batch", str(self.batch),
            "--out", self.out_rev.get_path(),
        ]
        if self.cold_seed is not None:
            args += ["--cold-seed", str(self.cold_seed)]
        print("RUN:", " ".join(args), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([recipe, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        sp.check_call(args, env=env)

        with open(self.out_rev.get_path()) as fh:
            d = json.load(fh)
        print(json.dumps(d["splits"], indent=2), flush=True)
