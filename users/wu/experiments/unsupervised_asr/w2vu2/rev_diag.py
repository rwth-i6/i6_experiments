"""SAE 4a step 4 -- the DIAGNOSTIC readout of one arm's trained reverse model phi, as a sisyphus job.

Same env split as ``W2vu2RevDevJob``: the worker runs under the speech_llm python and spawns the GPU
forward under the ``w2vu`` python.  The numbers, their conventions and the self-checks are
``rev_diag_worker.py``'s docstring -- it asks how well phi predicts the K = 500 unit stream along
the lattice's own Viterbi path, along the greedy phone path, under a deranged recognizer posterior
and under the GOLD phone string, against the split's majority/unigram/uniform floors.

The job draws NO conclusion and defines no gate; it writes ``rev_diag.json`` and ``rev_diag.txt``.
``expect_logz_per_frame`` is the arm's own registered ``rev_dev.json`` ``own.logz_per_frame`` per
split: the worker asserts it reproduces it, which is what ties this readout to the banked one.
"""

from __future__ import annotations

import json
import os
import subprocess as sp
from typing import Dict, Optional

from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.gan import REV_USER_DIR
from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_WORKER = os.path.join(os.path.dirname(__file__), "rev_diag_worker.py")


class W2vu2RevDiagJob(Job):
    """Unit-prediction diagnostics of one checkpoint's own phi on dev-clean and dev-other."""

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
        expect_logz_per_frame: Optional[Dict[str, float]] = None,
        expect_tol: float = 1e-3,
        derange_seed: int = 0,
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
        self.expect_logz_per_frame = dict(expect_logz_per_frame or {})
        self.expect_tol = expect_tol
        self.derange_seed = derange_seed
        self.batch = batch
        self.python_exe = python_exe

        self.out_diag = self.output_path("rev_diag.json")
        self.out_report = self.output_path("rev_diag.txt")
        # W2vu2RevDevJob.NcFwMKvzqadV (the same 5567 dev utterances, two phis, batch 8) used 0.13 h
        # of GPU.  This job runs two forward passes of that lattice (own, deranged), two Viterbi
        # passes of it with a backtrace, and the gold forced alignment: ~6x that, so 2 h with
        # headroom on the same 80 GB GPU.
        self.rqmt = {"gpu": 1, "gpu_mem": 80, "mem": 32, "time": 2, "cpu": 4}

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
            "--derange-seed", str(self.derange_seed),
            "--expect", json.dumps(self.expect_logz_per_frame),
            "--expect-tol", str(self.expect_tol),
            "--batch", str(self.batch),
            "--out", self.out_diag.get_path(),
            "--out-txt", self.out_report.get_path(),
        ]
        print("RUN:", " ".join(args), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([recipe, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        sp.check_call(args, env=env)
