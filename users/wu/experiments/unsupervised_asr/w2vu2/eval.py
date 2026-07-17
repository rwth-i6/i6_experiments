"""SAE §1c — greedy-PER evaluation of trained wav2vec-U 2.0 generators.

Two jobs, split by which env they need:

  GoldPhonesJob      speech_llm (pandas + repr_audit), CPU, checkpoint-independent -> gold.json once.
  W2vu2PerEvalJob    w2vu env (torch + fairseq), GPU, per checkpoint -> per.json.

The decode is fairseq's own VITERBI path with its all-zero transition matrix, i.e. frame-wise argmax
+ CTC collapse + silence drop, done without flashlight. See eval_per.py for the faithfulness argument.
"""

from __future__ import annotations

import json
import os
import subprocess as sp
from typing import Dict, Sequence

from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_WORKER = os.path.join(os.path.dirname(__file__), "eval_per.py")


class GoldPhonesJob(Job):
    """dev gold phone sequences (sil-free, stress-free) as one json {split: {utt_id: [phones]}}.

    Checkpoint-independent, so it is computed once and shared by every PER eval. Runs under the
    default (speech_llm) env because the gilkeyio parquet read needs pandas, which is deliberately
    kept out of the numpy-pinned w2vu env.
    """

    def __init__(self, *, splits: Sequence[str] = ("dev-clean", "dev-other")):
        super().__init__()
        self.splits = list(splits)
        self.out_gold = self.output_path("gold.json")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # recipe root = five levels up from this file (.../recipe/i6_experiments/users/wu/experiments/
        # unsupervised_asr/w2vu2/eval.py) -> pass it to compute_gold for the repr_audit import.
        recipe = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 6))
        from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.eval_per import compute_gold

        gold = {s: compute_gold(s, recipe) for s in self.splits}
        for s, d in gold.items():
            assert d, f"no gold for split {s}"
        with open(self.out_gold.get_path(), "w") as f:
            json.dump(gold, f)


class W2vu2PerEvalJob(Job):
    """Greedy PER of one generator checkpoint on the dumped dev features, per split.

    worker_wrapper runs this worker under the speech_llm python (as always), so the GPU forward is an
    explicit subprocess under the w2vu python; `requires_env` is what grants it the w2vu env.
    """

    requires_env = "w2vu"

    def __init__(
        self,
        *,
        checkpoint: tk.Path,
        data_dir: tk.Path,
        text_data: tk.Path,
        feats_dir: tk.Path,     # MergeW2vu2DataJob-style dir holding valid.npy/.lengths/.ids
        gold: tk.Path,
        python_exe: tk.Path = W2VU_PYTHON,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.data_dir = data_dir
        self.text_data = text_data
        self.feats_dir = feats_dir
        self.gold = gold
        self.python_exe = python_exe

        self.out_per = self.output_path("per.json")
        self.rqmt = {"gpu": 1, "gpu_mem": 40, "mem": 24, "time": 2, "cpu": 4}

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
            "--gold", self.gold.get_path(),
            "--out", self.out_per.get_path(),
        ]
        print("RUN:", " ".join(args), flush=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([recipe, env.get("PYTHONPATH", "")]).strip(os.pathsep)
        sp.check_call(args, env=env)

        with open(self.out_per.get_path()) as f:
            print(json.dumps(json.load(f), indent=2), flush=True)
