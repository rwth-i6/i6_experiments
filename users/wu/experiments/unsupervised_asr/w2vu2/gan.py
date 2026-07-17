"""SAE §1c — the wav2vec-U 2.0 GAN, run as fairseq's reference implementation.

We do not reimplement the GAN. `fairseq-hydra-train` is invoked with fairseq's own `w2vu2.yaml`
plus the overrides below, so every documented paper-vs-code divergence (gradient-penalty norm taken
over the *time* axis, perplexity-space diversity `(V-PPL)/V`, mean-reduction smoothness, BCE with
flipped labels, 1:1 G/D alternation) is inherited from the reference rather than re-derived.

Overrides forced by our encoder (BEST-RQ 512-d @ 25 Hz vs wav2vec2-Large 1024-d @ 50 Hz):

  input_dim              1024 -> 512
  generator_stride          3 -> 2     25/2 = 12.5 Hz output; theirs is 50/3 = 16.7 Hz
  generator_kernel          9 -> 5     time-matched: 9 frames @50 Hz = 180 ms ~ 5 frames @25 Hz
  target_downsample_rate    2 -> 1     our .km is pre-aligned to the 25 Hz encoder rate (features.py)

The stride is the one deviation with published evidence behind it: Table 1 shows generator *output*
rate is what governs convergence (14-16.7 Hz converges; 25-28 Hz gives >100 PER), and our measured
phone rate is ~11.9 Hz (SAE_0a: ~2.1 frames/phone @25 Hz), so stride 2 -> 12.5 Hz is closer to
ground truth than the paper's own 16.7 Hz. Stride 1 (=25 Hz) sits exactly on their divergent
configuration and is expected to fail; it is kept only as a labelled negative control.

`generator_batch_norm` (gamma init) is the single item the paper calls "critical for convergence"
(30 for wav2vec2-Large, 35 for XLSR-53 -- i.e. it tracks the feature distribution, so it must be
swept for BEST-RQ rather than inherited; SAE_PLAN §1c says {20, 30, 40}).
"""

from __future__ import annotations

import os
import subprocess as sp
from typing import Any, Dict

from sisyphus import Job, Task, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2vu2.text import W2VU_PYTHON, assert_w2vu_env

_alias_prefix = "sae/1c"


def w2vu2_overrides(
    *,
    input_dim: int = 512,
    generator_stride: int = 2,
    generator_kernel: int = 5,
    generator_batch_norm: int = 30,
    target_downsample_rate: int = 1,
    gradient_penalty: float = 1.0,
    smoothness_weight: float = 1.5,
    code_penalty: float = 3.0,
    mmi_weight: float = 0.5,
    seed: int = 0,
    max_update: int = 150_000,
) -> Dict[str, Any]:
    """The (model, optimization, checkpoint) knobs we set; everything else stays at w2vu2.yaml."""
    return {
        "model.input_dim": input_dim,
        "model.generator_stride": generator_stride,
        "model.generator_kernel": generator_kernel,
        "model.generator_batch_norm": generator_batch_norm,
        "model.target_downsample_rate": target_downsample_rate,
        "model.gradient_penalty": gradient_penalty,
        "model.smoothness_weight": smoothness_weight,
        "model.code_penalty": code_penalty,
        "model.mmi_weight": mmi_weight,
        "common.seed": seed,
        "optimization.max_update": max_update,
    }


class FairseqW2vu2TrainJob(Job):
    """One GAN run = fairseq-hydra-train, spawned under the `w2vu` env python (py3.9 + torch 2.6+cu126).

    worker_wrapper always runs the worker itself under the speech_llm python, so fairseq is an explicit
    subprocess; `requires_env` is what stops it inheriting speech_llm's LD_LIBRARY_PATH.
    """

    requires_env = "w2vu"  # class attr -> not an __init__ arg -> not hashed (settings.py::worker_wrapper)

    def __init__(
        self,
        *,
        data_dir: tk.Path,
        text_data: tk.Path,
        kenlm_path: tk.Path,
        overrides: Dict[str, Any],
        aux_target_postfix: str = "km",
        python_exe: tk.Path = W2VU_PYTHON,
        time_rqmt: float = 11.5,
        gpu_mem: int = 80,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.text_data = text_data
        self.kenlm_path = kenlm_path
        self.overrides = dict(overrides)
        self.aux_target_postfix = aux_target_postfix
        self.python_exe = python_exe

        self.out_dir = self.output_path("train", directory=True)
        self.out_best = self.output_path("train/checkpoint_best.pt")
        self.out_last = self.output_path("train/checkpoint_last.pt")
        self.out_log = self.output_path("train.log")
        self.rqmt = {"gpu": 1, "gpu_mem": gpu_mem, "mem": 60, "time": time_rqmt, "cpu": 8}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def _fairseq_dir(self) -> str:
        out = sp.check_output(
            [os.fspath(self.python_exe), "-c", "import fairseq, os; print(os.path.dirname(fairseq.__file__))"],
            text=True,
        )
        return out.strip()

    def run(self):
        assert_w2vu_env(self.python_exe)
        fs = self._fairseq_dir()
        cfg_dir = os.path.join(fs, "examples", "wav2vec", "unsupervised", "config", "gan")
        user_dir = os.path.join(fs, "examples", "wav2vec", "unsupervised")

        args = [
            os.fspath(self.python_exe), "-m", "fairseq_cli.hydra_train",
            "-m", f"--config-dir={cfg_dir}", "--config-name=w2vu2",
            f"task.data={self.data_dir.get_path()}",
            f"task.text_data={self.text_data.get_path()}",
            f"task.kenlm_path={self.kenlm_path.get_path()}",
            f"task.aux_target_postfix={self.aux_target_postfix}",
            f"common.user_dir={user_dir}",
            f"checkpoint.save_dir={self.out_dir.get_path()}",
            "distributed_training.distributed_world_size=1",
        ]
        args += [f"{k}={v}" for k, v in sorted(self.overrides.items())]
        print("RUN:", " ".join(args), flush=True)
        with open(self.out_log.get_path(), "w") as log:
            sp.check_call(args, stdout=log, stderr=sp.STDOUT)
