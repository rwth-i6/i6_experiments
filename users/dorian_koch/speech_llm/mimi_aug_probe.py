"""Backlog E19: does storing a corpus as MIMI CODES cost us waveform augmentation?

A permanent job rather than a one-off sbatch, deliberately: the /hpcwork one-off probes are covered
by nothing, and on 2026-09-18 all six of them turned out to have been silently broken for four days
by a venv path change nobody would have noticed until someone re-read a log. A statistic worth
asking for once is worth regenerating for free.

What it measures and why the ratio is the whole point: see the worker's module docstring
(`mimi_aug_probe_worker.py`). In one line -- augmentation can only survive code storage as
decode -> augment -> re-encode, that adds a codec round trip the waveform pipeline never pays, and
the design turns on whether the round trip moves the codes less than the augmentation does.
"""

from sisyphus import Job, Task, tk

from .common import (
    HF_CACHE_DIR,
    HF_HOME_DIR,
    add_cuda_npp_to_env,
    add_venv_python_lib_to_env,
    run_worker_script,
)
from .tts import InstallFFmpeg


class MimiAugmentationProbe(Job):
    """Encode/decode/augment round-trip costs and distances for one corpus."""

    def __init__(
        self,
        *,
        corpus: tk.Path,
        venv_python_path: tk.Path,
        n_rows: int = 64,
        seed: int = 0,
        duration_sec: float = 60.0,
        hf_repo: str = "kyutai/moshiko-pytorch-bf16",
        env_ffmpeg_path: tk.Path | None = None,
        code_version: int = 1,
    ):
        """``code_version`` is the lever for a worker change that is not in the hash otherwise --
        bump it to force a re-measure. ``env_ffmpeg_path`` travels with the job rather than being
        assumed present on the node (the standing rule; c23g is heterogeneous and a capability tag
        cannot express per-node library availability)."""
        self.corpus = corpus
        self.venv_python_path = venv_python_path
        self.n_rows = n_rows
        self.seed = seed
        self.duration_sec = duration_sec
        self.hf_repo = hf_repo
        self.env_ffmpeg_path = env_ffmpeg_path
        self.code_version = code_version

        self.out_json = self.output_path("mimi_aug_probe.json")
        self.out_report = self.output_path("report.txt")

        self.rqmt = {"time": 2, "cpu": 4, "mem": 32, "gpu": 1, "gpu_mem_gb": 40}

    @classmethod
    def hash(cls, parsed_args):
        # Where FFmpeg happens to live is a property of THIS cluster, not of the measurement. If it
        # reached the hash, supplying it would re-run the probe on every setup.
        d = dict(parsed_args)
        d.pop("env_ffmpeg_path", None)
        return super().hash(d)

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import os

        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mimi_aug_probe_worker.py")

        def env_hook(env):
            if self.env_ffmpeg_path is not None:
                InstallFFmpeg.add_to_env(self.env_ffmpeg_path, env)
            add_cuda_npp_to_env(self.venv_python_path.get(), env)
            add_venv_python_lib_to_env(self.venv_python_path.get(), env)

        run_worker_script(
            self.venv_python_path.get(),
            worker,
            [
                "--corpus",
                self.corpus.get(),
                "--out_json",
                self.out_json.get_path(),
                "--out_txt",
                self.out_report.get_path(),
                "--n_rows",
                self.n_rows,
                "--seed",
                self.seed,
                "--duration_sec",
                self.duration_sec,
                "--hf_repo",
                self.hf_repo,
            ],
            log_label="mimi_aug_probe",
            extra_env={"HF_HOME": HF_HOME_DIR.get(), "HF_HUB_CACHE": HF_CACHE_DIR.get()},
            env_hook=env_hook,
        )


def mimi_aug_probe_py(*, corpus, venv_python_path, tag: str, **kw):
    """Register the probe under ``output/mimi_aug_probe/<tag>/`` and return the job."""
    job = MimiAugmentationProbe(
        corpus=corpus,
        venv_python_path=venv_python_path,
        env_ffmpeg_path=InstallFFmpeg().out_path,
        **kw,
    )
    tk.register_output(f"mimi_aug_probe/{tag}/report", job.out_report)
    tk.register_output(f"mimi_aug_probe/{tag}/metrics", job.out_json)
    return job
