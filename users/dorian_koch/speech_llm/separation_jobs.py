"""Sisyphus job: benchmark a speaker separator against dual-channel ground truth.

Runs ``separation.benchmark_main`` in the ``separation_venv`` over the durable Seamless audio
(``seamless_jobs.seamless_audio_dir`` -> ``audio/<fid>.flac`` + ``manifest.jsonl``): each interaction's two
per-speaker channels are summed to a mono mixture, separated, and the recovered channels scored against the
references with permutation-invariant SI-SDR / SI-SDRi (full-file + overlap-only).

Model-agnostic: ``separator`` selects the approach (``separation.separators.build_separator``); to benchmark
a new method later, register it there and pass its name -- this job is unchanged. HF gating (PixIT needs an
accepted token) is handled by Sisyphus injecting ``HF_TOKEN`` into the job env (see ``settings.py``), which
``run_worker_script`` forwards. See ``projects/2026-01-speech-llm/audio_datasets.md``.
"""

from __future__ import annotations

import os

from sisyphus import Job, Task, tk

from .common import job_progress_fraction, run_worker_script
from .inference_harness import _moshi_family_pythonpath


class SpeakerSeparationBenchmarkJob(Job):
    """Score one separator on a dual-channel corpus (mix -> separate -> PIT SI-SDR/SI-SDRi)."""

    __sis_hash_exclude__ = {"sample_rate": 16000, "num_examples": 3, "rqmt": None}

    def __init__(
        self,
        *,
        venv_python_path,
        audio_dir: tk.Path,
        separator: str = "pixit",
        max_interactions: int = 0,
        sample_rate: int = 16000,
        num_examples: int = 3,
        rqmt: dict | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.audio_dir = audio_dir
        self.separator = separator
        self.max_interactions = max_interactions
        self.sample_rate = sample_rate
        self.num_examples = num_examples
        self.out_dir = self.output_path("out", directory=True)
        self.rqmt = rqmt or {"gpu": 1, "cpu": 6, "mem": 32, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        # Worker writes {"done", "total"} to progress.json in its cwd (= this work dir) each interaction.
        return job_progress_fraction(self)

    def run(self):
        lib_parent = _moshi_family_pythonpath()
        script = os.path.join(lib_parent, "separation", "benchmark_main.py")
        args = [
            "--audio_dir",
            self.audio_dir.get(),
            "--out_dir",
            self.out_dir.get(),
            "--separator",
            self.separator,
            "--max_interactions",
            self.max_interactions,
            "--sample_rate",
            self.sample_rate,
            "--num_examples",
            self.num_examples,
        ]
        run_worker_script(
            self.venv_python_path.get(),
            script,
            args,
            log_label=f"Separation benchmark ({self.separator})",
            with_hf_home=True,  # forwards HF_HOME + the injected HF_TOKEN from the job env
            extra_env={"PYTHONPATH": lib_parent},
        )
