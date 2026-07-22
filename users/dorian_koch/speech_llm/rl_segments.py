"""Sisyphus job: extract per-axis RL training segments from a lhotse ``CutSet`` (Fisher/Seamless).

Mirrors the other lib-worker jobs: it shells out to ``moshi_family.rl.extract_main`` in the
``moshi_family_venv`` (which has lhotse + silero-vad + torch), with the lib parent on ``PYTHONPATH`` via
the shared ``_moshi_family_pythonpath`` resolver. Output is a self-contained ``segments/`` dir
(materialised user-channel audio + ``index.jsonl``) consumed by ``moshi_family.rl.segments
.CorpusSegmentSource`` -- so it survives a ``tk.import_work_directory`` from a workdir that had raw-Fisher
read access (the train-time process never touches the permission-restricted corpus). See
``projects/2026-01-speech-llm/rl_alignment.md`` and ``moshi_family/rl/fisher_segments.py``.
"""

from __future__ import annotations

import os

from sisyphus import Job, Task, tk

from .common import run_worker_script
from .inference_harness import _moshi_family_pythonpath


class RLFisherSegments(Job):
    """Extract <=``per_axis_cap`` segments/axis (pause/turn/backchannel/interruption) from a cutset.

    ``cutset_manifest`` is a lhotse ``CutSet`` jsonl (e.g. ``get_fisher_original_cutset()``); ``MonoCut``
    user channel + turn supervisions are required. CPU-only (Silero VAD runs on CPU)."""

    __sis_hash_exclude__ = {"vad_backend": "silero", "sample_rate": 24000, "seed": 0, "rqmt": None}

    def __init__(
        self,
        *,
        venv_python_path,
        cutset_manifest: tk.Path,
        per_axis_cap: int = 2000,
        vad_backend: str = "silero",
        sample_rate: int = 24000,
        seed: int = 0,
        rqmt: dict | None = None,
    ):
        self.venv_python_path = venv_python_path
        self.cutset_manifest = cutset_manifest
        self.per_axis_cap = per_axis_cap
        self.vad_backend = vad_backend
        self.sample_rate = sample_rate
        self.seed = seed
        self.out_dir = self.output_path("segments", directory=True)
        self.rqmt = rqmt or {"cpu": 8, "mem": 32, "time": 12}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        lib_parent = _moshi_family_pythonpath()
        script = os.path.join(lib_parent, "moshi_family", "rl", "extract_main.py")
        args = [
            "--manifest",
            self.cutset_manifest.get(),
            "--out_dir",
            self.out_dir.get(),
            "--per_axis_cap",
            self.per_axis_cap,
            "--vad_backend",
            self.vad_backend,
            "--sample_rate",
            self.sample_rate,
            "--seed",
            self.seed,
        ]
        run_worker_script(
            self.venv_python_path.get(),
            script,
            args,
            log_label="RL Fisher segment extraction",
            with_hf_home=True,
            extra_env={"PYTHONPATH": lib_parent},
        )
