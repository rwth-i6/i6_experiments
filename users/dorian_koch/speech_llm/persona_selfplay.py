"""Free-form self-play of PersonaPlex models: two instances converse, one per channel.

Each conversation takes one training window's two sides -- per side its own persona prompt (at a
chosen level) and its own speaker's voice prompt, from ``AttachPersonaPrompts(with_other_side=True)``
-- and lets two models talk for ``duration_sec``, each hearing the other's output codes. The worker
is ``moshi_family.personaplex.selfplay`` (see its docstring for the frame-level protocol); the output
is one arrow dataset of codes + text streams per conversation, in the podcast codes schema.
"""

from __future__ import annotations

import os
import subprocess

from sisyphus import Job, Task, tk

from .podcast_ingest import _moshi_pythonpath


class PersonaSelfPlay(Job):
    """``n`` seeded conversations of ``duration_sec`` between model A (the window's assistant side)
    and model B (the other side). ``overlay_*``: a resolved ``lora.safetensors`` (None = base
    PersonaPlex), with its ``lora_rank_*``. ``label_*`` name the models in the output rows."""

    def __init__(
        self,
        *,
        data: tk.Path,
        venv_python_path: tk.Path,
        n: int = 100,
        seed: int = 0,
        level: str = "topic",
        duration_sec: float = 60.0,
        overlay_a: tk.Path | None = None,
        overlay_b: tk.Path | None = None,
        lora_rank_a: int | None = None,
        lora_rank_b: int | None = None,
        label_a: str = "a",
        label_b: str = "b",
        hf_repo: str = "nvidia/personaplex-7b-v1",
    ):
        assert (overlay_a is None) == (lora_rank_a is None) and (overlay_b is None) == (lora_rank_b is None)
        self.data = data
        self.venv_python_path = venv_python_path
        self.n = int(n)
        self.seed = int(seed)
        self.level = level
        self.duration_sec = float(duration_sec)
        self.overlay_a, self.overlay_b = overlay_a, overlay_b
        self.lora_rank_a, self.lora_rank_b = lora_rank_a, lora_rank_b
        self.label_a, self.label_b = label_a, label_b
        self.hf_repo = hf_repo
        self.out_dir = self.output_path("dataset", directory=True)
        # Two 7B models in bf16 (~16 GB each) plus four mimi on one card.
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 6, "gpu_mem_gb": 80}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        import json

        from sisyphus import global_settings as gs

        try:
            d = json.load(open(os.path.join(self._sis_path(gs.JOB_WORK_DIR), "progress.json")))
            return min(d["done"] / d["total"], 1.0)
        except (OSError, ValueError, KeyError, ZeroDivisionError):
            return None

    def run(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.personaplex.selfplay",
            "--data",
            self.data.get_path(),
            "--out",
            self.out_dir.get_path(),
            "--n",
            str(self.n),
            "--seed",
            str(self.seed),
            "--level",
            self.level,
            "--duration_sec",
            str(self.duration_sec),
            "--hf_repo",
            self.hf_repo,
            "--label_a",
            self.label_a,
            "--label_b",
            self.label_b,
        ]
        for s, ov, rank in (("a", self.overlay_a, self.lora_rank_a), ("b", self.overlay_b, self.lora_rank_b)):
            if ov is not None:
                cmd += [f"--overlay_{s}", ov.get_path(), f"--lora_rank_{s}", str(rank)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
