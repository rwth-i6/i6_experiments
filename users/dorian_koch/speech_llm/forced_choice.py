"""Forced-choice knowledge scoring of a PersonaPlex model (D5a; ``moshi_family.rl.forced_choice``)."""

from __future__ import annotations

import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import worker_env


class KnowledgeForcedChoice(Job):
    """Every question of a prepared knowledge set (``questions``: an ``LLMPreprocess`` out_hf; ``audio``: its
    ``ChatterboxSingleSpeakerInference`` tts_output, storage hf) heard by the model, which is then made to say the gold
    answer and the gold answers of ``n_distractors`` other questions of the set; candidates ranked by log-probability
    with the speak-or-stay-silent decision taken out (``moshi_family.rl.forced_choice`` docstring). ``overlay`` (+
    ``lora_rank``): a PersonaPlex finetune (None = released). Outputs ``scores.jsonl`` (per question) and
    ``summary.json`` (acc, acc_norm, margin, cap rate)."""

    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
        hf_repo: str,
        questions: tk.Path,
        audio: tk.Path,
        overlay: tk.Path | None = None,
        lora_rank: int | None = None,
        n_distractors: int = 4,
        cap_s: float = 2.0,
        seed: int = 0,
        limit: int | None = None,
        batch_size: int = 40,
    ):
        assert batch_size % (n_distractors + 1) == 0, (batch_size, n_distractors)
        self.venv_python_path = venv_python_path
        self.hf_repo = hf_repo
        self.questions = questions
        self.audio = audio
        self.overlay = overlay
        self.lora_rank = lora_rank
        self.n_distractors = n_distractors
        self.cap_s = cap_s
        self.seed = seed
        self.limit = limit
        self.batch_size = batch_size
        self.out_scores = self.output_path("scores.jsonl")
        self.out_summary = self.output_path("summary.json")

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 48, "time": 4, "gpu_mem_gb": 40})

    def run(self):
        out = os.path.dirname(self.out_summary.get_path())
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.rl.forced_choice"]
        cmd += ["--hf_repo", self.hf_repo, "--questions", self.questions.get_path(), "--audio", self.audio.get_path()]
        cmd += ["--out", out, "--n_distractors", str(self.n_distractors), "--cap_s", str(self.cap_s)]
        cmd += ["--seed", str(self.seed), "--batch_size", str(self.batch_size)]
        if self.overlay is not None:
            cmd += ["--overlay", self.overlay.get_path()]
            if self.lora_rank is not None:
                cmd += ["--lora_rank", str(self.lora_rank)]
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=worker_env(offline=True), check=True)
        s = json.load(open(self.out_summary.get_path()))
        print(f"[forced-choice] acc {s['acc']:.3f} acc_norm {s['acc_norm']:.3f} (chance {s['chance']:.2f}, n {s['n']})")

    def info(self):
        """The driver's last progress line (``[forced-choice] k/n ... acc ...``)."""
        log = os.path.join(self._sis_path(), "log.run.1")
        try:
            lines = [ln for ln in open(log, errors="replace") if ln.startswith("[forced-choice]")]
        except OSError:
            return None
        return lines[-1].strip()[:160] if lines else None
