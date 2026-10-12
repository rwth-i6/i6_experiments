"""Text-only knowledge probe of a Moshi-family LM (``moshi_family.text_mode_probe``; literature_2026-10.md pick 1)."""

from __future__ import annotations

import subprocess

from sisyphus import Job, Task, tk

from .common import worker_env


class TextModeKnowledgeProbe(Job):
    """``questions`` (a knowledge-benchmark ``transcriptions.jsonl``: index, question, answer, aliases) answered by the
    LM in TEXT-ONLY mode (all audio codebooks = the zero token), 5-shot, greedy. ``family`` personaplex / moshi;
    ``overlay`` (+ ``lora_rank``) a PersonaPlex finetune. Output ``answers.jsonl`` in the transcription schema, for
    ``LLMGrading``. Gap-ladder knobs (2026-10-12, hash-excluded at their defaults): ``prompt`` "persona" (PersonaPlex's
    system prompt + the question text, no shots), ``temp`` > 0 (sampled, top-k 25, ``seed``), ``max_new``."""

    __sis_hash_exclude__ = {"prompt": "fewshot", "temp": 0.0, "seed": 0, "max_new": None}

    def __init__(
        self,
        *,
        venv_python_path: tk.Path,
        family: str,
        hf_repo: str,
        questions: tk.Path,
        overlay: tk.Path | None = None,
        lora_rank: int | None = None,
        limit: int | None = None,
        prompt: str = "fewshot",
        temp: float = 0.0,
        seed: int = 0,
        max_new: int | None = None,
    ):
        assert prompt in ("fewshot", "persona") and (prompt == "fewshot" or family == "personaplex"), (prompt, family)
        assert family in ("personaplex", "moshi"), family
        assert overlay is None or family == "personaplex", "overlay: PersonaPlex only"
        self.venv_python_path = venv_python_path
        self.family = family
        self.hf_repo = hf_repo
        self.questions = questions
        self.overlay = overlay
        self.lora_rank = lora_rank
        self.limit = limit
        self.prompt = prompt
        self.temp = temp
        self.seed = seed
        self.max_new = max_new
        self.out_json = self.output_path("answers.jsonl")

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 48, "time": 2, "gpu_mem_gb": 40})

    def run(self):
        env = worker_env(offline=True)
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.text_mode_probe"]
        cmd += ["--family", self.family, "--hf_repo", self.hf_repo]
        cmd += ["--questions", self.questions.get_path(), "--out", self.out_json.get_path()]
        if self.overlay is not None:
            cmd += ["--overlay", self.overlay.get_path()]
            if self.lora_rank is not None:
                cmd += ["--lora_rank", str(self.lora_rank)]
        if self.limit is not None:
            cmd += ["--limit", str(self.limit)]
        cmd += ["--prompt", self.prompt, "--temp", str(self.temp), "--seed", str(self.seed)]
        if self.max_new is not None:
            cmd += ["--max_new", str(self.max_new)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
