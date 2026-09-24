"""Synthetic knowledge dialogues as PersonaPlex training rows (ablation ab_epad1_know30, 2026-09-25).

User (2026-09-25): "mix in 30% in-batch of the synthetic knowledge dialogues. Think of something smart to make
them work with personaplex." PersonaPlex trains on rows that carry a HYBRID system prompt (voice + role text);
the TTS'd TriviaQA-train dialogues carry neither. ``KnowledgeDialogueCodes`` gives them both, the way the
PersonaPlex paper built its own QA data (arXiv 2602.06053, section 3.2):

  * role text: the paper's fixed QA role, verbatim -- "We use a fixed role for this dataset: 'You are a wise and
    friendly teacher. Answer questions or provide advice in a clear and engaging way.'" It is also the engine's
    DEFAULT_SYSTEM_PROMPT, i.e. the prompt the knowledge benchmark runs every PersonaPlex model with, so the
    knowledge rows teach "teacher prompt -> answer" while the JRE rows keep their podcast personas;
  * voice: the paper clones "a randomly chosen voice sample for each role" with Chatterbox and uses that sample as
    the agent's voice prompt. Our corpus clones ONE assistant voice (``pipelines.DEFAULT_SPEAKER_ALIAS``), so every
    knowledge row gets that reference clip as its voice prompt, encoded like the JRE voices. It is not cut from any
    dialogue, so it cannot leak a row's content;
  * audio + text: the JRE rows' conventions (loudness, offline Mimi encode) plus the A21 clamp that a codes
    corpus can only get at build time. See the worker ``moshi_family.knowledge_codes`` for each step.
"""

from __future__ import annotations

import json
import os
import subprocess

from sisyphus import Job, Task, tk

from .common import job_progress_fraction
from .podcast_ingest import _moshi_pythonpath

#: PersonaPlex's QA role, verbatim from the paper (section 3.2.1) and ``personaplex.engine.DEFAULT_SYSTEM_PROMPT``.
PERSONAPLEX_QA_ROLE = (
    "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."
)


class KnowledgeDialogueCodes(Job):
    """TTS'd knowledge dialogues (``MoshiAnnotate`` waveform corpora) -> one PersonaPlex codes corpus.

    ``corpora``: ``[(name, annotated dataset dir)]``, concatenated in this order (every row of each; for the v6
    corpus the per-template weights are the slices' row shares, so all rows = the merged corpus). ``voice_dir`` +
    ``voice_alias``: the TTS speaker dir and the assistant's alias (``<dir>/<alias>.wav``, the clip Chatterbox
    cloned). ``system_prompt``: the role text of every row. ``loudness_norm``: the ingest's dict
    (``synthetic_train_data.PODCAST_INGEST["loudness_norm"]``). Output ``out_dir`` (dataset) + ``out_summary``.
    """

    def __init__(
        self,
        *,
        corpora: list,
        voice_dir: tk.Path,
        voice_alias: str,
        venv_python_path: tk.Path,
        loudness_norm: dict,
        system_prompt: str = PERSONAPLEX_QA_ROLE,
        voice_sec: float = 4.0,
        max_sec: float = 60.0,
    ):
        names = [n for n, _ in corpora]
        assert len(set(names)) == len(names), f"corpus names must be unique: {names}"
        self.corpora = [(str(n), p) for n, p in corpora]
        self.voice_dir = voice_dir
        self.voice_alias = str(voice_alias)
        self.venv_python_path = venv_python_path
        self.loudness_norm = dict(loudness_norm)
        self.system_prompt = str(system_prompt)
        self.voice_sec = float(voice_sec)
        self.max_sec = float(max_sec)
        self.out_dir = self.output_path("dataset", directory=True)
        self.out_summary = self.output_path("summary.json")
        # One GPU for the offline Mimi encode (~327 h of stereo audio for the v6 corpus); decode, loudness and the
        # clamp run in CPU threads beside it. Small GPU memory: Mimi alone.
        self.rqmt = {"gpu": 1, "cpu": 8, "mem": 48, "time": 8, "gpu_mem_gb": 20}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def completed_fraction(self):
        return job_progress_fraction(self)

    def info(self):
        f = job_progress_fraction(self)
        return None if f is None else f"{100 * f:.1f}% of rows encoded"

    def run(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        voice_wav = os.path.join(self.voice_dir.get_path(), self.voice_alias + ".wav")
        if not os.path.isfile(voice_wav):
            raise RuntimeError(f"voice clip not found: {voice_wav}")
        cmd = [
            self.venv_python_path.get(),
            "-m",
            "moshi_family.knowledge_codes",
            "--out",
            self.out_dir.get_path(),
            "--voice_wav",
            voice_wav,
            "--voice_sec",
            str(self.voice_sec),
            "--system_prompt",
            self.system_prompt,
            "--loudness_norm",
            json.dumps(self.loudness_norm, sort_keys=True),
            "--max_sec",
            str(self.max_sec),
            "--device",
            "cuda",
            "--progress",
            "progress.json",
        ] + [f"{n}={p.get_path()}" for n, p in self.corpora]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
        # the worker writes summary.json beside the dataset dir, i.e. into output/
        if not os.path.isfile(self.out_summary.get_path()):
            raise RuntimeError(f"worker wrote no {self.out_summary.get_path()}")
