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
        batch_size: int = 32,
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
        self.batch_size = int(batch_size)
        self.out_dir = self.output_path("dataset", directory=True)
        # Two 7B models in bf16 (~16 GB each), four mimi, and a KV cache per conversation sized by the
        # worker's context (conversation frames + 256; see selfplay --context). Measured 2026-09-22 at
        # 60 s: batch 32 -> 66.3 GiB peak, so it fits the 80 GB cards of either partition. (The full
        # 3000-frame context took 82.7 GiB at batch 16.) Outside the measured range, ask for more.
        big = self.batch_size > 32 or self.duration_sec > 60.0
        self.rqmt = {"gpu": 1, "cpu": 4, "mem": 32, "time": 6, "gpu_mem_gb": 94 if big else 80}

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
            "--batch_size",
            str(self.batch_size),
        ]
        for s, ov, rank in (("a", self.overlay_a, self.lora_rank_a), ("b", self.overlay_b, self.lora_rank_b)):
            if ov is not None:
                cmd += [f"--overlay_{s}", ov.get_path(), f"--lora_rank_{s}", str(rank)]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)


class SelfPlayTranscribe(Job):
    """Per-channel word-level ASR of self-play conversations, with the podcast corpus's OWN ASR.

    Decodes each conversation's two code streams to audio (``moshi_family.personaplex.selfplay_decode``)
    and transcribes each channel separately with ``moshi_family/podcast_asr.py --serve`` -- the same
    backend and model (faster_whisper/medium) and the same word cleaning that produced the JRE
    corpus's ``words_a``/``words_b``. So turn-taking statistics computed on this output and on the
    real windows come from the same ASR, not two. Output: the self-play dataset plus ``words_a`` /
    ``words_b`` (JSON lists of ``{text, start, end, ...}``, seconds from the recording start).
    """

    def __init__(
        self,
        *,
        selfplay_data: tk.Path,
        venv_python_path: tk.Path,
        asr_venv_python: tk.Path,
        asr_backend: str = "faster_whisper",
        asr_model: str = "medium",
        asr_batch_size: int = 16,
    ):
        self.selfplay_data = selfplay_data
        self.venv_python_path = venv_python_path
        self.asr_venv_python = asr_venv_python
        self.asr_backend = asr_backend
        self.asr_model = asr_model
        self.asr_batch_size = int(asr_batch_size)
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 32, "time": 2})

    def run(self):
        import json
        import shutil

        from datasets import load_from_disk

        env = dict(os.environ)
        pypath = _moshi_pythonpath()
        env["PYTHONPATH"] = pypath + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        wav_dir = os.path.abspath("wav_scratch")
        subprocess.run(
            [
                self.venv_python_path.get(),
                "-m",
                "moshi_family.personaplex.selfplay_decode",
                "--data",
                self.selfplay_data.get_path(),
                "--out_dir",
                wav_dir,
            ],
            env=env,
            check=True,
        )
        worker = os.path.join(pypath.split(os.pathsep)[0], "moshi_family", "podcast_asr.py")
        cmd = [
            self.asr_venv_python.get(),
            worker,
            "--backend",
            self.asr_backend,
            "--model",
            self.asr_model,
            "--batch_size",
            str(self.asr_batch_size),
            "--serve",
        ]
        print(" ".join(cmd), flush=True)
        ds = load_from_disk(self.selfplay_data.get_path())
        words = {}
        # One long-lived ASR process (model loaded once); requests one per line on stdin, one JSON
        # response per line on stdout, logging on stderr (see podcast_asr.py).
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)
        try:
            for i in range(len(ds)):
                for side in ("a", "b"):
                    req = {
                        "wav": os.path.join(wav_dir, f"{i}_{side}.wav"),
                        "out": os.path.join(wav_dir, f"{i}_{side}.json"),
                    }
                    proc.stdin.write(json.dumps(req) + "\n")
                    proc.stdin.flush()
                    resp = json.loads(proc.stdout.readline())
                    if not resp.get("ok"):
                        raise RuntimeError(f"ASR failed on row {i} side {side}: {resp}")
                    with open(req["out"]) as f:
                        words[(i, side)] = json.load(f)["words"]
                print(f"[selfplay-asr] {i + 1}/{len(ds)}", flush=True)
        finally:
            proc.stdin.close()
            proc.wait()
        out = ds.add_column("words_a", [json.dumps(words[(i, "a")]) for i in range(len(ds))])
        out = out.add_column("words_b", [json.dumps(words[(i, "b")]) for i in range(len(ds))])
        out.save_to_disk(self.out_dir.get_path())
        n_words = sum(len(v) for v in words.values())
        print(f"[selfplay-asr] {len(ds)} conversations, {n_words} words", flush=True)
        shutil.rmtree(wav_dir)  # scratch audio; the dataset keeps the codes


class CustomSelfPlayPrompts(Job):
    """A self-play input dataset from hand-written prompt PAIRS instead of LLM-described windows.

    ``pairs``: ``[(name, prompt_a, prompt_b), ...]`` -- one prompt per side (the paper's Minimal line
    is prepended). Each pair is used ``voice_pairs`` times, each time with the two speakers of a
    different seeded JRE window from ``voice_data`` (an ``AttachPersonaPrompts(with_other_side=True)``
    output), so a pair is heard in more than one voice combination. Rows carry the columns
    ``moshi_family.personaplex.selfplay`` reads, with the single level ``custom``.
    """

    def __init__(self, *, pairs: list, voice_data: tk.Path, voice_pairs: int = 2, seed: int = 0):
        self.pairs = [tuple(p) for p in pairs]
        assert len({p[0] for p in self.pairs}) == len(self.pairs), "pair names must be unique"
        self.voice_data = voice_data
        self.voice_pairs = int(voice_pairs)
        self.seed = int(seed)
        self.out_dir = self.output_path("dataset", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np
        from datasets import Dataset, load_from_disk

        from .persona_prompts import PERSONA_MINIMAL

        src = load_from_disk(self.voice_data.get_path())
        ok = [i for i, v in enumerate(src["other_ok"]) if v]
        rng = np.random.default_rng(self.seed)
        pick = rng.choice(ok, size=len(self.pairs) * self.voice_pairs, replace=False).tolist()
        cols = ["id", "voice_codes", "voice_codes_other", "voice_label", "voice_label_other"]
        rows = []
        for n, (name, pa, pb) in enumerate(self.pairs):
            for k in range(self.voice_pairs):
                v = src.select_columns(cols)[int(pick[n * self.voice_pairs + k])]
                rows.append(
                    {
                        "id": f"custom:{name}#{k}",
                        "voice_window": v["id"],
                        "other_ok": True,
                        "context_level": ["custom"],
                        "context": [f"{PERSONA_MINIMAL} {pa}"],
                        "context_other": [f"{PERSONA_MINIMAL} {pb}"],
                        "voice_codes": v["voice_codes"],
                        "voice_codes_other": v["voice_codes_other"],
                        "voice_label": v["voice_label"],
                        "voice_label_other": v["voice_label_other"],
                    }
                )
        Dataset.from_list(rows).save_to_disk(self.out_dir.get_path())
        print(
            f"[custom-prompts] {len(self.pairs)} pairs x {self.voice_pairs} voice pairs = {len(rows)} rows", flush=True
        )
