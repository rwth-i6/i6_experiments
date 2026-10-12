"""PersonaPlex voice-prompt embeddings from wav clips (``moshi_family.personaplex.voice_embeddings``)."""

from __future__ import annotations

import subprocess

from sisyphus import Job, Task, tk

from .common import worker_env


class PersonaPlexVoiceEmbeddings(Job):
    """``wavs`` ``{name: wav path}`` and/or ``codes`` ``{name: codes dataset dir}`` -> ``out_dir/<name>.pt`` (+ the
    copied ``<name>.wav``, ``voices.json``): NVIDIA's released voice format, made with upstream's own save path at
    batch 1, so a custom voice works on the batched engine (a ``.wav`` voice prompt fails at B > 1; module docstring
    of ``voice_embeddings``). A wav is used WHOLE (upstream's path, no cut); ``codes`` replays a training dataset's
    stored voice prompt (``voice_codes``, e.g. ``KnowledgeDialogueCodes``: the 4 s / 50 frames training saw)."""

    __sis_hash_exclude__ = {"codes": None}

    def __init__(
        self,
        *,
        wavs: dict | None = None,
        codes: dict | None = None,
        venv_python_path: tk.Path,
        hf_repo: str = "nvidia/personaplex-7b-v1",
    ):
        names = [*(wavs or {}), *(codes or {})]
        assert names and len(set(names)) == len(names), (wavs, codes)
        assert all(isinstance(k, str) and "=" not in k for k in names), names
        self.wavs = dict(sorted((wavs or {}).items()))
        self.codes = dict(sorted(codes.items())) if codes else None
        self.venv_python_path = venv_python_path
        self.hf_repo = hf_repo
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 48, "time": 1, "gpu_mem_gb": 40})

    def run(self):
        env = worker_env(offline=True)
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.personaplex.voice_embeddings"]
        cmd += ["--out_dir", self.out_dir.get_path(), "--hf_repo", self.hf_repo]
        for name, path in self.wavs.items():
            cmd += ["--wav", f"{name}={tk.uncached_path(path)}"]
        for name, path in (self.codes or {}).items():
            cmd += ["--codes", f"{name}={tk.uncached_path(path)}"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
