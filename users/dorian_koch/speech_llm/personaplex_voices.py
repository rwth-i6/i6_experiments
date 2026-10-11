"""PersonaPlex voice-prompt embeddings from wav clips (``moshi_family.personaplex.voice_embeddings``)."""

from __future__ import annotations

import os
import subprocess

from sisyphus import Job, Task, tk

from .podcast_ingest import _moshi_pythonpath


class PersonaPlexVoiceEmbeddings(Job):
    """``wavs`` ``{name: wav path}`` -> ``out_dir/<name>.pt`` (+ the copied ``<name>.wav``, ``voices.json``): NVIDIA's
    released voice format, made with upstream's own save path at batch 1, so a custom voice works on the batched
    engine (a ``.wav`` voice prompt fails at B > 1; module docstring of ``voice_embeddings``)."""

    def __init__(self, *, wavs: dict, venv_python_path: tk.Path, hf_repo: str = "nvidia/personaplex-7b-v1"):
        assert wavs and all(isinstance(k, str) and "=" not in k for k in wavs), wavs
        self.wavs = dict(sorted(wavs.items()))
        self.venv_python_path = venv_python_path
        self.hf_repo = hf_repo
        self.out_dir = self.output_path("voices", directory=True)

    def tasks(self):
        yield Task("run", rqmt={"gpu": 1, "cpu": 4, "mem": 48, "time": 1, "gpu_mem_gb": 40})

    def run(self):
        env = dict(os.environ)
        env["PYTHONPATH"] = _moshi_pythonpath() + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.setdefault("HF_HUB_OFFLINE", "1")
        cmd = [self.venv_python_path.get(), "-u", "-m", "moshi_family.personaplex.voice_embeddings"]
        cmd += ["--out_dir", self.out_dir.get_path(), "--hf_repo", self.hf_repo]
        for name, path in self.wavs.items():
            cmd += ["--wav", f"{name}={tk.uncached_path(path)}"]
        print(" ".join(cmd), flush=True)
        subprocess.run(cmd, env=env, check=True)
