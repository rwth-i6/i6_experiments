"""Measure what the squashfs ``vllm_venv`` buys at vLLM startup, on a real allocation.

``common.vllm_server`` launched vLLM from the setup ``.venv`` (a plain directory on the shared filesystem) until
2026-10-01; it now prefers the squashfs ``vllm_venv`` image (``common.vllm_python``). This job times both
interpreters back to back on ONE GPU node, counterbalanced (round 0: sqsh then venv; round 1: venv then sqsh) because
whatever runs first also warms the shared weights: (a) a full ``vllm_server`` start to "ready" (two Python imports +
weights + compile + graph capture) -- round 0 is each interpreter's COLD start, what a real job pays -- and (b) a bare
``import vllm`` in a fresh process right after. Output ``results.json``: every timing, plus each
interpreter's vllm/torch versions (they must match -- the image is pinned to the ``.venv``'s stack).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from sisyphus import Job, Task, tk

from .common import vllm_gpu_mem_gb


class VllmStartupProbe(Job):
    def __init__(
        self, *, sqsh_python: tk.Variable, model: str = "google/gemma-3-27b-it", rounds: int = 2, compare: bool = True
    ):
        self.sqsh_python = sqsh_python
        self.model = model
        self.rounds = rounds
        # False = a serving smoke only: one cold start + one chat request from the image, no .venv comparison
        self.compare = compare
        self.out_results = self.output_path("results.json")
        self.rqmt = {"gpu": 1, "cpu": 6, "mem": 24, "time": 1, "gpu_mem_gb": vllm_gpu_mem_gb(model)}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        from .common import vllm_server

        pythons = {"sqsh": self.sqsh_python.get(), "venv": sys.executable}
        if not self.compare:
            pythons = {"sqsh": pythons["sqsh"]}
        out = {"model": self.model, "pythons": pythons, "versions": {}, "import_s": [], "server_ready_s": []}
        for name, py in pythons.items():
            v = subprocess.run(
                [py, "-c", "import importlib.metadata as m; print(m.version('vllm'), m.version('torch'))"],
                capture_output=True,
                text=True,
            )
            out["versions"][name] = v.stdout.strip() or v.stderr.strip()[-300:]
        for r in range(self.rounds):
            for name in (("sqsh", "venv") if r % 2 == 0 else ("venv", "sqsh")) if self.compare else ("sqsh",):
                # Server first: round 0 is then a COLD start for each interpreter, which is what a real job pays.
                os.environ["VLLM_PYTHON"] = pythons[name]
                t0 = time.time()
                with vllm_server(self.model, max_model_len=4096, gpu_memory_utilization=0.85) as url:
                    ready = time.time() - t0
                    # One real request through the same client path our judges use (incl. reasoning_effort).
                    from openai import OpenAI

                    resp = OpenAI(api_key="EMPTY", base_url=url).chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": [{"type": "text", "text": "Reply with the word OK."}]}],
                        max_tokens=64,
                        temperature=0.0,
                        reasoning_effort="low",
                    )
                    out.setdefault("reply", {})[f"{name}_{r}"] = resp.choices[0].message.content
                out["server_ready_s"].append({"round": r, "python": name, "s": round(ready, 1)})
                print(f"[probe] vllm_server ready via {name}: {ready:.1f} s", flush=True)
                t0 = time.time()
                subprocess.run([pythons[name], "-c", "import vllm"], check=True)
                out["import_s"].append({"round": r, "python": name, "s": round(time.time() - t0, 1)})
                print(f"[probe] import vllm via {name} (after its server): {out['import_s'][-1]['s']} s", flush=True)
            if not self.compare:
                break
        os.environ.pop("VLLM_PYTHON", None)
        with open(self.out_results.get(), "w") as f:
            json.dump(out, f, indent=2)
