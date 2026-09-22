"""Guard: ``common.map_concurrent`` (the shared LLM-request fan-out) and MTRJudge, which uses it.

WHY. Two vLLM jobs kept too few requests in flight -- PersonaPromptGen 32 (5% KV-cache use) and
MTRJudge 1, a serial loop (0.3%) -- measured 2026-09-22. Both now fan out through ``map_concurrent``.
What must hold for that to be a pure speed change:

  1. ORDER. Results come back in input order although completions arrive out of order (the stub's
     random delays make them; asserted, so the order check is not vacuous).
  2. CONCURRENCY. The requested number of calls is really in flight at once (peak measured), and
     ``concurrency=1`` is serial.
  3. ERRORS. An exception in ``fn`` propagates, as it did from the serial loop.
  4. MTRJudge, driven for real with a stubbed vLLM server + OpenAI client: ``per_round`` is in row
     order with each row's own score, and the vLLM path runs MTR_JUDGE_CONCURRENCY calls at once.

Recipe venv, login node, a few seconds:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_map_concurrent.py
"""

import contextlib
import json
import os
import random
import sys
import tempfile
import threading
import time
import types

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")

from sisyphus import tk  # noqa: E402

import i6_experiments.users.dorian_koch.speech_llm.common as common  # noqa: E402
import i6_experiments.users.dorian_koch.speech_llm.mtr_duplexbench as mtr  # noqa: E402


class Probe:
    """Counts calls in flight; records completion order."""

    def __init__(self):
        self.lock, self.now, self.peak, self.finished = threading.Lock(), 0, 0, []

    def __call__(self, x):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(random.Random(x).uniform(0.0, 0.03))
        with self.lock:
            self.now -= 1
            self.finished.append(x)
        return x * x


# ---- 1 + 2 ----
items = list(range(200))
p = Probe()
got = common.map_concurrent(p, items, concurrency=50)
assert got == [x * x for x in items], "results not in input order"
assert p.finished != items, "completions arrived in order: the order check would be vacuous"
assert p.peak == 50, f"peak in flight {p.peak}, wanted 50"
p1 = Probe()
assert common.map_concurrent(p1, items[:20], concurrency=1) == [x * x for x in items[:20]]
assert p1.peak == 1, p1.peak
print(f"[1-2] 200 items: input order kept (completions were out of order), peak in flight {p.peak}; concurrency=1 is serial")

# ---- 3 ----
def boom(x):
    if x == 7:
        raise ValueError("boom 7")
    return x


try:
    common.map_concurrent(boom, range(20), concurrency=8)
    raise AssertionError("exception swallowed")
except ValueError as e:
    assert "boom 7" in str(e)
print("[3] an exception in fn propagates")

# ---- 4 ----
state = Probe()


class _Completions:
    def create(self, *, messages, **_kw):
        text = messages[-1]["content"]
        rid = int(text.split("ROW")[1].split()[0])
        state(rid)  # delay + in-flight count
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=str(rid % 2)))])


class _Client:
    def __init__(self, **_kw):
        self.chat = types.SimpleNamespace(completions=_Completions())


import openai  # noqa: E402

openai.OpenAI = _Client
common.vllm_server = lambda *_a, **_k: contextlib.nullcontext("http://stub")
rows = [{"dialogue_id": i, "round_num": 0, "model_text": f"ROW{i} reply"} for i in range(150)]
tmp = tempfile.mkdtemp(prefix="check_mapc_")
with open(os.path.join(tmp, "asr.json"), "w") as f:
    json.dump(rows, f)
job = mtr.MTRJudge(asr_results=tk.Path(os.path.join(tmp, "asr.json")), dimension="safety")
job.out_file = tk.Path(os.path.join(tmp, "scores.json"))
job.run()
res = json.load(open(os.path.join(tmp, "scores.json")))
assert [r["dialogue_id"] for r in res["per_round"]] == list(range(150)), "per_round not in row order"
assert all(r["score"] == float(r["dialogue_id"] % 2) for r in res["per_round"]), "a row got another row's score"
assert state.finished != list(range(150)), "judge calls completed in order: order check vacuous"
assert state.peak == min(mtr.MTR_JUDGE_CONCURRENCY, 150), f"judge peak in flight {state.peak}"
assert res["summary"]["n_scored"] == 150
print(f"[4] MTRJudge: 150 rows scored in row order, {state.peak} judge calls in flight")
print("OK")
