"""Guard: the Qwen3-TTS backend of ChatterboxInferenceV2 (``tts_model: "qwen3"``, ``qwen3_tts_worker.py``).

Asserted against the real modules:

  * ``plan_dialogue`` routes every item of a dialogue to the Qwen3 worker (kind ``q3``), one item per short turn,
    a code-switched turn as ONE item in its main language with no language tag left in its text, and refuses
    ``tts_model: "qwen3"`` without ``code_switch: "single_pass"``;
  * ``Tts.synth`` hands the worker exactly the items' texts, languages, voice files and seeds, and puts the returned
    audio back in item order;
  * the worker process (the real ``qwen3_tts_worker.py`` over a stand-in ``qwen3_tts_fast``): the ICL prompt gets the
    transcript beside the voice file, ISO codes become Qwen3's language names, a voice without a transcript is an
    error, results come back in request order.

Shown failing on the inference script before the backend existed (no ``q3`` kind) and on a worker that ignores the
transcript sidecar.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_qwen3_tts_backend.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).absolute().parent.parent  # speech_llm/
sys.path.insert(0, str(HERE))
for name in ("chatterbox_fast", "chatterbox_fast_mtl", "chatterbox", "chatterbox.tts_turbo"):
    m = types.ModuleType(name)
    m.ChatterboxTurboTTS = None
    sys.modules.setdefault(name, m)
try:
    import torchaudio  # noqa: F401
except ImportError:
    sys.modules["torchaudio"] = types.ModuleType("torchaudio")

import chatterbox_v2_inference as inf  # noqa: E402

bad = []


def check_plan(tmp):
    vd = Path(tmp) / "voices"
    for lg in ("de", "en"):
        (vd / lg / "assistant").mkdir(parents=True)
        (vd / lg / "assistant" / f"v_{lg}.wav").write_bytes(b"")
    cfg = {"default_lang": "de", "english_model": "turbo", "code_switch": "single_pass", "tts_model": "qwen3"}
    dlg = {"lang": "de", "voices": {"user": "en/assistant/rng_v", "assistant": "de/assistant/rng_v"},
           "turns": [{"speaker": "user", "text": "Hallo, wie sagt man <en>good morning</en> auf Deutsch?"},
                     {"speaker": "assistant", "text": "Man sagt einfach guten Morgen."}]}  # fmt: skip
    p = inf.plan_dialogue(dlg, 7, cfg, str(vd))
    if p["kind"] != "q3" or {it["model"] for it in p["items"]} != {"q3"}:
        bad.append(f"qwen3 dialogue not routed to the q3 worker: {p['kind']}")
    if len(p["items"]) != 2:
        bad.append(f"expected one item per turn, got {len(p['items'])}")
    t0 = p["items"][0]
    if "<" in t0["text"] or t0["lang"] != "de" or "good morning" not in t0["text"]:
        bad.append(f"code-switched turn not one main-language item without tags: {t0['lang']} {t0['text']!r}")
    try:
        inf.plan_dialogue(dlg, 7, {**cfg, "code_switch": "segments"}, str(vd))
        bad.append("tts_model qwen3 accepted code_switch segments")
    except AssertionError:
        pass
    return p


def check_synth(p):
    calls = []

    class FakeWorker:
        def call_split(self, method, lists, **kw):
            calls.append((method, kw))
            return [np.full(10 + i, i, np.float32) for i in range(len(kw["texts"]))]

    tts = inf.Tts("cpu")
    tts.models["q3"] = FakeWorker()
    items = [dict(it, seed=100 + k) for k, it in enumerate(p["items"])]
    out = tts.synth(items)
    if len(calls) != 1 or calls[0][0] != "synth_many":
        bad.append(f"worker calls: {calls}")
        return
    kw = calls[0][1]
    want = {"texts": [it["text"] for it in items], "languages": [it["lang"] for it in items],
            "voices": [it["voice"] for it in items], "seeds": [it["seed"] for it in items]}  # fmt: skip
    for k, v in want.items():
        if kw.get(k) != v:
            bad.append(f"worker got {k}={kw.get(k)!r}, want {v!r}")
    if [len(w) for w in out] != [10, 11]:
        bad.append(f"audio not back in item order: {[len(w) for w in out]}")


FAKE_Q3 = """
import numpy as np
MODEL = "fake"
def load(device, path):
    return "model"
def prompt(model, ref_audio, ref_text):
    return {"audio": ref_audio, "text": ref_text}
def synthesize(model, items, slots=64):
    import json, os
    with open(os.environ["Q3_LOG"], "a") as f:
        f.write(json.dumps([{"text": it["text"], "language": it["language"], "ref": it["prompt"]["text"],
                             "seed": it["seed"]} for it in items]) + "\\n")
    return [np.full(5 + i, float(i), np.float32) for i in range(len(items))], 24000
"""


def check_worker(tmp):
    wd = Path(tmp) / "w"
    wd.mkdir()
    shutil.copy(HERE / "qwen3_tts_worker.py", wd / "qwen3_tts_worker.py")
    (wd / "qwen3_tts_fast.py").write_text(FAKE_Q3)
    v1, v2 = Path(tmp) / "a.wav", Path(tmp) / "b.wav"
    v1.write_bytes(b"")
    v2.write_bytes(b"")
    (Path(tmp) / "a.txt").write_text("Das ist meine Stimme.\n")
    log = Path(tmp) / "q3.log"
    env = {**os.environ, "Q3_LOG": str(log)}
    proc = subprocess.Popen([sys.executable, str(wd / "qwen3_tts_worker.py"), json.dumps({"device": "cpu"})],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)  # fmt: skip
    w = inf.WorkerProc.__new__(inf.WorkerProc)
    w.name, w.p = "q3test", proc
    if w._read() != {"ready": True}:
        bad.append("worker did not start")
        return
    res = w.call("synth_many", texts=["Hallo.", "Привет."], languages=["de", "ru"], voices=[str(v1), str(v1)],
                 seeds=[3, 4])  # fmt: skip
    got = json.loads(log.read_text().splitlines()[-1])
    want = [{"text": "Hallo.", "language": "German", "ref": "Das ist meine Stimme.", "seed": 3},
            {"text": "Привет.", "language": "Russian", "ref": "Das ist meine Stimme.", "seed": 4}]  # fmt: skip
    if got != want:
        bad.append(f"worker items {got} != {want}")
    if [len(r) for r in res] != [5, 6]:
        bad.append(f"worker results out of order: {[len(r) for r in res]}")
    try:
        w.call("synth_many", texts=["x"], languages=["de"], voices=[str(v2)], seeds=[1])
        bad.append("a voice without a transcript was accepted")
    except RuntimeError as e:
        if "transcript" not in str(e):
            bad.append(f"unexpected error: {e}")
    proc.stdin.close()
    proc.wait(timeout=30)


with tempfile.TemporaryDirectory() as tmp:
    try:
        p = check_plan(tmp)
        check_synth(p)
    except Exception as e:  # noqa: BLE001 -- a guard failure, reported below
        bad.append(f"plan/synth raised {type(e).__name__}: {e}")
    check_worker(tmp)
for b in bad:
    print("FAIL", b)
print("check_qwen3_tts_backend:", "FAIL" if bad else "ok")
sys.exit(1 if bad else 0)
