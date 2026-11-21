"""Guard: the Qwen3-TTS backend of ChatterboxInferenceV2 (``tts_model: "qwen3"``, ``qwen3_tts_worker.py``).

Asserted against the real modules:

  * ``plan_dialogue`` routes every item of a dialogue to the Qwen3 worker (kind ``q3``), one item per short turn,
    a code-switched turn as ONE item in its main language with no language tag left in its text (``qwen3_mixed_lang:
    "auto"``: that item's TTS language is ``auto``, its check language stays the main one), with ``code_switch:
    "segments"`` one item per language span, a span in another language than its speaker's voice prompt with
    ``voice_mode: "timbre"`` (speaker embedding only; the ICL clone would carry the prompt's accent), and refuses
    Chatterbox's span-token modes;
  * ``Tts.synth`` hands the worker exactly the items' texts, TTS languages, voice files, seeds and voice modes, and
    puts the returned audio back in item order;
  * the worker process (the real ``qwen3_tts_worker.py`` over a stand-in ``qwen3_tts_fast``): the ICL prompt gets the
    transcript beside the voice file, ISO codes become Qwen3's language names, a voice without a transcript is an
    error, results come back in request order, ``modes`` "timbre" builds a speaker-embedding-only prompt (no
    transcript needed) and leaves "clone" items on the ICL prompt;
  * ``qwen3_tts_fast.decode`` decodes a timbre-only item (no reference codes) without cutting anything off;
  * ``ContinuousTalker.precapture`` called with autograd on (as ``synthesize`` calls it) runs the captured step without
    autograd (CUDA graph calls stood in on CPU): a capture that records the backward graph keeps the KV cache and every
    step intermediate allocated after the call (a 64-slot call left ~64 GB, every later call ran out of memory).

Shown failing on the inference script before the backend existed (no ``q3`` kind) and on a worker that ignores the
transcript sidecar; the timbre-only parts on the code before 2026-11-19 (no voice modes); the capture part on the code
before 2026-11-21 (step graphs captured with autograd on).

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
        inf.plan_dialogue(dlg, 7, {**cfg, "code_switch": "mixed"}, str(vd))
        bad.append("tts_model qwen3 accepted code_switch mixed (Chatterbox span tokens)")
    except AssertionError:
        pass
    a = inf.plan_dialogue(dlg, 7, {**cfg, "qwen3_mixed_lang": "auto"}, str(vd))["items"]
    if [it.get("tts_lang") for it in a] != ["auto", None] or a[0]["lang"] != "de":
        bad.append(f"qwen3_mixed_lang auto: tts_lang {[it.get('tts_lang') for it in a]}, check lang {a[0]['lang']}")
    g = inf.plan_dialogue(dlg, 7, {**cfg, "code_switch": "segments"}, str(vd))["items"]
    if [(it["turn"], it["lang"]) for it in g] != [(0, "de"), (0, "en"), (0, "de"), (1, "de")]:
        bad.append(f"qwen3 segments: {[(it['turn'], it['lang'], it['text']) for it in g]}")
    # turn 0 is the user's (English voice prompt): its German spans timbre only, its English span cloned
    if [it.get("voice_mode") for it in g] != ["timbre", None, "timbre", None]:
        bad.append(f"qwen3 segments voice modes: {[(it['lang'], it.get('voice_mode')) for it in g]}")
    return p


def check_synth(p):
    calls = []

    class FakeWorker:
        def call_split(self, method, lists, **kw):
            calls.append((method, kw))
            return [np.full(10 + i, i, "<f4").tobytes() for i in range(len(kw["texts"]))]

    tts = inf.Tts("cpu")
    tts.models["q3"] = FakeWorker()
    items = [dict(it, seed=100 + k) for k, it in enumerate(p["items"])]
    items[0]["tts_lang"] = "auto"
    items[0].pop("voice_mode", None)
    items[1]["voice_mode"] = "timbre"
    out = tts.synth(items)
    if len(calls) != 1 or calls[0][0] != "synth_many":
        bad.append(f"worker calls: {calls}")
        return
    kw = calls[0][1]
    want = {"texts": [it["text"] for it in items], "languages": ["auto"] + [it["lang"] for it in items[1:]],
            "voices": [it["voice"] for it in items], "seeds": [it["seed"] for it in items],
            "modes": ["clone", "timbre"]}  # fmt: skip
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
def prompt(model, ref_audio, ref_text, timbre_only=False):
    return {"audio": ref_audio, "text": "TIMBRE" if timbre_only else ref_text}
class _Kv:  # a decoder state left in a reference cycle, as a finished call's can be
    def __init__(self):
        self.me = self
_LIVE = []
def synthesize(model, items, slots=64):
    import json, os, weakref
    import torch
    if any(r() is not None for r in _LIVE):
        raise torch.OutOfMemoryError("CUDA out of memory (an earlier call's state is still alive)")
    if os.environ.get("Q3_MAX_SLOTS") and slots > int(os.environ["Q3_MAX_SLOTS"]):
        raise torch.OutOfMemoryError(f"CUDA out of memory at {slots} slots")
    _LIVE.append(weakref.ref(_Kv()))
    with open(os.environ["Q3_LOG"] + ".slots", "a") as f:
        f.write(f"{slots}\\n")
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
    env = {**os.environ, "Q3_LOG": str(log), "Q3_MAX_SLOTS": "16"}
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
    if not all(isinstance(r, bytes) for r in res):
        bad.append(f"worker returns {type(res[0]).__name__}, not bytes (numpy arrays do not unpickle across venvs)")
    elif [len(r) // 4 for r in res] != [5, 6]:
        bad.append(f"worker results out of order: {[len(r) for r in res]}")
    try:  # v2 has no transcript: fine for a timbre-only prompt
        w.call("synth_many", texts=["Hallo.", "Hi."], languages=["de", "en"], voices=[str(v1), str(v2)],
               seeds=[5, 6], modes=["clone", "timbre"])  # fmt: skip
        got = json.loads(log.read_text().splitlines()[-1])
        if [g["ref"] for g in got] != ["Das ist meine Stimme.", "TIMBRE"]:
            bad.append(f"worker voice modes: prompts {[g['ref'] for g in got]}, want ICL then timbre only")
    except RuntimeError as e:
        bad.append(f"worker voice modes: {e}")
    try:
        w.call("synth_many", texts=["x"], languages=["de"], voices=[str(v2)], seeds=[1])
        bad.append("a voice without a transcript was accepted")
    except RuntimeError as e:
        if "transcript" not in str(e):
            bad.append(f"unexpected error: {e}")
    proc.stdin.close()
    proc.wait(timeout=30)
    # spec slots 64, the stand-in fits 16: the first request retries 64 -> 32 -> 16, later ones start at 16; a
    # garbage cycle left by a call must not cost a later request a retry (it would go down to 8)
    seen = (Path(str(log) + ".slots").read_text().split() if Path(str(log) + ".slots").exists() else [])
    if seen != ["16", "16"]:
        bad.append(f"worker slots per successful call {seen}, want ['16', '16'] (OOM retry at half the slots, "
                   "garbage collected before each call)")


def check_decode():
    import torch

    import qwen3_tts_fast as q3

    class Tok:
        def decode(self, batch):
            return [np.arange(100 * int(b["audio_codes"].shape[0]), dtype=np.float32) for b in batch], 24000

    model = types.SimpleNamespace(model=types.SimpleNamespace(speech_tokenizer=Tok()))
    codes = [torch.zeros(3, 16), torch.zeros(4, 16)]
    out = q3.decode(model, codes, [torch.zeros(2, 16), None])
    if [len(w) for w in out] != [300, 400] or out[1][0] != 0:
        bad.append(f"decode with / without reference codes: lengths {[len(w) for w in out]} (want 300, 400)")


def check_capture_no_autograd():
    import contextlib

    import torch

    import qwen3_tts_fast as q3

    class _Stream:
        def wait_stream(self, other):
            pass

    stand_ins = {"Stream": _Stream, "current_stream": _Stream, "stream": lambda s: contextlib.nullcontext(),
                 "CUDAGraph": object, "graph": lambda g: contextlib.nullcontext(), "synchronize": lambda: None}  # fmt: skip
    saved = {k: getattr(torch.cuda, k) for k in stand_ins}
    lin = torch.nn.Linear(2, 2)  # parameters require grad, as the loaded model's do
    t = types.SimpleNamespace(graphs={}, bucket=128, lmax=256)
    for k in ("bias", "h", "code0", "g", "pos", "done", "seen", "out"):
        setattr(t, k, torch.zeros(2))
    grad_on = []

    def step(w):  # as ContinuousTalker._step: persistent state updated in place from the model's output
        grad_on.append(torch.is_grad_enabled())
        t.h.copy_(lin(t.h))

    t._step = step
    t._graph = lambda w: q3.ContinuousTalker._graph(t, w)
    try:
        for k, v in stand_ins.items():
            setattr(torch.cuda, k, v)
        with torch.enable_grad():
            q3.ContinuousTalker.precapture(t)
    finally:
        for k, v in saved.items():
            setattr(torch.cuda, k, v)
    if not grad_on or any(grad_on) or t.h.grad_fn is not None:
        bad.append(f"step graphs captured with autograd on (per step {grad_on}, state grad_fn {t.h.grad_fn}): the "
                   "backward graph keeps the KV cache and the step intermediates allocated after the call")


with tempfile.TemporaryDirectory() as tmp:
    try:
        p = check_plan(tmp)
        check_synth(p)
    except Exception as e:  # noqa: BLE001 -- a guard failure, reported below
        bad.append(f"plan/synth raised {type(e).__name__}: {e}")
    check_worker(tmp)
try:
    check_decode()
except Exception as e:  # noqa: BLE001 -- a guard failure, reported below
    bad.append(f"decode raised {type(e).__name__}: {e}")
try:
    check_capture_no_autograd()
except Exception as e:  # noqa: BLE001 -- a guard failure, reported below
    bad.append(f"capture check raised {type(e).__name__}: {e}")
for b in bad:
    print("FAIL", b)
print("check_qwen3_tts_backend:", "FAIL" if bad else "ok")
sys.exit(1 if bad else 0)
