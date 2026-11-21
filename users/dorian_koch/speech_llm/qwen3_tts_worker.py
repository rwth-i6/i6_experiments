"""Qwen3-TTS worker of ``ChatterboxInferenceV2`` (``tts_model: "qwen3"``): voice-clone synthesis through
``qwen3_tts_fast`` in the Qwen3-TTS venv, as a separate process beside the job's own (Chatterbox) venv.

Protocol as ``ru_stress_worker``: 8-byte little-endian length + pickle on stdin / stdout, first message
``{"ready": True}``, request ``{"method", ...}`` -> ``{"result"}`` or ``{"error"}``. Method ``synth_many(texts,
languages, voices, seeds, modes=None)`` -> ``list[bytes]``: raw little-endian float32 mono at ``SR`` (bytes, not numpy arrays:
this venv's numpy 2 pickles arrays the caller's numpy 1 cannot load). ``voices``: voice prompt wav paths; each
needs its transcript beside it (``<name>.txt``: the ICL clone prompt is the clip plus its text). ``languages``: ISO
codes (``en``, ``de``, ``zh``, ``ru``, ...). ``modes``: per item ``"clone"`` (default; ICL) or ``"timbre"`` (speaker
embedding only: for text in another language than the voice prompt's, which ICL would speak with the prompt's
accent; ``qwen3_tts_fast.prompt``). Argument: JSON ``{"model", "device", "slots"}``.

Memory: each call collects garbage first (a finished call's decoder state can sit in reference cycles until a full
collection; the full SFT run's TTS jobs held ~77 GB of live tensors and failed every retry, 2026-11-21), and a CUDA
out of memory retries the same request at half the slots, which then stay halved for the worker's later requests.
"""

import gc
import json
import os
import pickle
import struct
import sys

SR = 24000  # the rate the inference script renders at (``Tts.sr``)
LANG = {"en": "English", "de": "German", "zh": "Chinese", "ru": "Russian", "fr": "French", "es": "Spanish",
        "it": "Italian", "pt": "Portuguese", "ja": "Japanese", "ko": "Korean", "auto": "Auto"}  # fmt: skip


def read(f):
    head = f.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(f.read(n))


def write(f, obj) -> None:
    b = pickle.dumps(obj, protocol=5)
    f.write(struct.pack("<Q", len(b)) + b)
    f.flush()


def ref_text(voice: str) -> str:
    p = os.path.splitext(voice)[0] + ".txt"
    if not os.path.exists(p):
        raise FileNotFoundError(f"{p}: a Qwen3-TTS voice prompt needs its transcript")
    with open(p) as f:
        return f.read().strip()


def serve() -> None:
    out = os.fdopen(os.dup(1), "wb")  # the protocol channel; anything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    spec = json.loads(sys.argv[1])
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import numpy as np
    import torch

    import qwen3_tts_fast as q3

    torch.set_float32_matmul_precision("high")
    model = q3.load(spec.get("device", "cuda"), spec.get("model") or q3.MODEL)
    prompts = {}
    slots = int(spec.get("slots", 64))

    def synth(items):
        nonlocal slots
        while True:
            gc.collect()
            torch.cuda.empty_cache()
            try:
                return q3.synthesize(model, items, slots=slots)
            except torch.OutOfMemoryError:
                if slots <= 4:
                    raise
            slots //= 2  # outside the except block: its traceback (the failed call's tensors) is released by now
            print(f"[qwen3_tts_worker] CUDA out of memory on {len(items)} items; now {slots} slots", flush=True)

    write(out, {"ready": True})
    inp = sys.stdin.buffer
    while (req := read(inp)) is not None:
        try:
            if req["method"] != "synth_many":
                raise ValueError(f"unknown method {req['method']!r}")
            modes = req.get("modes") or ["clone"] * len(req["voices"])
            for v, md in zip(req["voices"], modes, strict=True):
                if md not in ("clone", "timbre"):
                    raise ValueError(f"unknown voice mode {md!r}")
                if (v, md) not in prompts:
                    prompts[v, md] = (q3.prompt(model, v, None, timbre_only=True) if md == "timbre"
                                      else q3.prompt(model, v, ref_text(v)))  # fmt: skip
            items = [{"text": t, "language": LANG[lg], "prompt": prompts[v, md], "seed": int(s)}
                     for t, lg, v, s, md in zip(req["texts"], req["languages"], req["voices"], req["seeds"], modes)]  # fmt: skip
            wavs, sr = synth(items)
            if sr != SR:
                import torchaudio.functional as AF

                wavs = [AF.resample(torch.from_numpy(w), sr, SR).numpy() for w in wavs]
            write(out, {"result": [np.asarray(w, "<f4").tobytes() for w in wavs], "id": req.get("id")})
        except Exception as e:  # reported to the caller, which raises (CUDA OOM: the caller splits the batch)
            write(out, {"error": f"{type(e).__name__}: {e}", "id": req.get("id")})


if __name__ == "__main__":
    serve()
