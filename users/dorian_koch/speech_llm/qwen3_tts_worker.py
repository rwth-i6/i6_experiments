"""Qwen3-TTS worker of ``ChatterboxInferenceV2`` (``tts_model: "qwen3"``): voice-clone synthesis through
``qwen3_tts_fast`` in the Qwen3-TTS venv, as a separate process beside the job's own (Chatterbox) venv.

Protocol as ``ru_stress_worker``: 8-byte little-endian length + pickle on stdin / stdout, first message
``{"ready": True}``, request ``{"method", ...}`` -> ``{"result"}`` or ``{"error"}``. Method ``synth_many(texts,
languages, voices, seeds)`` -> ``list[np.ndarray]`` (float32, mono, ``SR``). ``voices``: voice prompt wav paths; each
needs its transcript beside it (``<name>.txt``: the ICL clone prompt is the clip plus its text). ``languages``: ISO
codes (``en``, ``de``, ``zh``, ``ru``, ...). Argument: JSON ``{"model", "device", "slots"}``.
"""

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
    write(out, {"ready": True})
    inp = sys.stdin.buffer
    while (req := read(inp)) is not None:
        try:
            if req["method"] != "synth_many":
                raise ValueError(f"unknown method {req['method']!r}")
            for v in req["voices"]:
                if v not in prompts:
                    prompts[v] = q3.prompt(model, v, ref_text(v))
            items = [{"text": t, "language": LANG[lg], "prompt": prompts[v], "seed": int(s)}
                     for t, lg, v, s in zip(req["texts"], req["languages"], req["voices"], req["seeds"])]  # fmt: skip
            wavs, sr = q3.synthesize(model, items, slots=int(spec.get("slots", 64)))
            if sr != SR:
                import torchaudio.functional as AF

                wavs = [AF.resample(torch.from_numpy(w), sr, SR).numpy() for w in wavs]
            write(out, {"result": [np.asarray(w, np.float32) for w in wavs], "id": req.get("id")})
        except Exception as e:  # reported to the caller, which raises (CUDA OOM: the caller splits the batch)
            torch.cuda.empty_cache()
            write(out, {"error": f"{type(e).__name__}: {e}", "id": req.get("id")})


if __name__ == "__main__":
    serve()
