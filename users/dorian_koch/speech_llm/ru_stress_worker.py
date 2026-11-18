"""Russian stress worker of ``ChatterboxInferenceV2``: stress marks (combining acute, U+0301) on Russian text, as
Chatterbox Multilingual's own tokenizer adds them when ``russian_text_stresser`` is importable
(``chatterbox.models.tokenizers.tokenizer.add_russian_stress``; its vocabulary has the mark). That package
(github.com/Vuizur/add-stress-to-epub) needs Python < 3.12 and spaCy 3.6, so it cannot live in the TTS venv; without it
upstream silently synthesizes unstressed Russian (the 2026-11-17 pilot: 525 "not available" warnings, Russian rated
worst by listeners).

Protocol as ``speech_annotate_worker``: 8-byte little-endian length + pickle on stdin / stdout, first message
``{"ready": True}``, request ``{"method", ...}`` -> ``{"result"}`` or ``{"error"}``. Method: ``stress_many(texts)`` ->
``list[str]`` (words the dictionary cannot decide stay unmarked). The dictionary must already be in the package
directory (it downloads on first use; jobs run offline).
"""

import os
import pickle
import struct
import sys


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


def serve() -> None:
    out = os.fdopen(os.dup(1), "wb")  # the protocol channel; anything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    from russian_text_stresser.text_stresser import RussianTextStresser

    ts = RussianTextStresser()
    write(out, {"ready": True})
    inp = sys.stdin.buffer
    while (req := read(inp)) is not None:
        try:
            if req["method"] != "stress_many":
                raise ValueError(f"unknown method {req['method']!r}")
            write(out, {"result": [ts.stress_text(t) for t in req["texts"]], "id": req.get("id")})
        except Exception as e:  # reported to the caller, which raises
            write(out, {"error": f"{type(e).__name__}: {e}", "id": req.get("id")})


if __name__ == "__main__":
    serve()
