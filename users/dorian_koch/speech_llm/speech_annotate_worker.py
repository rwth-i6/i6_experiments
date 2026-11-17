"""Speech annotation worker of ``ChatterboxInferenceV2``: Qwen3-ASR transcripts, Qwen3-ForcedAligner timings of a given
text, and the round-trip error rate (``speech_text_norm``).

Run by path under the annotate venv (``qwen-asr`` pins its own transformers, so it cannot share the TTS or the NeMo
venv). Protocol as ``speech_llm``'s ``parakeet_worker``: 8-byte little-endian length + pickle on stdin / stdout, first
message ``{"ready": True}``, request ``{"method", ...}`` -> ``{"result"}`` or ``{"error"}``. Methods:

* ``transcribe_many(pcms, sr, language)`` -> ``list[str]`` (Qwen3-ASR, language forced);
* ``align_many(pcms, sr, texts, language)`` -> per clip ``[{"text", "start", "end"}]`` in seconds: Qwen3-ForcedAligner
  timings of ``texts``' units (words for space-delimited languages, single characters for Chinese / Japanese; the
  aligner's own segmentation, punctuation dropped);
* ``score_many(refs, hyps, language)`` -> per pair ``speech_text_norm.error_rate`` (CPU; needs no model).

Spec (argv[1], JSON): ``{"asr_model", "aligner_model", "device", "batch_size"}``; either model may be null (not loaded;
both null = a CPU-only scorer).
Models are read from the shared HF cache only (jobs run offline).
"""

import json
import os
import pickle
import struct
import sys

import numpy as np

#: language ids (ISO 639-1, as ``--language``) -> the names Qwen3-ASR / -ForcedAligner take
LANG_NAMES = {"zh": "Chinese", "en": "English", "de": "German", "fr": "French", "es": "Spanish", "it": "Italian",
              "pt": "Portuguese", "ru": "Russian", "ja": "Japanese", "ko": "Korean", "yue": "Cantonese"}  # fmt: skip
#: what Qwen3-ForcedAligner aligns (its model card); Qwen3-ASR covers more
ALIGNER_LANGS = frozenset(LANG_NAMES)


def read(f):
    head = f.read(8)
    if len(head) < 8:
        return None
    (n,) = struct.unpack("<Q", head)
    return pickle.loads(f.read(n))


def write(f, obj) -> None:
    b = pickle.dumps(obj, protocol=5)
    f.write(struct.pack("<Q", len(b)))
    f.write(b)
    f.flush()


def _text_norm():
    """``speech_text_norm`` from this file's directory, by path (this file runs as a script)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_speech_text_norm", os.path.join(os.path.dirname(os.path.abspath(__file__)), "speech_text_norm.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _local(repo: str) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, local_files_only=True)


class Qwen3Speech:
    def __init__(self, asr_model, aligner_model, device: str, batch_size: int = 32):
        self.bs = int(batch_size)
        self.norm = _text_norm()
        self.asr = self.aligner = None
        if not (asr_model or aligner_model):
            return  # scoring only
        import torch
        from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

        kw = {"dtype": torch.bfloat16, "device_map": device}
        self.asr = Qwen3ASRModel.from_pretrained(_local(asr_model), max_inference_batch_size=self.bs,
                                                 **kw) if asr_model else None  # fmt: skip
        self.aligner = Qwen3ForcedAligner.from_pretrained(_local(aligner_model), **kw) if aligner_model else None

    @staticmethod
    def _name(language: str) -> str:
        if language not in LANG_NAMES:
            raise ValueError(f"qwen3: no language name for {language!r} (known: {sorted(LANG_NAMES)})")
        return LANG_NAMES[language]

    def transcribe_many(self, pcms, sr: int, language: str) -> list[str]:
        assert self.asr is not None, "qwen3 worker started without an ASR model"
        out = [""] * len(pcms)
        idx = [i for i, p in enumerate(pcms) if len(p) >= int(0.1 * sr)]  # < 100 ms: nothing to transcribe
        for a in range(0, len(idx), self.bs):
            part = idx[a : a + self.bs]
            res = self.asr.transcribe([(np.asarray(pcms[i], np.float32), sr) for i in part],
                                      language=self._name(language))  # fmt: skip
            for i, r in zip(part, res):
                out[i] = r.text.strip()
        return out

    def align_many(self, pcms, sr: int, texts, language: str) -> list[list[dict]]:
        assert self.aligner is not None, "qwen3 worker started without an aligner model"
        if language not in ALIGNER_LANGS:
            raise ValueError(f"Qwen3-ForcedAligner does not align {language!r}")
        out: list[list[dict]] = [[] for _ in pcms]
        idx = [i for i, (p, t) in enumerate(zip(pcms, texts)) if len(p) >= int(0.1 * sr) and t.strip()]
        for a in range(0, len(idx), self.bs):
            part = idx[a : a + self.bs]
            res = self.aligner.align(audio=[(np.asarray(pcms[i], np.float32), sr) for i in part],
                                     text=[texts[i] for i in part], language=self._name(language))  # fmt: skip
            for i, r in zip(part, res):
                out[i] = [{"text": it.text, "start": float(it.start_time), "end": float(it.end_time)} for it in r]
        return out

    def score_many(self, refs, hyps, language: str) -> list[dict]:
        return [self.norm.error_rate(r, h, language) for r, h in zip(refs, hyps, strict=True)]


def serve(spec: dict) -> None:
    out = os.fdopen(os.dup(1), "wb")  # the protocol channel; anything else printed goes to stderr
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    w = Qwen3Speech(spec.get("asr_model"), spec.get("aligner_model"), spec.get("device", "cuda"),
                    spec.get("batch_size", 32))  # fmt: skip
    write(out, {"ready": True})
    inp = sys.stdin.buffer
    while (req := read(inp)) is not None:
        try:
            if req["method"] == "transcribe_many":
                res = w.transcribe_many(req["pcms"], req["sr"], req["language"])
            elif req["method"] == "align_many":
                res = w.align_many(req["pcms"], req["sr"], req["texts"], req["language"])
            elif req["method"] == "score_many":
                res = w.score_many(req["refs"], req["hyps"], req["language"])
            else:
                raise ValueError(f"unknown method {req['method']!r}")
            write(out, {"result": res, "id": req.get("id")})
        except Exception as e:  # reported to the caller, which raises
            write(out, {"error": f"{type(e).__name__}: {e}", "id": req.get("id")})


if __name__ == "__main__":
    serve(json.loads(sys.argv[1]))
