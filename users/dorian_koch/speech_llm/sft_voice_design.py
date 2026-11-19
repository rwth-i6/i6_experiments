"""``SftDesignVoices``' worker: Qwen3-TTS VoiceDesign clips from ``jobs.json`` and their recognition check.

    sft_voice_design.py gen <dir> <model>      (Qwen3-TTS venv, GPU): <dir>/jobs.json -> <dir>/wav/<id>.wav
    sft_voice_design.py check <dir> <asr>      (annotate venv, GPU): Qwen3-ASR (language given) against each clip's
        text; clips with CER <= MAX_CER and MIN_S..MAX_S long go to <dir>/voices/<lang>/<role>/<gender>/vd_<id>.wav
        with the text beside them (<name>.txt: the voice-clone prompt's transcript); <dir>/check.json per clip.

Resumable: ``gen`` skips clips already written. Batches of 16, the codec frames capped at 2.5x a slow reading of
the longest text + 3 s (the default 8192 frames let one clip that never ends hold a batch for minutes).
"""

import json
import os
import sys
import time

MAX_CER, MIN_S, MAX_S = 0.08, 5.0, 16.0


def gen(d: str, model: str) -> None:
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    jobs = [j for j in json.load(open(f"{d}/jobs.json")) if not os.path.exists(f"{d}/wav/{j['id']}.wav")]
    os.makedirs(f"{d}/wav", exist_ok=True)
    t0, audio = time.time(), 0.0
    m = Qwen3TTSModel.from_pretrained(model, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
    jobs.sort(key=lambda j: len(j["text"]))
    for i in range(0, len(jobs), 16):
        b = jobs[i : i + 16]
        est = max(len(j["text"]) / (3 if j["language"] == "Chinese" else 12) for j in b)
        wavs, sr = m.generate_voice_design(text=[j["text"] for j in b], language=[j["language"] for j in b],
                                           instruct=[j["instruct"] for j in b],
                                           max_new_tokens=int(12.5 * (2.5 * est + 3)))  # fmt: skip
        for j, w in zip(b, wavs):
            sf.write(f"{d}/wav/{j['id']}.wav", w, sr)
            audio += len(w) / sr
        print(f"[voice design] {i + len(b)}/{len(jobs)}: {audio:.0f} s audio in {time.time() - t0:.0f} s", flush=True)


def check(d: str, asr: str) -> None:
    import shutil

    import soundfile as sf

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from speech_annotate_worker import Qwen3Speech

    jobs = json.load(open(f"{d}/jobs.json"))
    q = Qwen3Speech(asr, None, "cuda", 16)
    res = {}
    for lg in sorted({j["lang"] for j in jobs}):
        js = [j for j in jobs if j["lang"] == lg]
        au = [sf.read(f"{d}/wav/{j['id']}.wav", dtype="float32") for j in js]
        hyp = q.transcribe_many([a for a, _ in au], au[0][1], lg)
        for j, (a, sr), h, s in zip(js, au, hyp, q.score_many([j["text"] for j in js], hyp, lg)):
            dur = len(a) / sr
            ok = s["cer"] <= MAX_CER and MIN_S <= dur <= MAX_S
            res[j["id"]] = {"cer": s["cer"], "dur": round(dur, 2), "ok": ok, "asr": h}
            if ok:
                o = f"{d}/voices/{lg}/{j['role']}/{j['gender']}"
                os.makedirs(o, exist_ok=True)
                shutil.copy(f"{d}/wav/{j['id']}.wav", f"{o}/vd_{j['id']}.wav")
                with open(f"{o}/vd_{j['id']}.txt", "w", encoding="utf-8") as f:
                    f.write(j["text"] + "\n")
    with open(f"{d}/check.json", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    kept = sum(r["ok"] for r in res.values())
    print(f"[voice design] kept {kept} of {len(res)} clips", flush=True)
    if kept < 0.8 * len(res):
        raise RuntimeError(f"only {kept} of {len(res)} designed voices passed the check")


if __name__ == "__main__":
    {"gen": gen, "check": check}[sys.argv[1]](sys.argv[2], sys.argv[3])
