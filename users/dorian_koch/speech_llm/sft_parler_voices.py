"""``SftParlerVoices``' worker (Parler-TTS venv, GPU): Parler-TTS large v1 clips from ``<dir>/jobs.json``
(``{"id", "instruct", "text", "seed"}``) -> ``<dir>/wav/<id>.wav`` at 24 kHz. Resumable: clips already written are
skipped. The recognition check that keeps a clip is ``sft_voice_design.py check`` (annotate venv).

    sft_parler_voices.py <dir>
"""

import json
import os
import sys
import time

MODEL = "parler-tts/parler-tts-large-v1"
#: the main revision does not load with the venv's transformers pin (``parlertts_inference.py``)
REVISION = "refs/pr/9"
SR = 24000  # Qwen3-TTS's rate; the clone resamples anyway, this keeps the clips small


def main(d: str) -> None:
    import soundfile as sf
    import torch
    import torchaudio.functional as AF
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    jobs = [j for j in json.load(open(f"{d}/jobs.json")) if not os.path.exists(f"{d}/wav/{j['id']}.wav")]
    os.makedirs(f"{d}/wav", exist_ok=True)
    m = ParlerTTSForConditionalGeneration.from_pretrained(MODEL, revision=REVISION).to("cuda").eval()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    t0, audio = time.time(), 0.0
    print(f"[parler voices] {len(jobs)} clips to generate", flush=True)
    for i, j in enumerate(jobs):
        torch.manual_seed(j["seed"])
        desc, text = tok(j["instruct"], return_tensors="pt").to("cuda"), tok(j["text"], return_tensors="pt").to("cuda")
        with torch.inference_mode():
            a = m.generate(input_ids=desc.input_ids, attention_mask=desc.attention_mask,
                           prompt_input_ids=text.input_ids, prompt_attention_mask=text.attention_mask, do_sample=True)  # fmt: skip
        w = AF.resample(a.float().cpu().reshape(-1), m.config.sampling_rate, SR).numpy()
        tmp = f"{d}/wav/{j['id']}.part.wav"
        sf.write(tmp, w, SR)
        os.replace(tmp, f"{d}/wav/{j['id']}.wav")
        audio += len(w) / SR
        if (i + 1) % 10 == 0 or i + 1 == len(jobs):
            print(f"[parler voices] {i + 1}/{len(jobs)}: {audio:.0f} s audio in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
