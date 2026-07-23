"""ASR-transcribe our model's reply wavs and assemble VoiceBench's response JSONL schema.

Reuses the same faster-whisper + Silero-VAD approach as our WhisperTranscription worker (strips trailing
silence so Whisper doesn't hallucinate on the digital silence after the reply). Emits, per index `i` that
has a `{i}.wav` reply, one line: `{**non_audio_fields(ref_ds[i]), 'response': transcript}` — VoiceBench's
exact schema, consumed unchanged by their scorers.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_from_disk
from faster_whisper import BatchedInferencePipeline, WhisperModel

TARGET_SR = 16000


def load_audio(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
    return audio


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True, help="Directory of model reply wavs ({i}.wav)")
    p.add_argument("--ref_ds", required=True, help="HF dataset of non-audio VoiceBench fields (index-aligned)")
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--model_size", default="large-v3-turbo")
    p.add_argument("--batch_size", type=int, default=24)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(args.model_size, device=device, compute_type=compute_type)
    batched = BatchedInferencePipeline(model)

    ref_ds = load_from_disk(args.ref_ds)
    valid = [(i, ex) for i, ex in enumerate(ref_ds) if os.path.exists(os.path.join(args.in_dir, f"{i}.wav"))]
    print(f"Transcribing {len(valid)}/{len(ref_ds)} replies", flush=True)

    n = 0
    with open(args.out_jsonl, "w") as f:
        for i, example in valid:
            audio = load_audio(os.path.join(args.in_dir, f"{i}.wav"))
            if audio.size < TARGET_SR // 100:
                text = ""  # silent reply -> empty hypothesis (a benchmark miss), don't feed whisper garbage
            else:
                segments, _ = batched.transcribe(
                    audio,
                    batch_size=args.batch_size,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    condition_on_previous_text=False,
                )
                text = "".join(s.text for s in segments).strip()
            record = {**example, "response": text}  # example already excludes audio
            f.write(json.dumps(record, default=str) + "\n")
            n += 1
            if n % 100 == 0:
                print(f"Transcribed {n}/{len(valid)}", flush=True)
    print(f"Done. Wrote {n} responses to {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
