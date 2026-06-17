"""
Whisper ASR transcription for knowledge benchmark Moshi responses.

Used by WhisperTranscription job via subprocess (runs inside the whisper venv).

Uses faster-whisper with Silero VAD to strip trailing silence before transcription.
This prevents Whisper from hallucinating phrases ("Thank you.") on the 16-37s of
pure digital silence that follows Moshi speech in the output wavs.
"""

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_from_disk
from faster_whisper import WhisperModel, BatchedInferencePipeline


TARGET_SR = 16000


def load_audio(path: str) -> np.ndarray:
    """Read wav, downmix to mono, resample to 16 kHz, return float32 numpy array."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), sr, TARGET_SR).numpy()
    return audio


def main():
    parser = argparse.ArgumentParser(description="Whisper transcription for benchmark responses")
    parser.add_argument("--in_dir", required=True, help="Directory of Moshi response wav files")
    parser.add_argument("--reference_data", required=True, help="Path to reference HF dataset")
    parser.add_argument("--out_json", required=True, help="Output transcriptions jsonl path")
    parser.add_argument("--model_size", default="large-v3-turbo", help="Whisper model size")
    parser.add_argument("--batch_size", type=int, default=24, help="Batch size for faster-whisper internal chunking")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    print(f"Loading WhisperModel({args.model_size!r}, device={device!r}, compute_type={compute_type!r})", flush=True)
    model = WhisperModel(args.model_size, device=device, compute_type=compute_type)
    batched = BatchedInferencePipeline(model)

    ref_ds = load_from_disk(args.reference_data)

    # Examples that actually have a Moshi response wav, in dataset order.
    valid = [(i, ex) for i, ex in enumerate(ref_ds) if os.path.exists(os.path.join(args.in_dir, f"{i}.wav"))]
    print(f"Transcribing {len(valid)}/{len(ref_ds)} clips (batch_size={args.batch_size})", flush=True)

    n = 0
    with open(args.out_json, "w") as f:
        for i, example in valid:
            audio = load_audio(os.path.join(args.in_dir, f"{i}.wav"))

            segments, _ = batched.transcribe(
                audio,
                batch_size=args.batch_size,
                # Silero VAD: skip non-speech regions (incl. trailing silence)
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                # Standard Whisper anti-hallucination thresholds
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                condition_on_previous_text=False,
            )
            text = "".join(s.text for s in segments).strip()

            f.write(
                json.dumps(
                    {
                        "question": example["question"],
                        "answer": example["answer"],
                        "aliases": example["aliases"],
                        "category": example.get("category", "unknown"),
                        "transcription": text,
                    }
                )
                + "\n"
            )
            n += 1
            if n % 100 == 0:
                print(f"Transcribed {n}/{len(valid)} clips", flush=True)

    print(f"Done. Wrote {n} transcriptions to {args.out_json}")


if __name__ == "__main__":
    main()
