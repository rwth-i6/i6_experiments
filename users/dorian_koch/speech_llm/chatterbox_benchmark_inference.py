"""
Single-speaker TTS inference for knowledge benchmark questions.

Used by ChatterboxSingleSpeakerInference job via subprocess
(to run inside the chatterbox venv).
"""

import argparse
import gc
import os
import random

import torch
import torchaudio
from chatterbox.tts_turbo import ChatterboxTurboTTS
from datasets import load_from_disk


def resolve_speaker_path(speaker_dir: str, speaker_name: str) -> str:
    """Resolve speaker path, supporting rng_ prefix for random selection."""
    based = os.path.basename(speaker_name)
    if based.startswith("rng_"):
        dirname = os.path.dirname(speaker_name)
        search_dir = os.path.join(speaker_dir, dirname) if dirname else speaker_dir
        wavs = [f for f in os.listdir(search_dir) if f.endswith(".wav")]
        return os.path.join(search_dir, random.choice(wavs))
    return os.path.join(speaker_dir, speaker_name + ".wav")


def main():
    parser = argparse.ArgumentParser(description="Single-speaker TTS for benchmark questions")
    parser.add_argument("--in_hf", required=True, help="Path to input HF dataset")
    parser.add_argument("--speaker_dir", required=True, help="Directory containing speaker wavs")
    parser.add_argument("--speaker_name", default="user_voices/rng_a", help="Speaker name (supports rng_ prefix)")
    parser.add_argument("--out_dir", required=True, help="Output directory for wav files")
    args = parser.parse_args()

    random.seed(42)
    ds = load_from_disk(args.in_hf)

    speaker_path = resolve_speaker_path(args.speaker_dir, args.speaker_name)
    print(f"Using speaker: {speaker_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    torch._dynamo.config.capture_scalar_outputs = True
    torch.set_float32_matmul_precision("high")
    model.t3.inference_turbo = torch.compile(model.t3.inference_turbo, dynamic=True)

    model.prepare_conditionals(speaker_path, exaggeration=0.5, norm_loudness=True)

    os.makedirs(args.out_dir, exist_ok=True)
    with torch.inference_mode():
        for i, example in enumerate(ds):
            wav = model.generate(text=example["question"], audio_prompt_path=None)
            out_path = os.path.join(args.out_dir, f"{i}.wav")
            torchaudio.save(out_path, wav.cpu(), model.sr)
            del wav
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processed {i + 1}/{len(ds)} examples", flush=True)

    print(f"Done. Generated {len(ds)} audio files in {args.out_dir}")


if __name__ == "__main__":
    main()
