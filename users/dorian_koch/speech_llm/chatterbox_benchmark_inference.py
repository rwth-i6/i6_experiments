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
from datasets import Dataset, load_from_disk

# NOTE: deliberately duplicated from speech_llm/clip_store.py -- see the note in
# chatterbox_inference.py. This runs under the chatterbox job venv, which has neither sisyphus nor
# i6_experiments importable (clip_store is fine on its own, but lives next to common.py whose first
# line is `from sisyphus import tk`). Keep the column names identical to clip_store's.
COL_INDEX, COL_AUDIO, COL_SR = "index", "audio", "sampling_rate"
COL_MONOLOGUE, COL_TRACE = "monologue", "trace_json"


def write_clips(out_path, items):
    """Write (index, samples, sample_rate) triples as one arrow dataset. Mirrors clip_store."""
    rows = {COL_INDEX: [], COL_AUDIO: [], COL_SR: [], COL_MONOLOGUE: [], COL_TRACE: []}
    for index, samples, sr in items:
        rows[COL_INDEX].append(int(index))
        rows[COL_AUDIO].append([float(x) for x in samples])
        rows[COL_SR].append(int(sr))
        rows[COL_MONOLOGUE].append(None)
        rows[COL_TRACE].append(None)
    Dataset.from_dict(rows).save_to_disk(out_path)


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
    parser.add_argument("--out_dir", required=True, help="Output directory for wav files / arrow dataset")
    parser.add_argument(
        "--storage",
        default="wav",
        choices=("wav", "hf"),
        help="wav: one <i>.wav per question (original). hf: one arrow dataset (~3 inodes total).",
    )
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
    # storage="hf" accumulates clips in memory and writes ONE arrow dataset at the end instead of a
    # wav per question. A 1000-question benchmark drops from ~1000 inodes to ~3, which is what the
    # shared /hpcwork project volume actually runs out of. Questions are short (a few seconds), so
    # the whole set is a few hundred MB -- fine to hold; a corpus-scale job would need batching.
    rows = []
    with torch.inference_mode():
        for i, example in enumerate(ds):
            wav = model.generate(text=example["question"], audio_prompt_path=None)
            if args.storage == "hf":
                rows.append((i, wav.cpu().numpy().reshape(-1).astype("float32"), model.sr))
            else:
                torchaudio.save(os.path.join(args.out_dir, f"{i}.wav"), wav.cpu(), model.sr)
            del wav
            if (i + 1) % 100 == 0:
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Processed {i + 1}/{len(ds)} examples", flush=True)

    if args.storage == "hf":
        write_clips(args.out_dir, rows)
        print(f"Done. Wrote {len(rows)} clips as an arrow dataset in {args.out_dir}")
    else:
        print(f"Done. Generated {len(ds)} audio files in {args.out_dir}")


if __name__ == "__main__":
    main()
