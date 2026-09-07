"""
Single-speaker TTS inference for knowledge benchmark questions.

Used by ChatterboxSingleSpeakerInference job via subprocess
(to run inside the chatterbox venv).
"""

import argparse
import gc
import os
import random

import numpy as np
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
    """Write (index, samples, sample_rate) triples as one arrow dataset. Mirrors clip_store.

    ``samples`` stays a float32 NUMPY array. It used to be ``[float(x) for x in samples]``, which
    boxes every audio sample as a Python float -- ~8x the memory of the array, allocated for ALL
    clips at once just before ``from_dict``. On the 12,000-prompt rehearsal corpus that is a few
    billion boxed floats: the job ran 2 h 09 m, jumped 11.3 -> 16.1 GB in ten seconds and was
    OOM-killed with zero output (2026-09-07). ``clip_store.write_clips`` -- which this file is a
    deliberate copy of, because the worker runs in a venv with no i6_experiments -- always used
    ``np.asarray``; the copy had silently diverged from the thing it says it mirrors.
    """
    rows = {COL_INDEX: [], COL_AUDIO: [], COL_SR: [], COL_MONOLOGUE: [], COL_TRACE: []}
    seen = set()
    for index, samples, sr in items:
        index = int(index)
        if index in seen:
            # Same rule as clip_store: downstream joins address clips BY INDEX, so a duplicate
            # silently drops one clip and misaligns every row after it.
            raise ValueError(f"duplicate clip index {index} -- downstream joins address clips by index")
        seen.add(index)
        rows[COL_INDEX].append(index)
        rows[COL_AUDIO].append(np.asarray(samples, dtype=np.float32).reshape(-1))
        rows[COL_SR].append(int(sr))
        rows[COL_MONOLOGUE].append(None)
        rows[COL_TRACE].append(None)
    Dataset.from_dict(rows).save_to_disk(out_path)


#: Base seed for both the speaker draw and the per-clip TTS sampling. Changing it regenerates every
#: corpus this worker produces, so treat it as a version, not a knob.
SEED = 42


#: An ``rng_`` speaker is ONE draw from ``SEED`` over ``os.listdir`` order -- so which of the 128
#: user voices the benchmark asks its questions in has been a property of the FILESYSTEM, not of the
#: config. It happens to be stable on this cluster, which is why it was never noticed; the listing is
#: verifiably not in sorted order, so it is stable by luck.
#:
#: Sorting the listing is the obvious fix and is the wrong one HERE: with ``sorted()`` the same seed
#: draws ``prompt_12_prompt_0_voice_4`` instead, i.e. it would silently re-voice the benchmark and
#: make every existing knowledge number non-comparable with anything measured afterwards. So record
#: the voice that was actually used instead. This changes no past result and makes it reproducible
#: rather than incidental. `speaker_name` IS hashed, so the pin cannot live in the recipe without
#: re-hashing (and re-running) every benchmark that has ever been scored.
#:
#: Verified 2026-09-07 against the log of a finished ChatterboxSingleSpeakerInference:
#:     Using speaker: .../user_voices/prompt_9_prompt_0_voice_7.wav
PINNED_RNG_SPEAKERS = {"user_voices/rng_a": "prompt_9_prompt_0_voice_7.wav"}


def resolve_speaker_path(speaker_dir: str, speaker_name: str) -> str:
    """Resolve speaker path, supporting rng_ prefix for random selection."""
    based = os.path.basename(speaker_name)
    if based.startswith("rng_"):
        dirname = os.path.dirname(speaker_name)
        search_dir = os.path.join(speaker_dir, dirname) if dirname else speaker_dir
        pinned = PINNED_RNG_SPEAKERS.get(speaker_name)
        if pinned is not None:
            path = os.path.join(search_dir, pinned)
            assert os.path.exists(path), (
                f"pinned speaker {path} is missing. It is the voice every knowledge benchmark to "
                f"date was measured in -- falling back to a fresh draw would re-voice the benchmark "
                f"silently, so refuse instead."
            )
            return path
        # sorted(): an unsorted os.listdir makes the seeded draw depend on filesystem order, which
        # is the bug this whole block exists for. The corpus worker (chatterbox_inference.py) sorts
        # for exactly this reason; this copy did not.
        wavs = sorted(f for f in os.listdir(search_dir) if f.endswith(".wav"))
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

    random.seed(SEED)  # speaker choice
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
            # Seed PER CLIP, from the clip's index -- not once per run. `model.generate` is a
            # SAMPLING TTS, so without a torch seed the benchmark's questions are different audio on
            # every regeneration: measured 2026-08-03, two runs of this job on the same subsample
            # gave 61 of 64 clips differing in LENGTH (worst per-sample diff 0.725, where a PCM-16
            # step is 6.1e-5). That put an unmeasured noise floor under every knowledge number and
            # made a storage-format A/B uninterpretable.
            #
            # Index-derived rather than run-level so a clip's audio does not depend on iteration
            # order: a shard, a resumed run, or a re-run of a subset all reproduce the same audio
            # for the same clip.
            torch.manual_seed(SEED + i)
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
