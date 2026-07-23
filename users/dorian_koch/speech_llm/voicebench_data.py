"""Materialize one VoiceBench subset (runs in a venv with `datasets` + soundfile + torchaudio).

Loads exactly as VoiceBench's main.py (`hlt-lab/voicebench`), but decodes audio ourselves with soundfile
(`Audio(decode=False)`) to avoid `datasets`' torchcodec decode path (broken in our cu126 venv). VoiceBench
audio is stored as WAV, so soundfile reads it directly; we downmix + resample to 16 kHz (the rate their
main.py casts to). Writes `{i}.wav` (dataset order) + the non-audio columns as an HF dataset
(`out_ref`, index-aligned) carrying every field the scorers need (prompt, reference, ifeval kwargs, ...).
"""

import argparse
import io
import os

import soundfile as sf
import torch
import torchaudio
from datasets import Audio, load_dataset

TARGET_SR = 16000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_ref", required=True)
    args = p.parse_args()

    from datasets import concatenate_datasets, get_dataset_split_names

    avail = get_dataset_split_names("hlt-lab/voicebench", args.subset)
    if args.split in avail:
        ds = load_dataset("hlt-lab/voicebench", args.subset, split=args.split)
    else:
        # Some subsets (e.g. mmsu) have no `test` split -- their data is sharded into per-subject
        # splits (law/biology/physics/...). Concatenate all of them (= the full subset) so scoring
        # sees every question, matching VoiceBench's intent.
        print(
            f"[voicebench_data] subset {args.subset!r} has no split {args.split!r}; "
            f"concatenating all splits {avail}",
            flush=True,
        )
        ds = concatenate_datasets(
            [load_dataset("hlt-lab/voicebench", args.subset, split=s) for s in avail]
        )
    ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes -> we decode w/ soundfile (no torchcodec)

    os.makedirs(args.out_dir, exist_ok=True)
    for i, ex in enumerate(ds):
        a = ex["audio"]
        raw = a["bytes"] if a.get("bytes") is not None else open(a["path"], "rb").read()
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != TARGET_SR:
            data = torchaudio.functional.resample(torch.from_numpy(data), sr, TARGET_SR).numpy()
        sf.write(os.path.join(args.out_dir, f"{i}.wav"), data, TARGET_SR)
        if (i + 1) % 200 == 0:
            print(f"wrote {i + 1}/{len(ds)} prompt wavs", flush=True)

    ref = ds.remove_columns(["audio"])
    ref.save_to_disk(args.out_ref)
    print(f"Done: {len(ds)} prompts -> {args.out_dir}; ref cols {ref.column_names} -> {args.out_ref}", flush=True)


if __name__ == "__main__":
    main()
