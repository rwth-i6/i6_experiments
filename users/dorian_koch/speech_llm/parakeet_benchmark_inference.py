"""
Parakeet (NeMo TDT) ASR transcription for knowledge benchmark replies.

Used by ``WhisperTranscription(asr="nvidia/parakeet-tdt-...")`` via subprocess, under the NeMo venv
(``fdb.fdb_asr_venv``). Same CLI and output schema as ``whisper_benchmark_inference.py``, so the job,
the per-GPU fan-out and every reader stay unchanged. No VAD: Parakeet does not hallucinate on silence
the way Whisper does, so the reply is transcribed whole.
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_from_disk

try:
    from .clip_codec import decode_row
except ImportError:  # run as a script under the job venv: the sibling file
    from clip_codec import decode_row


TARGET_SR = 16000
# full attention is quadratic in length; past this, switch to local attention (NeMo's long-form setting)
LOCAL_ATTN_ABOVE_S = 600


class _LazyArrowClips:
    """Arrow clip store decoded one clip per lookup (as in whisper_benchmark_inference.py)."""

    def __init__(self, ds):
        self._ds = ds
        self._row = {int(i): k for k, i in enumerate(ds["index"])}

    def __contains__(self, i):
        return i in self._row

    def __getitem__(self, i):
        r = self._ds[self._row[i]]
        return decode_row(r), int(r["sampling_rate"])


def open_clips(path: str):
    """``{index: (samples, sr) | wav path}`` over an arrow clip store or a dir of ``<i>.wav`` (clip_store layouts)."""
    p = str(path)
    if os.path.isfile(os.path.join(p, "dataset_info.json")) or os.path.isfile(os.path.join(p, "state.json")):
        return _LazyArrowClips(load_from_disk(p))
    out = {}
    for name in os.listdir(p) if os.path.isdir(p) else []:
        if name.endswith(".wav") and name[:-4].isdigit():
            out[int(name[:-4])] = os.path.join(p, name)
    return out


def load_audio(entry) -> np.ndarray:
    if isinstance(entry, tuple):
        audio, sr = entry
    else:
        audio, sr = sf.read(entry, dtype="float32")
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)
    if int(sr) != TARGET_SR:
        audio = torchaudio.functional.resample(torch.from_numpy(audio), int(sr), TARGET_SR).numpy()
    return audio.astype(np.float32)


def load_model(name: str, device: str):
    """Restore the ``.nemo`` from the HF cache without network (jobs run offline)."""
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import snapshot_download

    snap = snapshot_download(name, local_files_only=True)
    nemo_files = glob.glob(os.path.join(snap, "*.nemo"))
    assert len(nemo_files) == 1, (name, snap, nemo_files)
    model = nemo_asr.models.ASRModel.restore_from(nemo_files[0], map_location=device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Parakeet transcription for benchmark responses")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--reference_data", required=True)
    parser.add_argument("--out_json", required=True)
    parser.add_argument("--model", default="nvidia/parakeet-tdt-0.6b-v3")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()
    mine = (lambda i: True) if args.num_shards is None else (lambda i: i % args.num_shards == args.shard)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    model = load_model(args.model, device)
    print(f"Loaded {args.model} on {device} in {time.time() - t0:.1f}s", flush=True)

    ref_ds = load_from_disk(args.reference_data)
    clips = open_clips(args.in_dir)
    valid = [(i, ex) for i, ex in enumerate(ref_ds) if i in clips and mine(i)]
    print(f"Transcribing {len(valid)}/{len(ref_ds)} clips (batch_size={args.batch_size})", flush=True)

    audios = {i: load_audio(clips[i]) for i, _ in valid}
    longest = max((a.size for a in audios.values()), default=0) / TARGET_SR
    if longest > LOCAL_ATTN_ABOVE_S:
        print(f"longest clip {longest:.0f}s > {LOCAL_ATTN_ABOVE_S}s: local attention", flush=True)
        model.change_attention_model("rel_pos_local_attn", [256, 256])

    # Sub-10 ms / empty audio: an empty hypothesis (a benchmark miss), as in the Whisper worker.
    todo = [i for i, _ in valid if audios[i].size >= TARGET_SR // 100]
    texts = {i: "" for i, _ in valid}
    audio_s = sum(audios[i].size for i in todo) / TARGET_SR
    t1 = time.time()
    for k in range(0, len(todo), args.batch_size):
        chunk = todo[k : k + args.batch_size]
        with torch.inference_mode():
            hyps = model.transcribe([audios[i] for i in chunk], batch_size=args.batch_size, verbose=False)
        for i, h in zip(chunk, hyps):
            texts[i] = (h.text if hasattr(h, "text") else str(h)).strip()
        if (k // args.batch_size) % 10 == 0:
            print(f"Transcribed {min(k + args.batch_size, len(todo))}/{len(todo)} clips", flush=True)
    dt = time.time() - t1
    print(
        f"[asr-speed] {args.model}: {audio_s:.0f}s audio in {dt:.1f}s = {audio_s / max(dt, 1e-9):.0f}x real time",
        flush=True,
    )

    with open(args.out_json, "w") as f, open(args.out_json + ".idx", "w") as idx:
        for i, example in valid:
            idx.write(f"{i}\n")
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "question": example["question"],
                        "answer": example["answer"],
                        "aliases": example["aliases"],
                        "category": example.get("category", "unknown"),
                        "transcription": texts[i],
                    }
                )
                + "\n"
            )
    print(f"Done. Wrote {len(valid)} transcriptions to {args.out_json}")


if __name__ == "__main__":
    main()
