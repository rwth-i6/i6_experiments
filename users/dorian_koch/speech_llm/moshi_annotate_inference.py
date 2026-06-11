"""
moshi_annotate_inference.py — Moshi annotation using whisper_timestamped.

Two modes:
  1. Arrow-native (new, default): reads a Chatterbox TTS out_hf dataset,
     annotates each row in-memory, writes an HF dataset with columns:
       audio  ({"array": float32[], "sampling_rate": int})
       alignments  (list of [text, [start_s, end_s], "SPEAKER_MAIN"])
       duration  (float, seconds)
  2. Legacy file mode: converts to wav+json on disk (kept for compat).

The core Whisper alignment is factored into annotate_array() which is
array-in / alignment-list-out and can be called without touching disk.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torchaudio.functional as F_audio
import torch

# ---------------------------------------------------------------------------
# Import shared helpers from the vendored moshi-finetune fork (read-only).
# We reuse their Whisper setup + VAD patch; we do NOT edit those files.
# ---------------------------------------------------------------------------
from moshi_finetune.annotate import (
    init_logging,
    Params,
    logger,
    torch as _torch,
    whisper,
    old_get_vad_segments,
    SAMPLE_RATE,
)

# whisper_timestamped patch lives in annotate; re-import so it's wired up
import whisper_timestamped
import importlib

transcribe_mod = importlib.import_module("whisper_timestamped.transcribe")

from datasets import load_from_disk, Dataset, Features, List, Value
from concurrent.futures import ProcessPoolExecutor
import functools

_log = logging.getLogger(__name__)


def _write_progress(done, total, path="progress.json"):
    """Write a tiny {done,total} marker so Sisyphus Job.completed_fraction can
    report progress. Lands in cwd, which is the job work dir."""
    import tempfile

    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".progress-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump({"done": int(done), "total": int(total)}, f)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# On-disk schema for the per-row alignments column. Stored as a struct (list of
# dicts) rather than a ragged [text, [start, end], speaker] list so HF datasets
# represents it natively instead of falling back to the Json feature, whose
# decode silently re-parses numeric-looking words (e.g. "100" -> 100) and breaks
# Interleaver._tokenize (word.strip()). See moshi_arrow_dataset._norm_aligns.
ALIGNMENT_FEATURE = List(
    {
        "text": Value("string"),
        "start": Value("float32"),
        "end": Value("float32"),
        "speaker": Value("string"),
    }
)


# ---------------------------------------------------------------------------
# In-memory annotation (B1)
# ---------------------------------------------------------------------------


def annotate_array(
    audio_mono: np.ndarray,
    sr: int,
    w_model,
    params: Params,
    language: str = "en",
) -> list:
    """Annotate a mono audio array; return alignments list.

    Output format (same as process_one's json):
        [ [word_text, [start_s, end_s], "SPEAKER_MAIN"], ... ]

    Raises RuntimeError on CUDA errors; silently returns [] on content errors.
    """
    gc.collect()
    _torch.cuda.empty_cache()

    # Resample to 16 kHz
    x = _torch.from_numpy(audio_mono).float()
    if sr != SAMPLE_RATE:
        x = F_audio.resample(x, sr, SAMPLE_RATE)
    vocals = x.cpu().numpy()

    dur = vocals.shape[-1] / SAMPLE_RATE

    # Apply the VAD-segment patch if requested (same logic as annotate.py process_one)
    if params.keep_silence_in_segments:
        d = int(SAMPLE_RATE * params.keep_silence_in_segments)

        def _patched_get_vad_segments(*args, **kwargs):
            segs = old_get_vad_segments(*args, **kwargs)
            outs = []
            last_end = 0
            for seg in segs:
                outs.append({"start": max(last_end, seg["start"] - d), "end": seg["end"] + d})
                last_end = outs[-1]["end"]
            return outs

        transcribe_mod.get_vad_segments = _patched_get_vad_segments

    pipe_output = whisper.transcribe(
        w_model,
        vocals,
        language=language,
        vad="auditok" if dur > 10 else None,
        best_of=5,
        beam_size=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        verbose=None,
    )

    # Restore original VAD function
    transcribe_mod.get_vad_segments = old_get_vad_segments

    alignments = []
    for segment in pipe_output.get("segments", []):
        for word in segment.get("words", []):
            try:
                alignments.append([word["text"], [word["start"], word["end"]], "SPEAKER_MAIN"])
            except KeyError:
                _log.warning("Missing key in word: %r", word)
    return alignments


# ---------------------------------------------------------------------------
# Arrow-native annotation (B2)
# ---------------------------------------------------------------------------


def annotate_hf_to_hf(
    in_hf_path: str,
    out_dir: str,
    shard_idx: int | None = None,
    num_shards: int | None = None,
    *,
    whisper_model: str = "medium",
    language: str = "en",
    keep_silence_in_segments: float = True,
):
    """Read a Chatterbox TTS out_hf dataset and write an annotated HF dataset.

    Each row has columns:
        audio          {"array": float32[], "sampling_rate": int}  (stereo)
        alignments     list of [text, [start, end], "SPEAKER_MAIN"]
        duration       float
    """
    os.makedirs(out_dir, exist_ok=True)

    dataset: Dataset = load_from_disk(in_hf_path)
    if shard_idx is not None and num_shards is not None:
        dataset = dataset.shard(num_shards=num_shards, index=shard_idx)

    _log.info("Loading Whisper model %s …", whisper_model)
    device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    w_model = whisper.load_model(whisper_model, device=device)

    params = Params(
        egs=Path("/dev/null"),  # unused
        verbose=False,
        lang=language,
        whisper_model=whisper_model,
        keep_silence_in_segments=keep_silence_in_segments,
        rerun_errors=False,
        shards=1,
        shard=0,
    )

    # Progress tracking for Sisyphus Job.completed_fraction (dataset.map runs
    # process_row sequentially in-process, so a simple counter is accurate).
    _progress = {"done": 0, "total": len(dataset)}
    _write_progress(0, _progress["total"])

    def _tick():
        _progress["done"] += 1
        if _progress["done"] % 8 == 0 or _progress["done"] == _progress["total"]:
            _write_progress(_progress["done"], _progress["total"])

    def process_row(row: dict) -> dict:
        _tick()
        # Extract speaker tracks from the HF Audio-feature structure.
        # ChatterboxInference stores speaker_audio as a dict of lists.
        d = row.get("speaker_audio", {})
        if d:
            speaker_tracks = [dict(zip(d.keys(), vals)) for vals in zip(*d.values())]
        else:
            # Fallback: try direct 'audio' column (single speaker)
            speaker_tracks = []

        user_audio = None
        assistant_audio = None
        sample_rate = None

        for track in speaker_tracks:
            speaker_name = track.get("speaker", "")
            audio_array = np.squeeze(np.array(track["audio"]["array"], dtype=np.float32))
            sr = track["audio"]["sampling_rate"]
            if speaker_name == "user":
                user_audio = audio_array
                sample_rate = sr
            elif speaker_name == "assistant":
                assistant_audio = audio_array
                sample_rate = sr

        if user_audio is None or assistant_audio is None:
            _log.warning("Row missing user or assistant audio; skipping.")
            return {"audio": None, "alignments": [], "duration": 0.0}

        # Align lengths with graceful zero-pad (don't crash on minor mismatches)
        max_len = max(len(user_audio), len(assistant_audio))
        if len(assistant_audio) < max_len:
            assistant_audio = np.pad(assistant_audio, (0, max_len - len(assistant_audio)))
        if len(user_audio) < max_len:
            user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))

        # Stereo: channel 0 = assistant (what we annotate), channel 1 = user
        stereo = np.stack([assistant_audio, user_audio], axis=0)  # (2, T)
        duration = stereo.shape[-1] / sample_rate

        try:
            aligns = annotate_array(stereo[0], sample_rate, w_model, params, language=language)
        except Exception as exc:
            if "cuda" in repr(exc).lower():
                raise
            _log.exception("Error annotating row; returning empty alignments.")
            aligns = []

        # Emit alignments as a struct (list of dicts) matching ALIGNMENT_FEATURE
        # so the column is stored natively rather than via the Json fallback.
        aligns_struct = [
            {
                "text": str(a[0]),
                "start": float(a[1][0]),
                "end": float(a[1][1]),
                "speaker": a[2],
            }
            for a in aligns
        ]

        return {
            "audio": {
                "array": stereo.T.tolist(),  # (T, 2) for HF Audio
                "sampling_rate": sample_rate,
            },
            "alignments": aligns_struct,
            "duration": duration,
        }

    _log.info("Annotating %d rows …", len(dataset))
    import datasets as _ds_lib

    _ds_lib.disable_caching()
    annotated = dataset.map(
        process_row,
        remove_columns=[c for c in dataset.column_names if c not in ("id",)],
        desc="annotating",
        writer_batch_size=8,
    )
    # Drop rows that failed (audio=None)
    annotated = annotated.filter(lambda row: row["audio"] is not None)
    # Pin the alignments schema (float32 timestamps, string text) so it is never
    # stored via the Json feature regardless of inference.
    annotated = annotated.cast_column("alignments", ALIGNMENT_FEATURE)
    annotated.save_to_disk(out_dir)
    _log.info("Wrote annotated dataset to %s (%d rows).", out_dir, len(annotated))


# ---------------------------------------------------------------------------
# Legacy file-based path (kept for backward compat, used by old MoshiAnnotate)
# ---------------------------------------------------------------------------


def extract_hf_to_files_multproc(output_dir: str, dataset: Dataset, num_proc: int):
    print(f"{len(dataset)} rows in dataset")
    print(f"sharding into {num_proc}")
    shards = [dataset.shard(num_shards=num_proc, index=i) for i in range(num_proc)]
    with ProcessPoolExecutor(max_workers=num_proc) as executor:
        list(executor.map(functools.partial(extract_hf_to_files, output_dir), shards))


def extract_hf_to_files(output_dir: str, dataset: Dataset):
    from moshi_finetune.annotate import (
        init_logging,
        Params,
        logger,
        torch,
        whisper,
        load_audio_paths,
        process_one,
    )

    num_files_written = 0
    for row in dataset:
        row_id = row["id"]
        d = row["speaker_audio"]
        speaker_tracks = [dict(zip(d.keys(), values)) for values in zip(*d.values())]

        user_audio = None
        assistant_audio = None
        sample_rate = None

        for track in speaker_tracks:
            speaker_name = track["speaker"]
            audio_array = np.squeeze(np.array(track["audio"]["array"], dtype=np.float32))
            sr = track["audio"]["sampling_rate"]
            if speaker_name == "user":
                user_audio = audio_array
                sample_rate = sr
            elif speaker_name == "assistant":
                assistant_audio = audio_array
                sample_rate = sr
            else:
                raise ValueError(f"Invalid speaker name {speaker_name}")

        if user_audio is None or assistant_audio is None:
            raise ValueError("some audio is missing")

        max_len = max(len(user_audio), len(assistant_audio))
        if len(assistant_audio) < max_len:
            assistant_audio = np.pad(assistant_audio, (0, max_len - len(assistant_audio)))
        if len(user_audio) < max_len:
            user_audio = np.pad(user_audio, (0, max_len - len(user_audio)))

        stereo_audio = np.column_stack((assistant_audio, user_audio))
        output_path = os.path.join(output_dir, f"{row_id}.wav")
        assert not os.path.exists(output_path)
        sf.write(output_path, stereo_audio, sample_rate)
        num_files_written += 1

    print(f"Extracted {num_files_written} files to '{output_dir}'")


def hf_to_moshi_format(out_dir: str, dataset: Dataset) -> str:
    import sphn

    out_dir_audios = os.path.join(out_dir, "wav-dir/")
    os.makedirs(out_dir_audios, exist_ok=True)
    extract_hf_to_files_multproc(out_dir_audios, dataset, num_proc=6)
    paths = [str(f) for f in Path(out_dir_audios).glob("*.wav")]
    durations = sphn.durations(paths)
    data_json_path = os.path.join(out_dir, "data.jsonl")
    with open(data_json_path, "w") as fobj:
        for p, d in zip(paths, durations):
            if d is None:
                continue
            json.dump({"path": p, "duration": d}, fobj)
            fobj.write("\n")
    return data_json_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Run moshi annotate Inference")
    parser.add_argument("--in_hf", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--in_hf_shard", type=int, required=False)
    parser.add_argument("--in_hf_num_shards", type=int, required=False)
    parser.add_argument(
        "--mode",
        choices=["arrow", "legacy"],
        default="arrow",
        help="'arrow' emits an HF dataset (new default); 'legacy' writes wav+json files.",
    )
    parser.add_argument("--whisper_model", default="medium")
    args = parser.parse_args()

    init_logging(False)

    if args.mode == "arrow":
        annotate_hf_to_hf(
            in_hf_path=args.in_hf,
            out_dir=args.out_dir,
            shard_idx=args.in_hf_shard,
            num_shards=args.in_hf_num_shards,
            whisper_model=args.whisper_model,
        )
    else:
        # Legacy path
        dataset = load_from_disk(args.in_hf)
        if args.in_hf_shard is not None and args.in_hf_num_shards is not None:
            dataset = dataset.shard(num_shards=args.in_hf_num_shards, index=args.in_hf_shard)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        w_model = whisper.load_model("medium", device=device)
        data_path = hf_to_moshi_format(args.out_dir, dataset)
        print(f"Legacy annotation done. data.jsonl: {data_path}")


if __name__ == "__main__":
    main()
