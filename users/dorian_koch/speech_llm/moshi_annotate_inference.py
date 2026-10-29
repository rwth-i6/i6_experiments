"""
moshi_annotate_inference.py — Moshi annotation using whisper_timestamped.

Two modes:
  1. Arrow-native (new, default): reads a Chatterbox TTS out_hf dataset,
     annotates each row in-memory, writes an HF dataset with columns:
       audio_assistant, audio_user  (HF Audio(), mono per speaker)
       alignments  (list of [text, [start_s, end_s], "SPEAKER_MAIN"])
       duration  (float, seconds)
       with --annotate_user (MoshiAnnotate(user_words=True), every build from 2026-10-12): also the USER
       channel's words, by the same Whisper call, in their own columns -- ``alignments_other`` (speaker
       :data:`USER_SPEAKER`) and ``other_ok`` -- the layout the JRE/Open Yap corpora use for the other side.
     (Legacy datasets instead have a single stereo "audio" column; the
      moshi_arrow_dataset consumer reads both formats.)
  2. Legacy file mode: converts to wav+json on disk (kept for compat).

The core Whisper alignment is factored into annotate_array() which is
array-in / alignment-list-out and can be called without touching disk.
"""

import argparse
import faulthandler
import gc
import json
import logging
import os
import subprocess
import sys
import threading
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

from datasets import load_from_disk, Dataset, Features, List, Value, Audio

try:  # imported as part of the package (tests)
    from .clip_codec import wav_to_flac
except ImportError:  # run as a script under the job venv: the sibling file
    from clip_codec import wav_to_flac
from concurrent.futures import ProcessPoolExecutor
import functools

_log = logging.getLogger(__name__)

# Bound on CUDA init + Whisper load. A cached `medium` load on a healthy GPU is
# seconds; 300s only trips on a wedged driver. Overridable for debugging.
LOAD_TIMEOUT_S = int(os.environ.get("MOSHI_LOAD_TIMEOUT_S", "300"))


def _watchdog(seconds, label):
    """Force-kill this process if `label` is not cancelled within `seconds`.

    Bounds CUDA calls that can hang *inside the driver*: a SIGALRM handler or
    future.result(timeout=) cannot fire there, because the C call never returns
    control to the Python interpreter. A daemon thread calling os._exit is the
    only thing that reliably fires. Returns a cancel() callable.
    """
    done = threading.Event()

    def _worker():
        if not done.wait(seconds):  # deadline passed, watched call still running
            _log.error("WATCHDOG: %s did not finish within %ds; aborting.", label, seconds)
            faulthandler.dump_traceback()  # dumps the hung main thread's stack to stderr
            sys.stderr.flush()
            os._exit(2)  # bypass atexit/cleanup (which would touch CUDA)

    threading.Thread(target=_worker, name="load-watchdog", daemon=True).start()
    return done.set


# NOTE: deliberately duplicated, do not "consolidate" into common.py. This file is a standalone
# worker script: the Job ships it to a compute node and runs it with the *job's own* venv, which has
# neither sisyphus nor i6_experiments importable (common.py's first line is `from sisyphus import
# tk`). A shared import here would break the job at startup. Keep the copy small and identical.
def _audio_cell(samples, sample_rate):
    """An ``Audio()`` cell for ``samples``: the very WAV ``datasets`` would encode, stored as exact FLAC when that WAV
    is PCM_16 (2026-10-16; ``clip_codec.wav_to_flac``: identical decoded arrays, ~21% of the bytes)."""
    cell = {"array": samples, "sampling_rate": sample_rate}
    flac = wav_to_flac(Audio().encode_example(cell)["bytes"])
    return cell if flac is None else {"bytes": flac, "path": None}


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


#: Speaker label of the USER channel's words (``alignments_other``). The loaders ignore the field (the sides are
#: separated by COLUMN); a distinct label keeps a leak into ``alignments`` visible to the one-speaker asserts.
USER_SPEAKER = "user"


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
        try:
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
        finally:
            transcribe_mod.get_vad_segments = old_get_vad_segments
    else:
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
    sub_shard: int | None = None,
    sub_num_shards: int | None = None,
    whisper_model: str = "medium",
    language: str = "en",
    keep_silence_in_segments: float = 1.0,
    keep_columns: list[str] | None = None,
    annotate_user: bool = False,
):
    """Read a Chatterbox TTS out_hf dataset and write an annotated HF dataset.

    Each row has columns:
        audio          {"array": float32[], "sampling_rate": int}  (stereo)
        alignments     list of [text, [start, end], "SPEAKER_MAIN"]
        duration       float

    ``annotate_user`` (user-text head, 2026-10-12): also ``alignments_other`` (the user channel's words: the SAME
    ``annotate_array`` call on ``stereo[1]``, speaker :data:`USER_SPEAKER`) and ``other_ok`` (True iff that pass
    returned >= 1 word). A failed user pass keeps the row (``other_ok`` False, no words) and is counted, so the row
    set is the one an assistant-only build gives. The user's words never enter ``alignments``.
    """
    os.makedirs(out_dir, exist_ok=True)

    dataset: Dataset = load_from_disk(in_hf_path)
    if shard_idx is not None and num_shards is not None:
        dataset = dataset.shard(num_shards=num_shards, index=shard_idx)
    if sub_shard is not None and sub_num_shards is not None:
        # Per-GPU fan-out inside the Sisyphus shard: a CONTIGUOUS slice, so concatenating the
        # parts in order reproduces the single-worker row order (the same cut as
        # datasets' shard(contiguous=True)).
        div, mod = divmod(len(dataset), sub_num_shards)
        start = sub_shard * div + min(sub_shard, mod)
        end = start + div + (1 if sub_shard < mod else 0)
        dataset = dataset.select(range(start, end))
        _log.info("sub-shard %d/%d: rows [%d, %d)", sub_shard, sub_num_shards, start, end)
        assert len(dataset) > 0, f"sub-shard {sub_shard}/{sub_num_shards} is empty -- fewer rows than GPUs"

    node = os.uname().nodename
    _log.info("Loading Whisper model %s on node %s …", whisper_model, node)

    # Snapshot GPU state so any hang records the node's GPU condition. The 30s
    # subprocess timeout is a real OS-level kill, so this cannot hang us itself.
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30)
        _log.info("nvidia-smi at load:\n%s", smi.stdout)
    except Exception as e:
        _log.warning("nvidia-smi failed: %r", e)

    # Watchdog covers cuda.is_available() AND load_model — both can stall in the
    # driver on a bad node.
    cancel = _watchdog(LOAD_TIMEOUT_S, f"CUDA init + whisper.load_model({whisper_model}) on {node}")
    try:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        w_model = whisper.load_model(whisper_model, device=device)
    finally:
        cancel()
    _log.info("Whisper model loaded on %s.", device)

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

    # Progress tracking and failure counters (map is single-process, so mutable
    # dicts are safe for accumulation across process_row calls).
    _progress = {"done": 0, "total": len(dataset)}
    _counters = {"missing_audio": 0, "annotate_error": 0, "user_annotate_error": 0, "user_words": 0}
    _write_progress(0, _progress["total"])

    def _tick():
        _progress["done"] += 1
        if _progress["done"] % 8 == 0 or _progress["done"] == _progress["total"]:
            _write_progress(_progress["done"], _progress["total"])

    _FAILED_SENTINEL = {"audio_assistant": None, "audio_user": None, "alignments": [], "duration": 0.0}
    if annotate_user:
        _FAILED_SENTINEL.update({"alignments_other": [], "other_ok": False})

    def process_row(row: dict) -> dict:
        _tick()
        # Extract speaker tracks from the HF Audio-feature structure.
        # ChatterboxInference stores speaker_audio as a dict of lists.
        d = row.get("speaker_audio", {})
        if d:
            speaker_tracks = [dict(zip(d.keys(), vals)) for vals in zip(*d.values())]
        else:
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
            _counters["missing_audio"] += 1
            return _FAILED_SENTINEL

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
            _log.exception("Error annotating row; marking as failed.")
            _counters["annotate_error"] += 1
            return _FAILED_SENTINEL

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

        user_cols = {}
        if annotate_user:
            # The USER channel: the same Whisper call, its own column (never ``alignments``: the inner monologue).
            try:
                user_aligns = annotate_array(stereo[1], sample_rate, w_model, params, language=language)
            except Exception as exc:
                if "cuda" in repr(exc).lower():
                    raise
                _log.exception("Error annotating the user channel; row kept with other_ok=False.")
                _counters["user_annotate_error"] += 1
                user_aligns = []
            user_cols = {
                "alignments_other": [
                    {"text": str(a[0]), "start": float(a[1][0]), "end": float(a[1][1]), "speaker": USER_SPEAKER}
                    for a in user_aligns
                ],
                "other_ok": bool(user_aligns),
            }
            _counters["user_words"] += len(user_aligns)

        return {
            # Two mono HF Audio() columns (torchcodec-encoded in moshi_venv),
            # mirroring ChatterboxInference's per-speaker Audio storage.
            "audio_assistant": _audio_cell(np.ascontiguousarray(assistant_audio, dtype=np.float32), sample_rate),
            "audio_user": _audio_cell(np.ascontiguousarray(user_audio, dtype=np.float32), sample_rate),
            "alignments": aligns_struct,
            "duration": duration,
            **user_cols,
        }

    rows_in = len(dataset)
    _log.info("Annotating %d rows …", rows_in)
    import datasets as _ds_lib

    _ds_lib.disable_caching()
    # Store per-speaker audio as HF Audio() (torchcodec-encoded) rather than raw
    # nested float lists: far cheaper in memory/on disk and the canonical format.
    # Passthrough columns (e.g. RAG ``reference_text``): kept verbatim with their original
    # feature type, NOT removed and NOT produced by process_row -- datasets.map preserves any
    # input column absent from remove_columns. Their schema must be declared in out_features.
    passthrough = [c for c in (keep_columns or []) if c in dataset.column_names]
    kept = ("id", *passthrough)
    out_features = Features(
        {
            "id": Value("string"),
            "audio_assistant": Audio(),
            "audio_user": Audio(),
            "alignments": ALIGNMENT_FEATURE,
            "duration": Value("float32"),
            **({"alignments_other": ALIGNMENT_FEATURE, "other_ok": Value("bool")} if annotate_user else {}),
            **{c: dataset.features[c] for c in passthrough},
        }
    )
    annotated = dataset.map(
        process_row,
        remove_columns=[c for c in dataset.column_names if c not in kept],
        features=out_features,
        desc="annotating",
        writer_batch_size=4,
    )
    # Drop rows that failed (audio None) — includes both missing-audio and annotate-error rows
    annotated = annotated.filter(lambda row: row["audio_assistant"] is not None)
    annotated.save_to_disk(out_dir)
    rows_out = len(annotated)
    _log.info(
        "Wrote annotated dataset to %s (%d/%d rows; dropped %d missing-audio, %d annotate-errors).",
        out_dir,
        rows_out,
        rows_in,
        _counters["missing_audio"],
        _counters["annotate_error"],
    )
    # Persist per-job stats so they survive after the job directory is inspected.
    import json as _json_stats

    stats = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "dropped_missing_audio": _counters["missing_audio"],
        "dropped_annotate_error": _counters["annotate_error"],
    }
    if annotate_user:
        # counted over every row the map saw (dropped rows included); other_ok among the kept rows is what trains
        stats.update(
            {
                "user_annotate_error": _counters["user_annotate_error"],
                "user_words": _counters["user_words"],
                "other_ok_rows": int(sum(annotated["other_ok"])),
            }
        )
    with open(os.path.join(out_dir, "stats.json"), "w") as _sf:
        _json_stats.dump(stats, _sf, indent=2)


# ---------------------------------------------------------------------------
# User-only add-on pass over an ALREADY annotated dataset (backfill, 2026-10-13)
# ---------------------------------------------------------------------------


def annotate_user_only(
    in_hf_path: str,
    out_dir: str,
    *,
    sub_shard: int | None = None,
    sub_num_shards: int | None = None,
    whisper_model: str = "medium",
    language: str = "en",
    keep_silence_in_segments: float = 1.0,
):
    """The ``--annotate_user`` pass alone, over an annotated dataset (``annotate_hf_to_hf``'s output) that has none:
    per row the SAME ``annotate_array`` call (same model, Params, decoding) on the stored ``audio_user`` (the user
    channel the codes builder reads; lossless WAV), written as a row-aligned SIDECAR ``{id, alignments_other,
    other_ok}`` -- the columns ``--annotate_user`` adds, same feature, speaker :data:`USER_SPEAKER`, ``other_ok`` = >= 1
    word. The annotated dataset (its ``alignments``: the assistant pass) is not touched, so a corpus built from it
    keeps its assistant words bit-identical (``MoshiAnnotateUserWords``; 2026-10-13: backfill v6). Reads the
    arrow table's bytes (soundfile), never a datasets Audio() decode, so no torchcodec is needed."""
    import io

    os.makedirs(out_dir, exist_ok=True)
    table = load_from_disk(in_hf_path).data.table
    for col in ("id", "audio_user"):
        if col not in table.column_names:
            raise RuntimeError(f"{in_hf_path}: no {col!r} column (an annotate_hf_to_hf output?): {table.column_names}")
    lo, hi = 0, table.num_rows
    if sub_shard is not None and sub_num_shards is not None:
        div, mod = divmod(table.num_rows, sub_num_shards)  # contiguous, as annotate_hf_to_hf's sub-shards
        lo = sub_shard * div + min(sub_shard, mod)
        hi = lo + div + (1 if sub_shard < mod else 0)
        assert hi > lo, f"sub-shard {sub_shard}/{sub_num_shards} is empty"
    node = os.uname().nodename
    cancel = _watchdog(LOAD_TIMEOUT_S, f"CUDA init + whisper.load_model({whisper_model}) on {node}")
    try:
        device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
        w_model = whisper.load_model(whisper_model, device=device)
    finally:
        cancel()
    params = Params(
        egs=Path("/dev/null"),
        verbose=False,
        lang=language,
        whisper_model=whisper_model,
        keep_silence_in_segments=keep_silence_in_segments,
        rerun_errors=False,
        shards=1,
        shard=0,
    )  # exactly annotate_hf_to_hf's
    _log.info(
        "user-only pass on %s: rows [%d, %d) of %d, Whisper %s on %s (%s)",
        in_hf_path,
        lo,
        hi,
        table.num_rows,
        whisper_model,
        device,
        node,
    )
    import time as _time

    ids, words, oks = [], [], []
    counters = {"user_annotate_error": 0, "user_words": 0, "audio_sec": 0.0}
    t0 = _time.time()
    _write_progress(0, hi - lo)
    for k, i in enumerate(range(lo, hi)):
        ids.append(table.column("id")[i].as_py())
        x, sr = sf.read(io.BytesIO(table.column("audio_user")[i].as_py()["bytes"]), dtype="float32", always_2d=True)
        x = np.ascontiguousarray(x.mean(axis=1), dtype=np.float32)
        counters["audio_sec"] += x.shape[0] / sr
        try:
            aligns = annotate_array(x, sr, w_model, params, language=language)
        except Exception as exc:
            if "cuda" in repr(exc).lower():
                raise
            _log.exception("Error annotating the user channel of row %d; kept with other_ok=False.", i)
            counters["user_annotate_error"] += 1
            aligns = []
        words.append(
            [
                {"text": str(a[0]), "start": float(a[1][0]), "end": float(a[1][1]), "speaker": USER_SPEAKER}
                for a in aligns
            ]
        )
        oks.append(bool(aligns))
        counters["user_words"] += len(aligns)
        if (k + 1) % 50 == 0 or i + 1 == hi:
            _write_progress(k + 1, hi - lo)
            el = _time.time() - t0
            _log.info("user-only: %d/%d rows, %.0f s, %.3f s/row", k + 1, hi - lo, el, el / (k + 1))
    feats = Features({"id": Value("string"), "alignments_other": ALIGNMENT_FEATURE, "other_ok": Value("bool")})
    Dataset.from_dict({"id": ids, "alignments_other": words, "other_ok": oks}, features=feats).save_to_disk(out_dir)
    stats = {
        "rows": len(ids),
        "other_ok_rows": int(sum(oks)),
        "user_annotate_error": counters["user_annotate_error"],
        "user_words": counters["user_words"],
        "whisper_sec": round(_time.time() - t0, 1),
        "audio_sec": round(counters["audio_sec"], 1),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    _log.info("user-only pass wrote %s: %s", out_dir, stats)


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
    parser.add_argument("--sub_shard", type=int, required=False, help="per-GPU fan-out inside the shard (backlog G1)")
    parser.add_argument("--sub_num_shards", type=int, required=False)
    parser.add_argument(
        "--mode",
        choices=["arrow", "legacy", "user_only"],
        default="arrow",
        help="'arrow' emits an HF dataset (new default); 'legacy' writes wav+json files; 'user_only': --in_hf is an "
        "annotated dataset, write the user channel's words as a sidecar (MoshiAnnotateUserWords).",
    )
    parser.add_argument("--whisper_model", default="medium")
    parser.add_argument(
        "--annotate_user",
        action="store_true",
        help="also the user channel's words -> alignments_other + other_ok (MoshiAnnotate(user_words=True))",
    )
    parser.add_argument(
        "--keep_columns",
        nargs="*",
        default=None,
        help="Input columns to pass through to the output unchanged (e.g. RAG reference_text).",
    )
    args = parser.parse_args()

    init_logging(False)

    if args.mode == "user_only":
        assert args.in_hf_shard is None and not args.keep_columns and not args.annotate_user, (
            "user_only: sub-shards only"
        )
        annotate_user_only(
            args.in_hf,
            args.out_dir,
            sub_shard=args.sub_shard,
            sub_num_shards=args.sub_num_shards,
            whisper_model=args.whisper_model,
        )
    elif args.mode == "arrow":
        annotate_hf_to_hf(
            in_hf_path=args.in_hf,
            out_dir=args.out_dir,
            shard_idx=args.in_hf_shard,
            num_shards=args.in_hf_num_shards,
            sub_shard=args.sub_shard,
            sub_num_shards=args.sub_num_shards,
            whisper_model=args.whisper_model,
            keep_columns=args.keep_columns,
            annotate_user=args.annotate_user,
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
