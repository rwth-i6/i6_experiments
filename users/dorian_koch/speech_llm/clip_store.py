"""One-file-per-clip vs one-dataset-per-job: the benchmark clip store.

The benchmark chain used to write **one wav per example** at three points (question TTS, model reply,
and the symlink merge that joins shards). On a 1000-example run that is ~2000 inodes per benchmark,
and ``/hpcwork/p0023999`` is a *shared* project volume whose binding limit is inodes, not bytes --
5,242,880 soft, which we hit on 2026-08-01 and which killed 33 jobs with ``Errno 122``. Stored as one
arrow dataset the same run costs ~3 inodes.

Both layouts are supported **forever**, and which one you have is decided by looking at the path, not
by a flag:

    clips = open_clips(some_path)     # arrow dataset OR dir of {i}.wav -- caller does not care
    audio, sr = clips[7]

That matters because every already-finished benchmark output on disk is the old layout, and those
results must keep reproducing. ``storage="wav"`` remains the default everywhere; new runs opt in to
``storage="hf"``.

Clips are addressed by **integer index**, not filename: the whole chain (transcription, grading,
reference alignment) joins model output back to the reference dataset by row number, so the index is
identity and must survive the round trip. The arrow layout stores it as an explicit ``index`` column
rather than relying on row order, so a sharded write can be reassembled in any order.

⚠ The standalone worker scripts (``whisper_benchmark_inference.py``,
``chatterbox_benchmark_inference.py``, ...) CANNOT import this module -- they run under a *job's*
venv, which has neither sisyphus nor i6_experiments, and importing ``common.py`` would drag in
``from sisyphus import tk``. They carry a small commented copy of the reader instead. That
duplication is deliberate; see the note in ``chatterbox_inference.py``.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

#: Column names in the arrow layout. ``audio`` holds the raw samples; keeping the sample rate
#: per-row (rather than as dataset metadata) means a mixed-rate merge is representable instead of
#: silently wrong.
COL_INDEX = "index"
COL_AUDIO = "audio"
COL_SR = "sampling_rate"

#: Sidecar payloads the merge step must carry alongside the audio. The inner-monologue ``<i>.txt``
#: was dropped once by a .wav-only merge and every MCQ answer failed to parse, so these are columns
#: in the arrow layout rather than an afterthought.
COL_MONOLOGUE = "monologue"
COL_TRACE = "trace_json"

_WAV_RE = re.compile(r"^(\d+)\.wav$")


def is_clip_dataset(path: str | os.PathLike) -> bool:
    """True if ``path`` is an arrow clip dataset, False if it is a dir of ``{i}.wav``.

    Sniffs for the marker ``datasets.save_to_disk`` always writes. Anything else -- including a
    non-existent path -- is treated as the legacy wav layout, so a caller pointed at a directory
    that has not been produced yet still takes the old code path rather than erroring here.
    """
    p = Path(path)
    return (p / "dataset_info.json").is_file() or (p / "state.json").is_file()


def write_clips(out_path, items, *, monologues=None, traces=None) -> None:
    """Write ``items`` -- an iterable of ``(index, samples, sample_rate)`` -- as one arrow dataset.

    ``samples`` is a 1-D float32 array (mono). ``monologues`` / ``traces`` are optional
    ``{index: str}`` maps for the inner-monologue and RAG-trace sidecars.

    Raises on a duplicate index: two clips claiming the same row would silently drop one and
    misalign every downstream join, which is exactly the class of bug that is invisible in a
    metric.
    """
    from datasets import Dataset

    monologues = monologues or {}
    traces = traces or {}
    rows = {COL_INDEX: [], COL_AUDIO: [], COL_SR: [], COL_MONOLOGUE: [], COL_TRACE: []}
    seen: set[int] = set()
    for index, samples, sr in items:
        index = int(index)
        if index in seen:
            raise ValueError(f"duplicate clip index {index} -- downstream joins address clips by index")
        seen.add(index)
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        rows[COL_INDEX].append(index)
        rows[COL_AUDIO].append(arr.tolist())
        rows[COL_SR].append(int(sr))
        rows[COL_MONOLOGUE].append(monologues.get(index))
        rows[COL_TRACE].append(traces.get(index))

    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    Dataset.from_dict(rows).save_to_disk(out_path)


class _WavDirClips(Mapping):
    """The legacy layout: ``{i}.wav`` files addressed by their integer stem."""

    def __init__(self, path):
        self._dir = Path(path)
        self._index = {}
        if self._dir.is_dir():
            for entry in os.scandir(self._dir):
                m = _WAV_RE.match(entry.name)
                if m:
                    self._index[int(m.group(1))] = entry.path

    def __getitem__(self, key):
        import soundfile as sf

        audio, sr = sf.read(self._index[int(key)], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, int(sr)

    def sidecar(self, key, kind="monologue"):
        """Read the ``<i>.txt`` / ``<i>.json`` sidecar next to a clip, or None."""
        suffix = ".txt" if kind == "monologue" else ".json"
        p = self._dir / f"{int(key)}{suffix}"
        return p.read_text() if p.is_file() else None

    def __iter__(self):
        return iter(sorted(self._index))

    def __len__(self):
        return len(self._index)


class _ArrowClips(Mapping):
    """The arrow layout, addressed by the explicit ``index`` column (not row order)."""

    def __init__(self, path):
        from datasets import load_from_disk

        self._ds = load_from_disk(str(path))
        self._row_of = {int(v): row for row, v in enumerate(self._ds[COL_INDEX])}

    def __getitem__(self, key):
        row = self._ds[self._row_of[int(key)]]
        return np.asarray(row[COL_AUDIO], dtype=np.float32), int(row[COL_SR])

    def sidecar(self, key, kind="monologue"):
        col = COL_MONOLOGUE if kind == "monologue" else COL_TRACE
        if col not in self._ds.column_names:
            return None
        return self._ds[self._row_of[int(key)]][col]

    def __iter__(self):
        return iter(sorted(self._row_of))

    def __len__(self):
        return len(self._row_of)


def open_clips(path) -> Mapping:
    """Open a clip store in either layout. Returns an int-keyed mapping to ``(samples, sr)``.

    The returned object also exposes ``.sidecar(index, kind)`` for the monologue / trace payloads,
    so a consumer that needs them does not have to know the layout either.
    """
    return _ArrowClips(path) if is_clip_dataset(path) else _WavDirClips(path)


def materialise_clips(src, out_dir) -> None:
    """Write an arrow clip store back out as ``{i}.wav`` files in ``out_dir``.

    The bridge for consumers we do not want to change: the five offline drivers in ``moshi_family``
    take a manifest of wav paths and run in their own venvs, so under ``storage="hf"`` the input is
    unpacked to scratch wavs for them rather than teaching each driver to read arrow. Costs inodes
    only for the lifetime of the job, not in the durable output.
    """
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = open_clips(src)
    for i in clips:
        audio, sr = clips[i]
        sf.write(out_dir / f"{i}.wav", audio, sr)
        mono = clips.sidecar(i, "monologue")
        if mono:
            (out_dir / f"{i}.txt").write_text(mono)


def merge_clip_datasets(out_path, in_paths) -> None:
    """Concatenate sharded arrow clip stores into one.

    This is what replaces ``MergeMoshiOutputsViaSymlinks`` under ``storage="hf"``: the symlink merge
    spends one inode per clip *again*, doubling the cost of every sharded run, whereas concatenating
    arrow shards costs a handful of files total.
    """
    from datasets import concatenate_datasets, load_from_disk

    parts = [load_from_disk(str(p)) for p in in_paths]
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError(f"no non-empty shards to merge from {list(in_paths)}")
    merged = concatenate_datasets(parts)
    seen = merged[COL_INDEX]
    if len(set(seen)) != len(seen):
        raise ValueError("shards overlap: the same clip index appears in more than one shard")
    merged.save_to_disk(str(out_path))
