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
results must keep reproducing. Since 2026-09-10 ``knowledge_benchmark_py`` **defaults to**
``storage="hf"`` so new tags get the arrow layout for free; the 40 pre-existing call sites are pinned
to ``"wav"`` to keep their hashes, and the ``Job``'s own default stays ``"wav"``. ⚠ A new tag
therefore takes the ``"hf"`` write path on its very first run -- which is how the 3,000-clip pack in
``SpeechInference._pack_clips`` OOM-killed all four B6 evals on 2026-09-11. Assume any new
``storage="hf"`` consumer is seeing corpus scale for the first time.

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

import json
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

#: Max float32 samples per arrow chunk. Arrow's list offsets are int32, so one chunk may not exceed
#: 2 GiB of values; 400 M float32 = 1.6 GiB leaves room for the offset/validity buffers.
_MAX_SAMPLES_PER_CHUNK = 400_000_000


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

    **Bounded memory: at most one chunk of audio is ever live.** ``items`` is consumed as a stream
    and each chunk is flushed to a staged dataset as soon as it fills, so peak RSS is set by
    ``_MAX_SAMPLES_PER_CHUNK`` (~1.6 GB) rather than by the size of the corpus. The previous version
    accumulated every clip in ``rows``, then built ``parts`` as a second copy and concatenated as a
    third -- ~3x the corpus. That is what OOM-killed the four 3,000-clip B6 evals on 2026-09-11 in
    ``SpeechInference._pack_clips``, *after* inference had already succeeded: a monotone ramp from
    166 MB to 10.8 GB in 55 s against a 16 GB limit, with ~22 GB wanted. Pass a generator (as
    ``_pack_clips`` does) and the clips are never all in memory at once.

    Chunk boundaries, row order and dtypes are unchanged, so this writes the same dataset the
    accumulating version did -- guarded bit-for-bit by ``tests/check_clip_store.py``.
    """
    import shutil
    import tempfile

    from datasets import Dataset

    monologues = monologues or {}
    traces = traces or {}
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    def _empty():
        return {COL_INDEX: [], COL_AUDIO: [], COL_SR: [], COL_MONOLOGUE: [], COL_TRACE: []}

    # Staging honours $TMPDIR, which SLURM points at node-local NVMe -- so the parts never land on
    # the volume we are writing into. See CLAUDE.md: a tool must stage OFF the volume it is filling.
    tmpdir = tempfile.mkdtemp(prefix="write_clips.")
    loaded = []
    try:
        rows, used, parts = _empty(), 0, []

        def flush():
            nonlocal rows, used
            if not rows[COL_INDEX]:
                return
            part = os.path.join(tmpdir, f"part{len(parts):05d}")
            Dataset.from_dict(rows).save_to_disk(part)
            parts.append(part)
            rows, used = _empty(), 0

        seen: set[int] = set()
        for index, samples, sr in items:
            index = int(index)
            if index in seen:
                raise ValueError(f"duplicate clip index {index} -- downstream joins address clips by index")
            seen.add(index)
            # NOT arr.tolist(): that boxes every sample as a Python float, which costs ~8x the array
            # in RAM while the whole corpus is live AND makes datasets infer float64, doubling the
            # clips on disk. The worker copy in chatterbox_benchmark_inference.py has always written
            # float32 arrays, so the two writers of this "same" format disagreed on dtype until
            # 2026-09-09.
            arr = np.asarray(samples, dtype=np.float32).reshape(-1)
            # Chunks are capped by SAMPLE COUNT, not row count. Arrow addresses a list<float32>
            # chunk with 32-bit offsets, so a single chunk holding more than 2 GiB of audio dies
            # with "Value <n> too large to fit in C integer type" -- which is what a 1000-clip merge
            # dir of ~4 MB clips does (hit 2026-09-11 repacking
            # MergeMoshiOutputsViaSymlinks.0a7vxXPpRLed). A ChunkedArray over several sub-2-GiB
            # chunks is fine. Flushing BEFORE appending the overflowing clip puts the boundary in
            # exactly the place the old two-pass ``bounds`` scan did.
            if used and used + arr.size > _MAX_SAMPLES_PER_CHUNK:
                flush()
            used += arr.size
            rows[COL_INDEX].append(index)
            rows[COL_AUDIO].append(arr)
            rows[COL_SR].append(int(sr))
            rows[COL_MONOLOGUE].append(monologues.get(index))
            rows[COL_TRACE].append(traces.get(index))

        if not parts:
            # Everything fit in one chunk (the common case, and every empty write): straight out,
            # no staging round trip.
            Dataset.from_dict(rows).save_to_disk(out_path)
            return
        flush()

        from datasets import concatenate_datasets, load_from_disk

        # The staged parts are memory-mapped, so this streams instead of loading the corpus back in.
        loaded = [load_from_disk(p) for p in parts]
        concatenate_datasets(loaded).save_to_disk(out_path)
    finally:
        # Drop the mmaps before the files they point at go away.
        del loaded
        shutil.rmtree(tmpdir, ignore_errors=True)


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

    if not is_clip_dataset(src):
        clips = open_clips(src)
        for i in clips:
            audio, sr = clips[i]
            sf.write(out_dir / f"{i}.wav", audio, sr)
            mono = clips.sidecar(i, "monologue")
            if mono:
                (out_dir / f"{i}.txt").write_text(mono)
        return

    # Arrow input: iterate SEQUENTIALLY in numpy format, one row at a time.
    #
    # The obvious `for i in open_clips(src)` loop grows memory in proportion to the CORPUS: its
    # __getitem__ hands back each clip as a Python list of floats (a 700k-sample clip is a 700k-
    # element list, ~28 MB transiently), and 12,000 of those churn the allocator badly enough that
    # RSS climbs linearly -- measured 2026-09-11 at ~0.6 GB/min, which OOM-killed a 16 G eval of
    # B6's corpus 46 minutes in. That is the same shape as the TTS OOM that ClipSpool fixed, and
    # the same rule applies: bound the peak, do not raise the request. with_format("numpy") returns
    # the samples as arrays with no Python-list step, and sequential access lets arrow release
    # batches behind us, so peak is ONE clip regardless of corpus size.
    from datasets import load_from_disk

    ds = load_from_disk(str(src)).with_format("numpy")
    has_mono = COL_MONOLOGUE in ds.column_names
    for row in ds:
        i = int(row[COL_INDEX])
        sf.write(out_dir / f"{i}.wav", np.asarray(row[COL_AUDIO], dtype=np.float32), int(row[COL_SR]))
        if has_mono:
            mono = row[COL_MONOLOGUE]
            if mono is not None and str(mono):
                (out_dir / f"{i}.txt").write_text(str(mono))


def merge_clip_datasets(out_path, in_paths) -> None:
    """Concatenate sharded arrow clip stores into one.

    This is what replaces ``MergeMoshiOutputsViaSymlinks`` under ``storage="hf"``: the symlink merge
    spends one inode per clip *again*, doubling the cost of every sharded run, whereas concatenating
    arrow shards costs a handful of files total.
    """
    from datasets import concatenate_datasets, load_from_disk

    # Check for content BEFORE load_from_disk: a dataset dir written with zero rows has a
    # state.json listing no data files, and load_from_disk on that raises IndexError from deep
    # inside pyarrow ("list index out of range"), which says nothing about the real problem.
    parts = []
    for p in in_paths:
        state = Path(p) / "state.json"
        if state.is_file():
            try:
                if not json.loads(state.read_text()).get("_data_files"):
                    continue  # zero-row shard: the producing job emitted nothing
            except (json.JSONDecodeError, OSError):
                pass
        parts.append(load_from_disk(str(p)))
    parts = [p for p in parts if len(p) > 0]
    if not parts:
        raise ValueError(
            f"every shard is empty, so there is nothing to merge: {list(in_paths)}. The producing "
            "inference job(s) wrote no clips -- check that their input actually contained clips "
            "rather than treating this merge as the failure."
        )
    merged = concatenate_datasets(parts)
    seen = merged[COL_INDEX]
    if len(set(seen)) != len(seen):
        raise ValueError("shards overlap: the same clip index appears in more than one shard")
    merged.save_to_disk(str(out_path))


def repack_wav_dir(path, *, dry_run=True, follow_symlinks=False):
    """Rewrite a finished ``{i}.wav`` dir as one arrow dataset, in place. Returns a stats dict.

    Backlog E13(a). Sisyphus hashes a job's *inputs*, never its output content, so this is
    hash-neutral: the job keeps its id and its ``finished`` marker, and ``open_clips`` reads the
    result identically. It is the only lever that reclaims inodes from work already done -- flipping
    ``storage`` only helps future runs.

    **Verifies before it deletes.** Every clip is read back out of the new dataset and compared
    bit-exactly against the wav it came from, along with both sidecars; a single mismatch aborts
    with the originals untouched. The whole point is that this runs over finished experiment
    outputs, where a silent corruption would surface as an unreproducible number months later.

    Two refusals, both learned from the layout of the tree rather than guessed:

    * **A dir containing symlinks is refused unless ``follow_symlinks``.** ``MergeMoshiOutputsViaSymlinks``
      joins shards by symlinking their wavs, so a merged dir's entries point *into* shard dirs.
      Repacking the merge (with ``follow_symlinks=True``) is safe and is where the inodes are --
      119,426 of our 122,037 symlinks are in ``knowledge_benchmark``. Repacking a *shard* while a
      merge still points at it would leave the merge dangling, and ``os.symlink`` dangles silently.
      **So: merged dirs first, shards only after nothing references them.**
    * **A dir that is already an arrow dataset is a no-op**, so this is idempotent and safe to
      re-run over a list that partly succeeded.
    """
    import shutil

    path = Path(path)
    stats = {"path": str(path), "clips": 0, "freed": 0, "action": "none"}
    if is_clip_dataset(path):
        stats["action"] = "already-arrow"
        return stats
    if not path.is_dir():
        stats["action"] = "missing"
        return stats

    entries = list(os.scandir(path))
    links = [e for e in entries if e.is_symlink()]
    if links and not follow_symlinks:
        raise ValueError(
            f"{path} holds {len(links)} symlinks -- this is a shard-merge dir. Repacking it is "
            "correct and is where most of the inodes are, but pass follow_symlinks=True to say so "
            "deliberately. Never repack a SHARD that a merge still points into: the merge's links "
            "would dangle, and os.symlink dangles without complaining."
        )

    clips = _WavDirClips(path)
    if not len(clips):
        stats["action"] = "no-clips"
        return stats

    # Count without decoding: a dry run over ~1400 dirs would otherwise read every wav on Lustre
    # (93 merge dirs x 1000 clips) to answer a question that only needs the file listing.
    stats["clips"] = len(clips)
    stats["freed"] = sum(1 for e in entries if not e.is_dir())
    if dry_run:
        stats["action"] = "would-repack"
        return stats

    items, monologues, traces = [], {}, {}
    for i in clips:
        audio, sr = clips[i]
        items.append((i, audio, sr))
        mono = clips.sidecar(i, "monologue")
        if mono is not None:
            monologues[i] = mono
        trace = clips.sidecar(i, "trace")
        if trace is not None:
            traces[i] = trace

    assert len(items) == stats["clips"], "clip count changed between listing and read"

    # Stage the new dataset OFF the target volume. Writing it beside the original needs free
    # inodes on the very volume we are here to relieve, and at 99.6% full that deadlocks:
    # "[Errno 122] Disk quota exceeded: ....repack_tmp" before a single inode is freed
    # (hit 2026-09-11). $TMPDIR is node-local (1.5 TB NVMe, per-job, auto-cleaned), so staging there
    # costs the project nothing and the copy back happens only AFTER the originals are gone.
    stage_base = os.environ.get("TMPDIR")
    if stage_base and os.path.isdir(stage_base):
        tmp = Path(stage_base) / f"repack_{abs(hash(str(path)))}"
    else:
        tmp = path.parent / (path.name + ".repack_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    write_clips(tmp, items, monologues=monologues, traces=traces)

    # Read back from the NEW dataset and compare against what we just read off disk. Comparing the
    # in-memory arrays to themselves would prove nothing; this reopens the written files.
    #
    # Coverage is every clip and both sidecars -- but read in NUMPY format, not through _ArrowClips.
    # Its __getitem__ hands back a Python list of floats per clip and converts row by row, which on
    # 1000 clips of ~500k samples took >20 min for ONE dir (measured 2026-09-11) and would not have
    # fitted 91 dirs into a day's walltime. with_format("numpy") hands back the arrays directly.
    from datasets import load_from_disk

    ds = load_from_disk(str(tmp)).with_format("numpy")
    if len(ds) != len(items):
        shutil.rmtree(tmp)
        raise ValueError(f"{path}: wrote {len(ds)} clips, expected {len(items)}")
    row_of = {int(v): r for r, v in enumerate(ds[COL_INDEX])}
    for i, audio, sr in items:
        if i not in row_of:
            shutil.rmtree(tmp)
            raise ValueError(f"{path}: clip {i} missing from the written dataset")
        row = ds[row_of[i]]
        got = np.asarray(row[COL_AUDIO], dtype=np.float32)
        if int(row[COL_SR]) != sr or got.shape != audio.shape or not np.array_equal(got, audio):
            shutil.rmtree(tmp)
            raise ValueError(f"{path}: clip {i} does not round-trip -- refusing to delete originals")
        if (row[COL_MONOLOGUE] if row[COL_MONOLOGUE] is not None else None) != monologues.get(i):
            shutil.rmtree(tmp)
            raise ValueError(f"{path}: clip {i} monologue sidecar does not round-trip")
        if (row[COL_TRACE] if row[COL_TRACE] is not None else None) != traces.get(i):
            shutil.rmtree(tmp)
            raise ValueError(f"{path}: clip {i} trace sidecar does not round-trip")
    del ds

    # Only now is it safe to drop the originals. Files (and symlinks) only -- a subdirectory here is
    # something this function did not write and does not understand, so leave it alone.
    # Deleting first is what makes room for the copy back. Safe in the order that matters: for a
    # merge dir every entry is a SYMLINK and the audio itself lives in the shard, so nothing is
    # lost even if the copy were interrupted -- and the staged dataset has already been verified
    # clip-by-clip above. A shard dir holds the only copy, so --kind shard must only ever run with
    # real headroom (see repack_clips.py).
    for e in entries:
        if not e.is_dir(follow_symlinks=False):
            os.remove(e.path)
    for e in os.scandir(tmp):
        shutil.copy2(e.path, path / e.name)  # cross-device: stage is node-local, target is Lustre
    if not is_clip_dataset(path):
        raise ValueError(f"{path}: copy back did not produce a readable dataset -- staged copy is at {tmp}")
    shutil.rmtree(tmp)
    stats["action"] = "repacked"
    return stats
