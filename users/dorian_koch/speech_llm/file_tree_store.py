"""A directory tree stored as ONE arrow dataset: the MFDB (Moshified FDB) output store.

The MFDB inference output is a tree ``<item>/{output.wav, output.json, transcription.json, pause.json, ...}``: the
benchmark's own NeMo ASR (``asr.py``) and evaluators (``evaluate.py``) walk it by file name, and we run both
unmodified. As files it costs ~6 inodes per item, ~1,300 per task -- 306K inodes over 433 finished jobs on
2026-10-14, the largest consumer on the inode-bound ``/hpcwork/p0023999`` (clusters/i6-rz.md). Stored here it costs
~3, and every reader gets a real directory back for the duration of its run (``tree_root``).

Layout: one row per regular file, ``relpath`` (POSIX, relative to the tree root) and ``data`` (the file's bytes,
verbatim). Bytes, not decoded audio: unpacking reproduces every file byte for byte, so the benchmark code sees exactly
what it would have seen on the original tree. Directories are implied by the paths (an empty directory is not
representable; ``pack_tree`` refuses one rather than dropping it silently).

Both layouts are supported forever, told apart by looking at the path (``is_tree_dataset``), like ``clip_store``:

    with tree_root(path) as root:     # an arrow tree OR a plain dir -- the caller does not care
        run_evaluator(root)

``pack_tree`` validates every ``.json`` it packs: a quota-killed write left ``output.json`` files padded with NUL
bytes on 2026-10-13, and the MFDB eval only failed much later, in another job. Packing is where that is caught.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from pathlib import Path

COL_RELPATH = "relpath"
COL_DATA = "data"

#: Bytes per staged part. Arrow's binary offsets are int32, so one chunk may not exceed 2 GiB; parts are flushed far
#: below that, which also bounds the memory a pack holds (one part's file bytes).
_MAX_BYTES_PER_PART = 256 * 1024 * 1024


def is_tree_dataset(path: str | os.PathLike) -> bool:
    """True if ``path`` is a tree packed by ``pack_tree`` (not a plain dir, and not a ``clip_store`` dataset)."""
    p = Path(path)
    info = p / "dataset_info.json"
    if not ((p / "state.json").is_file() and info.is_file()):
        return False
    try:
        features = json.loads(info.read_text()).get("features", {})
    except (OSError, ValueError):
        return False
    return COL_RELPATH in features and COL_DATA in features


def _walk(src: Path) -> list[str]:
    """Sorted POSIX relpaths of every regular file under ``src``; raises on symlinks and empty dirs."""
    out = []
    for dirpath, dirnames, filenames in os.walk(src):
        dirnames.sort()
        d = Path(dirpath)
        if d != src and not dirnames and not filenames:
            raise ValueError(f"empty directory {d} cannot be represented in a packed tree")
        for name in sorted(filenames):
            f = d / name
            if f.is_symlink() or not f.is_file():
                raise ValueError(f"{f} is not a regular file; pack_tree stores regular files only")
            out.append(f.relative_to(src).as_posix())
    return out


def pack_tree(src_dir, out_path, *, validate_json: bool = True) -> int:
    """Pack every regular file under ``src_dir`` into an arrow dataset at ``out_path``; returns the file count.

    ``validate_json``: ``json.loads`` every ``*.json`` first and raise naming the file (NUL holes, truncation).
    The dataset is re-read and compared byte for byte against the source before this returns.
    """
    from datasets import Dataset, Features, Value, concatenate_datasets, load_from_disk

    src = Path(src_dir)
    rels = _walk(src)
    if not rels:
        raise ValueError(f"no files under {src} -- refusing to write an empty tree")
    if validate_json:
        bad = []
        for r in rels:
            if r.endswith(".json"):
                try:
                    json.loads((src / r).read_bytes())
                except ValueError as exc:
                    bad.append(f"{r}: {exc}")
        if bad:
            raise ValueError(f"{len(bad)} invalid json file(s) under {src}: {bad[:5]}")

    features = Features({COL_RELPATH: Value("string"), COL_DATA: Value("binary")})
    out_path = str(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    # Staging honours $TMPDIR (node-local on SLURM): the parts never land on the volume being written to.
    tmpdir = tempfile.mkdtemp(prefix="pack_tree.")
    loaded = []
    try:
        parts, rows, used = [], {COL_RELPATH: [], COL_DATA: []}, 0

        def flush():
            nonlocal rows, used
            if rows[COL_RELPATH]:
                part = os.path.join(tmpdir, f"part{len(parts):05d}")
                Dataset.from_dict(rows, features=features).save_to_disk(part)
                parts.append(part)
            rows, used = {COL_RELPATH: [], COL_DATA: []}, 0

        for r in rels:
            data = (src / r).read_bytes()
            if used and used + len(data) > _MAX_BYTES_PER_PART:
                flush()
            rows[COL_RELPATH].append(r)
            rows[COL_DATA].append(data)
            used += len(data)
        if not parts:
            Dataset.from_dict(rows, features=features).save_to_disk(out_path)
        else:
            flush()
            loaded = [load_from_disk(p) for p in parts]
            concatenate_datasets(loaded).save_to_disk(out_path)
    finally:
        del loaded
        shutil.rmtree(tmpdir, ignore_errors=True)

    bad = verify_tree(out_path, src)
    if bad:
        raise RuntimeError(f"packed tree {out_path} differs from {src}: {bad}")
    return len(rels)


def iter_tree(path):
    """Yield ``(relpath, bytes)`` for every file of a packed tree, sequentially (one row live at a time)."""
    from datasets import load_from_disk

    ds = load_from_disk(str(path))
    for row in ds:
        yield row[COL_RELPATH], row[COL_DATA]


def verify_tree(path, src_dir) -> str | None:
    """None if the packed tree at ``path`` holds exactly the regular files of ``src_dir``, byte for byte."""
    src = Path(src_dir)
    want = set(_walk(src))
    seen = set()
    for rel, data in iter_tree(path):
        if rel in seen:
            return f"{rel} packed twice"
        if rel not in want:
            return f"{rel} not in the source"
        seen.add(rel)
        if (src / rel).read_bytes() != data:
            return f"{rel} differs"
    return None if seen == want else f"{len(want - seen)} source file(s) missing, e.g. {sorted(want - seen)[:3]}"


def materialise_tree(path, out_dir) -> Path:
    """Write a packed tree back out as files under ``out_dir`` (created); returns ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    for rel, data in iter_tree(path):
        f = (out / rel).resolve()
        if root not in f.parents:
            raise ValueError(f"packed path {rel!r} escapes the tree root")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)
    return out


@contextlib.contextmanager
def tree_root(path, *, name: str = "speech_output", writable: bool = False):
    """A real directory with the tree at ``path``: ``path`` itself if it is a plain dir, else a temporary unpack.

    The unpack goes to ``$TMPDIR/<tmp>/<name>`` (node-local on SLURM, local disk on the login node) and is deleted on
    exit, so anything a reader writes into it (``eval_user_interruption`` writes ``rating.json`` per item) never
    touches the stored output. ``name`` keeps the root's last component equal to the original
    (``.../speech_output``), so per-item paths in evaluator logs parse the same (``evalstats._mfdb_item_of``).

    A plain dir is yielded in place unless ``writable``: then it is copied too, because a caller that writes into it
    (older, pre-2026-10-14 SpeechInference outputs are plain) would otherwise modify the stored output.
    """
    if not is_tree_dataset(path) and not writable:
        yield str(path)
        return
    tmp = tempfile.mkdtemp(prefix="tree_root.")
    try:
        if is_tree_dataset(path):
            yield str(materialise_tree(path, os.path.join(tmp, name)))
        else:
            yield shutil.copytree(str(path), os.path.join(tmp, name))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    # ``python file_tree_store.py unpack <packed> <out_dir>``: for readers in a venv without ``datasets`` (the analysis
    # scripts under the fdb_eval venv go through ``analysis/evalstats.fdb_tree_dir``, which calls this).
    import sys

    assert len(sys.argv) == 4 and sys.argv[1] == "unpack", "usage: file_tree_store.py unpack <packed> <out_dir>"
    assert is_tree_dataset(sys.argv[2]), f"{sys.argv[2]} is not a packed tree"
    print(materialise_tree(sys.argv[2], sys.argv[3]))
