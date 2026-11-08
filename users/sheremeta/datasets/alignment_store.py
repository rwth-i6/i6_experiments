"""Read-only stores of per-sequence alignments for dataset workers, staged once per node on local disk."""

import os
from typing import Any, Dict, Optional, Sequence

import numpy as np

_STORE_CACHE: Dict[Any, "MmapAlignmentStore"] = {}

_STAGE_CHUNK = 1 << 24
_STAGE_ROWS = 1 << 20
_STAGED_SUFFIXES = (".bin.npy", ".hash.npy", ".off.npy", ".len.npy", ".exc.pkl")


def _path(x) -> str:
    return x.get_path() if hasattr(x, "get_path") else str(x)


def _tag_hash64(tag: str) -> int:
    import hashlib

    return int.from_bytes(hashlib.blake2b(tag.encode("utf-8"), digest_size=8).digest(), "little")


class MmapAlignmentStore:
    """Read-only mmap store of integer sequences keyed by sequence tag.

    The sequences are stored in one flat integer array and indexed by tag hash, offset and length. Processes share
    data through the page cache. Hash collisions use an exact-match fallback.

    TMPDIR should be node-local for performance. An absent tag with a matching stored hash may return the wrong
    sequence, an accepted trade-off to avoid storing all tag strings.
    """

    def __init__(self, base: str):
        """
        :param base: the prefix of the staged files
        """
        import pickle

        self._flat = np.load(f"{base}.bin.npy", mmap_mode="r")
        self._hashes = np.load(f"{base}.hash.npy", mmap_mode="r")
        self._offsets = np.load(f"{base}.off.npy", mmap_mode="r")
        self._lengths = np.load(f"{base}.len.npy", mmap_mode="r")
        with open(f"{base}.exc.pkl", "rb") as f:
            self._exceptions: Dict[str, Any] = pickle.load(f)

    def get(self, tag: str) -> Optional[np.ndarray]:
        """
        :param tag: the sequence tag
        :return: the sequence of the tag, None when the store has none
        """
        hit = self._exceptions.get(tag)
        if hit is None:
            h = np.uint64(_tag_hash64(tag))
            i = int(np.searchsorted(self._hashes, h))
            if i >= self._hashes.shape[0] or self._hashes[i] != h:
                return None
            hit = (int(self._offsets[i]), int(self._lengths[i]))
        off, n = hit
        return self._flat[off : off + n]


def _stage_store(paths: Sequence[str], base: str, heartbeat=None, dtype=np.uint16) -> None:
    """Stream the HDF shard(s) into the staged flat file plus index arrays, then rename atomically."""
    tmp = f"{base}.build{os.getpid()}"
    try:
        _build_staged_files(paths, tmp, heartbeat, dtype=dtype)
        for suffix in _STAGED_SUFFIXES:
            os.replace(f"{tmp}{suffix}", f"{base}{suffix}")
        with open(f"{base}.done", "wt") as f:
            f.write("ok")
    finally:
        for suffix in _STAGED_SUFFIXES:
            if os.path.exists(f"{tmp}{suffix}"):
                os.remove(f"{tmp}{suffix}")


def _build_staged_files(paths: Sequence[str], tmp: str, heartbeat=None, dtype=np.uint16) -> None:
    """
    Writes the flat file and its index arrays under a private prefix, the heartbeat called throughout.

    :param paths: the HDF shards, in RETURNN's HDF layout
    :param tmp: prefix of the files to write
    :param heartbeat: called after every slice of lengths and tags and after every copied chunk
    :param dtype: integer type of the flat file
    """
    import h5py
    import pickle

    tags = []
    lengths = []
    totals = []
    for p in paths:
        with h5py.File(p, "r") as f:
            totals.append(int(f["inputs"].shape[0]))
            for s in range(0, int(f["seqTags"].shape[0]), _STAGE_ROWS):
                for row in f["seqLengths"][s : s + _STAGE_ROWS]:
                    lengths.append(int(row[0]))
                for t in f["seqTags"][s : s + _STAGE_ROWS]:
                    tags.append(t.decode("utf-8") if isinstance(t, (bytes, bytearray)) else str(t))
                if heartbeat is not None:
                    heartbeat()
    total = sum(totals)
    assert len(set(tags)) == len(tags), "duplicate seq tags across alignment shards"
    lengths_arr = np.asarray(lengths, dtype=np.int64)
    assert int(lengths_arr.sum()) == total, (int(lengths_arr.sum()), total)
    offsets_arr = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(lengths_arr)[:-1]])

    flat = np.lib.format.open_memmap(f"{tmp}.bin.npy", mode="w+", dtype=dtype, shape=(total,))
    pos = 0
    for p in paths:
        with h5py.File(p, "r") as f:
            ds = f["inputs"]
            for s in range(0, int(ds.shape[0]), _STAGE_CHUNK):
                chunk = np.asarray(ds[s : s + _STAGE_CHUNK]).reshape(-1)
                if chunk.size == 0:
                    continue
                # every value must fit the integer type of the flat file
                assert chunk.min() >= 0 and chunk.max() <= np.iinfo(dtype).max, (chunk.min(), chunk.max())
                flat[pos : pos + chunk.shape[0]] = chunk
                pos += chunk.shape[0]
                if heartbeat is not None:
                    heartbeat()
    assert pos == total, (pos, total)
    flat.flush()
    del flat

    hashes_arr = np.fromiter((_tag_hash64(t) for t in tags), dtype=np.uint64, count=len(tags))
    order = np.argsort(hashes_arr, kind="stable")
    hashes_arr = hashes_arr[order]
    offsets_arr = offsets_arr[order]
    lengths_arr = lengths_arr[order]
    dup = np.zeros(hashes_arr.shape[0], dtype=bool)
    if hashes_arr.shape[0] > 1:
        same = hashes_arr[1:] == hashes_arr[:-1]
        dup[1:] |= same
        dup[:-1] |= same
    exceptions = {}
    for i in np.nonzero(dup)[0].tolist():
        exceptions[tags[int(order[i])]] = (int(offsets_arr[i]), int(lengths_arr[i]))
    keep = ~dup
    np.save(f"{tmp}.hash.npy", hashes_arr[keep])
    np.save(f"{tmp}.off.npy", offsets_arr[keep])
    np.save(f"{tmp}.len.npy", lengths_arr[keep].astype(np.int32))
    with open(f"{tmp}.exc.pkl", "wb") as f:
        pickle.dump(exceptions, f)


def _stage_base(paths: Sequence[str], dtype=np.uint16) -> str:
    """
    :param paths: the HDF shards
    :param dtype: integer type of the flat file
    :return: the prefix of the staged files, per node and per state of the shards
    """
    import hashlib
    import socket
    import tempfile

    h = hashlib.blake2b(digest_size=8)
    for p in paths:
        st = os.stat(p)
        h.update(f"{p}\0{st.st_size}\0{st.st_mtime_ns}\0".encode("utf-8"))
    tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
    # the names predate the move into this module, so the stores already staged on the nodes are reused
    stage_dir = os.path.join(tmp_root, f"dsm_stage_{socket.gethostname()}")
    os.makedirs(stage_dir, exist_ok=True)
    suffix = "" if np.dtype(dtype) == np.uint16 else f"_{np.dtype(dtype).name}"
    return os.path.join(stage_dir, f"dsm_ctc_align_{h.hexdigest()}{suffix}")


def _refresh_lock(lock: str) -> None:
    """
    Marks the lock as alive, a lock that a takeover already removed is left alone since its pass wrote the same bytes.

    :param lock: the lock directory
    """
    try:
        os.utime(lock)
    except FileNotFoundError:
        pass


def _ensure_staged(paths: Sequence[str], base: str, dtype=np.uint16) -> None:
    """Have exactly one process of the job stage the store, everyone else waits on the done marker."""
    import shutil
    import time

    done = f"{base}.done"
    lock = f"{base}.lock"
    while not os.path.exists(done):
        try:
            os.mkdir(lock)
        except FileExistsError:
            time.sleep(5.0)
            try:
                # the holder heartbeats the lock per staged chunk, so a stale mtime means it
                # stopped making progress and its pid was recycled to some live process, force
                # takeover (a duplicate staging pass writes identical bytes, so this is benign)
                if time.time() - os.stat(lock).st_mtime > 600.0:
                    shutil.rmtree(lock, ignore_errors=True)
                    continue
                with open(os.path.join(lock, "pid"), "rt") as f:
                    holder = int(f.read())
            except (OSError, ValueError):
                continue
            try:
                os.kill(holder, 0)
            except OSError:
                # holder died mid-stage, remove its lock and let the loop retry
                shutil.rmtree(lock, ignore_errors=True)
            continue
        try:
            if os.path.exists(done):
                return
            with open(os.path.join(lock, "pid"), "wt") as f:
                f.write(str(os.getpid()))
            _stage_store(paths, base, heartbeat=lambda: _refresh_lock(lock), dtype=dtype)
        finally:
            shutil.rmtree(lock, ignore_errors=True)
        return


def load_alignment_store(store_path, dtype=np.uint16) -> MmapAlignmentStore:
    """
    Opens the staged node-local store, staging it first once per node and store.

    :param store_path: one HDF in RETURNN's HDF layout or a list of its shards, as paths or sisyphus paths
    :param dtype: integer type of the staged flat file, every value must fit it
    :return: the store, one per process for the same shards and type
    """
    paths = tuple(sorted(_path(p) for p in (store_path if isinstance(store_path, (list, tuple)) else [store_path])))
    key = paths if np.dtype(dtype) == np.uint16 else (paths, np.dtype(dtype).name)
    store = _STORE_CACHE.get(key)
    if store is None:
        base = _stage_base(paths, dtype)
        _ensure_staged(paths, base, dtype)
        store = MmapAlignmentStore(base)
        _STORE_CACHE[key] = store
    return store


def _stage_file_locally(path: str) -> str:
    """Copy the sqlite store to the job-local TMPDIR once, shared by all dataset workers.

    Random per-seq lookups over NFS stall the pipeline, a local copy makes them disk-speed.
    flock guards the copy across concurrent workers, staging failures fall back to the original path.
    """
    import fcntl
    import hashlib
    import shutil
    import tempfile

    try:
        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        if not os.path.isdir(tmp_root):
            return path
        dst = os.path.join(tmp_root, f"mfa_store_{hashlib.md5(path.encode()).hexdigest()[:12]}.sqlite")
        src_size = os.path.getsize(path)
        if os.path.isfile(dst) and os.path.getsize(dst) == src_size:
            return dst
        with open(dst + ".lock", "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not (os.path.isfile(dst) and os.path.getsize(dst) == src_size):
                shutil.copyfile(path, dst + ".part")
                os.replace(dst + ".part", dst)
        return dst
    except OSError:
        return path


def open_sqlite_store(path: str):
    """
    :param path: a sqlite word-alignment store, e.g. from ConvertWordAlignmentStoreToSqliteJob
    :return: a read-only connection of this process to the store's copy in TMPDIR
    """
    import sqlite3

    path = _stage_file_locally(path)
    # immutable=1: the file never changes so sqlite skips locking, safe across many readers
    return sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, check_same_thread=False)
