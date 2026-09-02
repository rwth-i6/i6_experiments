"""
Feature sources for the chunked clustering loop.

Replaces the RETURNN forward pass used by the single-process pipeline. In the
``precomputed=True`` setups that is all RETURNN was doing: iterating an HDF of
encoder outputs through a dummy ``nn.Module`` and a passthrough forward step.
"""

from __future__ import annotations

__all__ = ["HDFFeatureSource", "cached_file", "plan_chunks", "read_segment_file"]

import os
import subprocess
import time
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..util import PoolingRegistry

#: A transient fault in a node's NFS client clears in seconds; a node whose
#: open state is really gone never does. Three attempts with a short backoff
#: tells the two apart without parking a 9-CPU task on a node that is not
#: coming back.
_READ_ATTEMPTS = 3
_READ_RETRY_DELAY = 5.0

#: Deliberately generous: a cold multi-GB fetch runs for minutes, and a caller
#: that arrives while another task on the node is already transferring the
#: file waits for that transfer on top.
_CACHE_TIMEOUT = 3600.0


def plan_chunks(lengths: Sequence[int], num_chunks: int) -> List[List[int]]:
    """
    Assign sequence indices to chunks so every chunk gets a similar number of
    *frames* (which is what recognition time scales with), not a similar
    number of sequences.

    Sorting by length and striping round-robin brings the worst-case chunk
    from ~8% above the mean (contiguous split, measured on LibriSpeech-100)
    down to ~0.3%. Since every chunk of an epoch has to finish before the
    reduce step, that imbalance is pure added wall-clock.
    """
    if num_chunks < 1:
        raise ValueError(f"num_chunks must be >= 1, got {num_chunks}")
    order = np.argsort(-np.asarray(lengths, dtype=np.int64), kind="stable")
    chunks: List[List[int]] = [[] for _ in range(num_chunks)]
    for rank, seq_idx in enumerate(order):
        chunks[rank % num_chunks].append(int(seq_idx))
    return chunks


def read_segment_file(path: str) -> List[str]:
    """One segment name per line; blank lines and '#' comments ignored."""
    from i6_core.util import uopen

    segments = []
    with uopen(path, "rt") as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith("#"):
                segments.append(line)
    return segments


def cached_file(path: str) -> str:
    """
    Ask the i6 cache manager (``cf``) for a node-local copy of ``path``.

    Reading features straight off a group volume leaves a chunk task exposed
    for its whole pass: ~1400 small strided reads spread over ~25 minutes,
    through one handle held open throughout. A one-second fault in the node's
    NFS client kills all of it - on 2026-08-25 that took out 32 chunk tasks
    across five nodes, always in per-node bursts that hit unrelated jobs in
    the same second while the same job's chunks on other nodes finished
    normally. One bulk copy up front shrinks that window to the copy itself
    and reads the rest off local disk. The cache manager keeps one copy per
    node and makes concurrent callers wait on whoever is already fetching it,
    so the chunks of an array job share a single transfer instead of pulling
    the file once each.

    Never a correctness dependency: ``cf`` prints the original file (resolved
    through any symlink, which reads the same) when it cannot produce a local
    copy - cache manager down, disk full - and *exits non-zero while doing
    so*, which is why the return code is ignored and only stdout is trusted.
    Any other failure to run it at all is treated the same way, because
    reading over the network is slower, not wrong.
    """
    try:
        completed = subprocess.run(
            ["cf", path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_CACHE_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return path
    # cf writes the path to stdout and everything else (LOG:/WARN:/ERROR:) to
    # stderr, so stdout is either a usable path or nothing worth having.
    local = completed.stdout.decode("utf-8", "replace").strip()
    if not local or not os.path.isfile(local):
        return path
    if local != path:
        print(f"[CACHE] {path} -> {local}", flush=True)
    return local


class HDFFeatureSource:
    """
    Reads one chunk's sequences out of one or more RETURNN HDF files.

    :param files: HDF file(s) holding ``inputs``/``seqLengths``/``seqTags``
    :param segments: optional segment list file restricting the corpus, the
        equivalent of ``seq_list_filter_file`` on the RETURNN dataset
    :param subsampling: pooling stride; ``None`` or 1 disables pooling, which
        matches the ``if self.subsampling and self.subsampling > 1`` guard in
        the single-process callback
    :param pooling_function: name in :class:`PoolingRegistry`
    :param chunk: which chunk this instance serves
    :param num_chunks: total number of chunks the corpus is split into
    :param cache: fetch a node-local copy of every file through the i6 cache
        manager first (see :func:`cached_file`); ``False`` reads the files
        where they are
    """

    def __init__(
        self,
        files,
        segments: Optional[str] = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        chunk: int = 0,
        num_chunks: int = 1,
        cache: bool = True,
    ):
        if isinstance(files, (str, bytes)):
            files = [files]
        self.files = [str(f) for f in files]
        self.cache = cache
        self.segments = segments
        self.subsampling = subsampling
        self.pooling_function = pooling_function
        self.chunk = chunk
        self.num_chunks = num_chunks

        self.pool = PoolingRegistry.select(
            pooling_function,
            stride=subsampling,
            kernel_size=2 * subsampling if subsampling else None,
        )

        # Resolved once, here rather than per read: every construction site
        # builds this inside a job task (from a Spec, or directly in
        # GlobalCovarianceJob.compute), so this is already the node that wants
        # the local copy - and _build_index reads the same files.
        self._local_files = (
            [cached_file(f) for f in self.files] if cache else list(self.files)
        )

        self._index = self._build_index()
        # Read in file/offset order rather than the striped planning order:
        # accumulation is order independent, and sequential reads keep the
        # ~100 MB/s network filesystem from being the bottleneck.
        self._index.sort(key=lambda entry: (entry[0], entry[1]))

    def _build_index(self) -> List[Tuple[int, int, int, str]]:
        """``[(file_idx, frame_offset, length, seq_tag)]`` for this chunk."""
        import h5py

        entries: List[Tuple[int, int, int, str]] = []
        for file_idx, filename in enumerate(self._local_files):
            with h5py.File(filename, "r") as hdf:
                lengths = hdf["seqLengths"][:]
                if lengths.ndim == 2:
                    lengths = lengths[:, 0]
                tags = hdf["seqTags"][:]
                offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
                for seq_idx, tag in enumerate(tags):
                    if isinstance(tag, bytes):
                        tag = tag.decode("utf-8")
                    entries.append(
                        (file_idx, int(offsets[seq_idx]), int(lengths[seq_idx]), str(tag))
                    )

        if self.segments is not None:
            wanted = set(read_segment_file(self.segments))
            missing = wanted - {entry[3] for entry in entries}
            if missing:
                raise ValueError(
                    f"{len(missing)} segment(s) from {self.segments} are not in the HDF "
                    f"file(s), e.g. {sorted(missing)[:3]}"
                )
            entries = [entry for entry in entries if entry[3] in wanted]

        # Plan on a deterministic order so every chunk task, running in its
        # own process, derives the identical partition of the corpus.
        entries.sort(key=lambda entry: entry[3])
        assignment = plan_chunks([entry[2] for entry in entries], self.num_chunks)
        return [entries[i] for i in assignment[self.chunk]]

    @property
    def total_frames(self) -> int:
        return sum(entry[2] for entry in self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        import h5py

        handles = {}

        def read(file_idx: int, offset: int, length: int) -> np.ndarray:
            """
            One sequence's frames, reopening the file if the handle dies.

            An EIO in the middle of a pass is almost never a bad file: it is
            the node's NFS client losing the open state, which takes down
            every reader on that node in the same second and leaves the handle
            unusable afterwards - so the retry reopens rather than reading
            again through it. A node that is genuinely broken exhausts the
            attempts and the task fails as it did before, to be rescheduled
            somewhere healthy.
            """
            last_error: Optional[BaseException] = None
            for attempt in range(_READ_ATTEMPTS):
                try:
                    if file_idx not in handles:
                        handles[file_idx] = h5py.File(self._local_files[file_idx], "r")
                    return handles[file_idx]["inputs"][offset : offset + length]
                except (OSError, RuntimeError, KeyError, ValueError) as exc:
                    handle = handles.pop(file_idx, None)
                    # An OSError is the I/O failure this retry exists for, so
                    # it always gets another go. h5py reports a handle the
                    # library has given up on in whichever of the other three
                    # types the failing call happens to use - so those count
                    # only once the handle really is gone (or never opened).
                    # Raised against a file that is still open they are a real
                    # bug - a missing dataset, a bad slice - and reopening
                    # would only bury it under a delay.
                    retryable = isinstance(exc, OSError) or handle is None or not handle
                    if handle is not None:
                        try:
                            handle.close()
                        except Exception:
                            pass  # already dead; dropping the reference is the point
                    if not retryable:
                        raise
                    last_error = exc
                    if attempt + 1 < _READ_ATTEMPTS:
                        print(
                            f"[RETRY] {self._local_files[file_idx]}: {length} frames "
                            f"at {offset} failed ({exc!r}), reopening",
                            flush=True,
                        )
                        time.sleep(_READ_RETRY_DELAY * (attempt + 1))
            raise OSError(
                f"reading {length} frames at {offset} from "
                f"{self._local_files[file_idx]} failed {_READ_ATTEMPTS} times"
            ) from last_error

        try:
            for file_idx, offset, length, tag in self._index:
                features = read(file_idx, offset, length)
                if self.subsampling and self.subsampling > 1:
                    features = self.pool(features)
                yield tag, features
        finally:
            for handle in handles.values():
                handle.close()
