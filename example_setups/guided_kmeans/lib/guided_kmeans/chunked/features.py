"""
Feature sources for the chunked clustering loop.

Replaces the RETURNN forward pass used by the single-process pipeline. In the
``precomputed=True`` setups that is all RETURNN was doing: iterating an HDF of
encoder outputs through a dummy ``nn.Module`` and a passthrough forward step.
"""

from __future__ import annotations

__all__ = ["HDFFeatureSource", "plan_chunks", "read_segment_file"]

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..util import PoolingRegistry


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
    """

    def __init__(
        self,
        files,
        segments: Optional[str] = None,
        subsampling: Optional[int] = None,
        pooling_function: str = "maxpool_time_np",
        chunk: int = 0,
        num_chunks: int = 1,
    ):
        if isinstance(files, (str, bytes)):
            files = [files]
        self.files = [str(f) for f in files]
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

        self._index = self._build_index()
        # Read in file/offset order rather than the striped planning order:
        # accumulation is order independent, and sequential reads keep the
        # ~100 MB/s network filesystem from being the bottleneck.
        self._index.sort(key=lambda entry: (entry[0], entry[1]))

    def _build_index(self) -> List[Tuple[int, int, int, str]]:
        """``[(file_idx, frame_offset, length, seq_tag)]`` for this chunk."""
        import h5py

        entries: List[Tuple[int, int, int, str]] = []
        for file_idx, filename in enumerate(self.files):
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
        try:
            for file_idx, offset, length, tag in self._index:
                if file_idx not in handles:
                    handles[file_idx] = h5py.File(self.files[file_idx], "r")
                features = handles[file_idx]["inputs"][offset : offset + length]
                if self.subsampling and self.subsampling > 1:
                    features = self.pool(features)
                yield tag, features
        finally:
            for handle in handles.values():
                handle.close()
