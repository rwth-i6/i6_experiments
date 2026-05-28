"""
On-the-fly chunk segmentation of a CTC forced-align (per-frame best path) into
per-chunk label sub-sequences, for chunked streaming-decoder training.

The forced-align HDF (see ``exp2026_05_26_base_fzj._ls_train_forced_align``) stores,
per encoder frame, the CTC best-path label (or blank). We CTC-collapse it to the
emitted label sequence with each label's emission frame, then bucket each label
into the chunk ``emit_frame // chunk_size`` it falls in.

Assumption (v1): the streaming model's encoder reuses the same frontend
downsampling (factor 6 -> 60ms frames) as the base model that produced the
alignment, so chunk_size is expressed directly in alignment frames and no
resampling is needed. If a variant changes the encoder frame rate, convert
emit frames to seconds first (frame * frame_shift) and bucket by time.

Verified against the LS-train forced-align: the collapsed labels reproduce the
reference transcript exactly, and the per-chunk buckets concatenate back to the
full label sequence (non-decreasing chunk indices, exact reconstruction).
"""

from __future__ import annotations

from typing import Tuple
import numpy as np


def ctc_collapse(frames: np.ndarray, blank_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Collapse a per-frame CTC best path to the emitted label sequence.

    An emission happens at frame ``t`` iff the frame is non-blank and differs
    from the previous frame (standard CTC collapse: merge repeats, drop blanks).

    :param frames: [T] int per-frame labels (with blank).
    :param blank_idx: blank label index.
    :return: (labels [L] int64, emit_frames [L] int64), the emitted labels and
        the frame index at which each was emitted.
    """
    frames = np.asarray(frames).reshape(-1)
    if frames.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    change = np.empty(frames.shape, dtype=bool)
    change[0] = True
    change[1:] = frames[1:] != frames[:-1]
    mask = (frames != blank_idx) & change
    emit_frames = np.nonzero(mask)[0]
    return frames[emit_frames].astype(np.int64), emit_frames.astype(np.int64)


def num_chunks_for(num_frames: int, chunk_size: int) -> int:
    """Number of chunks covering ``num_frames`` frames (last chunk may be partial)."""
    return (int(num_frames) + chunk_size - 1) // chunk_size


def assign_chunks(emit_frames: np.ndarray, chunk_size: int) -> np.ndarray:
    """chunk index per emitted label = ``emit_frame // chunk_size`` ([L] int64, non-decreasing)."""
    return (np.asarray(emit_frames) // chunk_size).astype(np.int64)


def segment(frames: np.ndarray, *, blank_idx: int, chunk_size: int):
    """
    Full segmentation of a per-frame best path.

    :param frames: [T] int per-frame labels (with blank).
    :param blank_idx:
    :param chunk_size: chunk length in alignment frames.
    :return: dict with
        labels: [L] int64 emitted labels (CTC-collapsed; == reference transcript)
        emit_frames: [L] int64 emission frame per label
        chunk_idx: [L] int64 chunk index per label
        chunk_counts: [num_chunks] int64 number of labels per chunk (incl. empty chunks)
        num_chunks: int
        num_frames: int (= T)
    """
    frames = np.asarray(frames).reshape(-1)
    T = int(frames.shape[0])
    labels, emit_frames = ctc_collapse(frames, blank_idx)
    num_chunks = num_chunks_for(T, chunk_size)
    chunk_idx = assign_chunks(emit_frames, chunk_size)
    chunk_counts = np.bincount(chunk_idx, minlength=num_chunks).astype(np.int64) if labels.size else np.zeros(
        num_chunks, dtype=np.int64
    )
    return {
        "labels": labels,
        "emit_frames": emit_frames,
        "chunk_idx": chunk_idx,
        "chunk_counts": chunk_counts,
        "num_chunks": num_chunks,
        "num_frames": T,
    }


def chunk_augmented_targets(
    frames: np.ndarray,
    *,
    blank_idx: int,
    chunk_size: int,
    eoc_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the per-chunk label sequence with an end-of-chunk (EOC) marker after
    every chunk (including empty chunks):

        [chunk0 labels..., EOC, chunk1 labels..., EOC, ..., chunk_{K-1} labels..., EOC]

    Concatenating the per-chunk label groups (dropping EOC) yields the full label
    sequence; there is exactly one EOC per chunk, so the decoder learns to emit
    EOC to advance to the next chunk (and EOC alone for silent/empty chunks).

    :param frames: [T] int per-frame labels (with blank).
    :param blank_idx:
    :param chunk_size:
    :param eoc_idx: end-of-chunk marker id (distinct from all label ids and blank).
    :return: (targets [L + num_chunks] int64, pos_chunk_idx [L + num_chunks] int64),
        the augmented target sequence and the chunk index that each target position
        belongs to (an EOC of chunk c has chunk index c). ``pos_chunk_idx`` drives
        the chunk-restricted cross-attention mask in the decoder.
    """
    seg = segment(frames, blank_idx=blank_idx, chunk_size=chunk_size)
    labels = seg["labels"]
    counts = seg["chunk_counts"]
    num_chunks = seg["num_chunks"]

    out_targets = np.empty(int(labels.size) + num_chunks, dtype=np.int64)
    out_pos_chunk = np.empty_like(out_targets)
    w = 0
    li = 0
    for c in range(num_chunks):
        n = int(counts[c])
        out_targets[w : w + n] = labels[li : li + n]
        out_pos_chunk[w : w + n] = c
        w += n
        li += n
        out_targets[w] = eoc_idx
        out_pos_chunk[w] = c
        w += 1
    return out_targets, out_pos_chunk
