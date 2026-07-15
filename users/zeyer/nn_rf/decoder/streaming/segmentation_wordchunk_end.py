"""
End-anchored (delayed) word-chunk RNA target -- companion to :mod:`segmentation`.

Split out so the ~30-line function lives in its own file (cleaner than editing the shared segmentation
module in place). See :func:`word_chunk_frame_targets_end`.
"""

from __future__ import annotations

import numpy as np

from .segmentation import ctc_collapse, num_chunks_for


def word_chunk_frame_targets_end(
    frames: np.ndarray,
    *,
    blank_idx: int,
    word_start_ids: frozenset,
    pad_to_multiple: int = 1,
) -> np.ndarray:
    """
    Per-frame WORD-CHUNKED target, END-anchored (delayed DSM word-chunk layout).

    Like ``segmentation.word_chunk_frame_targets`` (onset-anchored), but each word's sub-word tokens
    are packed CONSECUTIVELY so the LAST sub-word lands on its own emission frame (the word offset),
    with the earlier sub-words on the immediately preceding frames. So the whole word is emitted only
    once its final sub-word's acoustics are in (a delay to the word boundary), never anticipated --
    the opposite of the onset-anchored variant (which emits the later sub-words before their acoustics).

    Left-anchored at ``emit_frames[last] - cnt + 1``, but never before the previous word's run
    (``prev_end`` keeps the writes monotonic); if two words collide the later one cascades right (its
    run may then end a few frames past its own offset). Total non-blank frames == #tokens <= T, so it
    always fits.

    :param frames: [T] int per-frame CTC best path (with blank).
    :param blank_idx:
    :param word_start_ids: set of token ids that begin a word (SPM marker-prefixed).
    :param pad_to_multiple: pad output length up to a multiple of this (use chunk_size).
    :return: wc [T_out] int64.
    """
    frames = np.asarray(frames).reshape(-1)
    T = int(frames.shape[0])
    labels, emit_frames = ctc_collapse(frames, blank_idx)
    T_out = num_chunks_for(T, pad_to_multiple) * pad_to_multiple if pad_to_multiple > 1 else T
    wc = np.full((T_out,), blank_idx, dtype=np.int64)
    n = int(labels.size)
    prev_end = -1  # last output frame used by the previous word (keeps writes monotonic)
    i = 0
    while i < n:
        j = i + 1
        while j < n and int(labels[j]) not in word_start_ids:
            j += 1
        cnt = j - i  # sub-words in this word (tokens i .. j-1)
        f_last = int(emit_frames[j - 1])  # last sub-word onset == the word-offset anchor
        start = max(f_last - cnt + 1, prev_end + 1, 0)  # end at f_last, never overlap / go negative
        for k in range(cnt):
            pos = min(start + k, T_out - 1)
            wc[pos] = labels[i + k]
        prev_end = min(start + cnt - 1, T_out - 1)
        i = j
    return wc
