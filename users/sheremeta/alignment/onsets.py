"""Convert frame-level CTC paths and MFA records into token or word onsets."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

_WORD_MARK = "▁"  # word-start marker used by SentencePiece


def path_to_token_onsets(path: List[int], blank: int) -> List[Tuple[int, int]]:
    """Extract the first frame of each non-blank CTC segment.

    Returns ``(onset_frame, token_id)`` pairs in emission order. Repeated tokens
    separated by a blank begin distinct segments, as required by CTC collapse.
    """
    out: List[Tuple[int, int]] = []
    prev = blank
    for t, tok in enumerate(path):
        if tok != blank and tok != prev:
            out.append((t, int(tok)))
        prev = tok
    return out


def path_to_word_spans(
    path: Sequence[int], blank: int, pieces: Sequence[str], *, merge_repeats: bool = True
) -> List[Tuple[int, int]]:
    """
    Reads the frames of every word off a frame-level path.

    :param path: the label or the blank of every frame
    :param blank: the blank index
    :param pieces: the SentencePiece piece of every label
    :param merge_repeats: whether a label on adjacent frames is one token, as in a CTC path, else every frame is one
    :return: the first frame of every word and the frame after its last one
    """
    spans: List[Tuple[int, int]] = []
    prev = blank
    for frame, label in enumerate(path):
        if label != blank:
            # a new token opens a word when its piece starts one, the first token of the path always does
            new_token = label != prev or not merge_repeats
            if new_token and (pieces[label].startswith(_WORD_MARK) or not spans):
                spans.append((frame, frame + 1))
            else:
                spans[-1] = (spans[-1][0], frame + 1)
        prev = label
    return spans


def _is_pseudo_word(word: str) -> bool:
    # angle-bracketed MFA labels represent non-lexical events
    return word.startswith("<") and word.endswith(">")


def mfa_words_to_onsets(words, *, time_scale: float = 1.0) -> Tuple[List[str], List[float]]:
    """Convert MFA word intervals into labels and onset times.

    ``words`` contains ``(word, start, end)`` records. Start and end values use
    the units expected by ``build_word_chunked``; ``time_scale`` converts those
    values to seconds. Non-lexical angle-bracketed labels are omitted.
    """
    ws: List[str] = []
    onsets: List[float] = []
    for w, start, _end in words:
        if _is_pseudo_word(str(w)):
            continue
        ws.append(str(w))
        onsets.append(float(start) * time_scale)
    return ws, onsets


def token_onsets_to_words(onsets: List[Tuple[int, int]], spm) -> Tuple[List[str], List[int]]:
    """Group SentencePiece token onsets into words.

    A piece carrying the word-start marker begins a new word. Each returned word
    inherits the onset frame of its first piece.
    """
    words: List[str] = []
    frames: List[int] = []
    cur = ""
    for i, (frame, tok) in enumerate(onsets):
        piece = spm.id_to_piece(int(tok))
        if piece.startswith(_WORD_MARK) or i == 0:
            if cur:
                words.append(cur.replace(_WORD_MARK, "").strip())
            cur = piece
            frames.append(frame)
        else:
            cur += piece
    if cur:
        words.append(cur.replace(_WORD_MARK, "").strip())
    keep = [(w, f) for w, f in zip(words, frames) if w]
    if not keep:
        return [], []
    ws, fs = zip(*keep)
    return list(ws), list(fs)


def delayed_path_to_ctc_path(
    path: Sequence[int], *, pad_id: int, blank: int, delay_frames: int
) -> Tuple[np.ndarray, int]:
    """
    Moves a path on the delayed frames of the audio padded by the delay to the frames of the audio itself, every frame
    without a token blank. A token before the delay or past the audio goes to the nearest free frame, in order.

    :param path: the token or the pad of every frame, ``delay_frames + 1`` frames longer than the audio
    :param pad_id: the code of a frame without a token in the path
    :param blank: the code of a frame without a token in the result
    :param delay_frames: frames the path lags the audio
    :return: the path on the frames of the audio and how many tokens left their own frame
    """
    labels = np.asarray(path).reshape(-1)
    num_frames = labels.shape[0] - delay_frames - 1
    frames = np.flatnonzero(labels != pad_id)
    assert frames.size <= num_frames, (frames.size, num_frames)

    # every token on its own frame of the audio, the ones outside pushed in, ties pushed apart
    wanted = frames - delay_frames
    placed = np.clip(wanted, 0, num_frames - 1)
    for k in range(1, placed.size):
        placed[k] = max(placed[k], placed[k - 1] + 1)
    for k in range(placed.size - 1, -1, -1):
        placed[k] = min(placed[k], num_frames - 1 if k == placed.size - 1 else placed[k + 1] - 1)

    out = np.full(num_frames, blank, dtype=np.int32)
    out[placed] = labels[frames]
    return out, int(np.sum(placed != wanted))
