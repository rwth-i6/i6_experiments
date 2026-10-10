"""Convert frame-level CTC paths and MFA records into token or word onsets."""

from __future__ import annotations

from typing import List, Tuple

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
