"""Recover mixed Granite whole-word/piece output without scoring normalization."""

import unicodedata


def granite_fallback_result(trace, tokenizer):
    """Decode whole piece runs, including UTF-8 byte fallback and separators.

    The && marker identifies the piece route, not an observed unknown LM event:
    the decoder can resolve an exact known pronunciation while on that route.
    Preserve raw traceback separately and report piece counts without claiming
    they are true OOV counts.
    """
    surface, pieces = "", []
    piece_count = max_word_pieces = word_pieces = 0

    def flush():
        nonlocal surface
        if pieces:
            decoded = tokenizer.convert_tokens_to_string(pieces)
            # Sequence detokenization strips the initial word-start marker.
            if pieces[0].startswith("▁"):
                decoded = " " + decoded
            surface += decoded
            pieces.clear()

    for item in trace:
        name = item.lemma
        if not name or name in ("[BLANK]", "[SILENCE]", "[SENTENCE-BEGIN]", "[SENTENCE-END]"):
            continue
        if name.endswith("&&"):
            token = name[:-2]
            pieces.append(token)
            if token.startswith("▁"):
                word_pieces = 0
            if token != "▁":
                word_pieces += 1
                piece_count += 1
                max_word_pieces = max(max_word_pieces, word_pieces)
        else:
            flush()
            word_pieces = 0
            punctuation = all(unicodedata.category(c).startswith("P") for c in name)
            surface += name if punctuation else " " + name
    flush()
    return dict(
        text=surface.strip(),
        fallback_piece_count=piece_count,
        max_fallback_word_pieces=max_word_pieces,
    )
