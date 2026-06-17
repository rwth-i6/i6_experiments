"""
Shared subword-nmt BPE de-tokenization for the (real-RETURNN) BPE pHMM / CTC decoders.

A single definition used by both BPE decoders so they cannot drift apart:
  * ``greedy_bpe_v1`` (frame-argmax greedy), and
  * ``rasr_phmm_lexfree_ngram_bpe_v1`` (lexicon-free count-LM search).

Specials (blank / silence / sentence boundaries) are dropped **token-wise BEFORE** de-BPE -- never by
substring replacement on the joined string. A special emitted between two subwords of one word would
otherwise leave a stray space that the ``"@@ "`` continuation-marker removal cannot heal, splitting one
word into two (e.g. ``["RE@@", "[BLANK]", "D"]`` -> ``"RE D"`` instead of ``"RED"``). Real subword-nmt
tokens are uppercase + ``"@@"`` and are never members of the strip set, so token-wise filtering cannot
drop a genuine subword.
"""
from typing import Iterable, Sequence


def debpe_tokens(tokens: Iterable[str], strip_tokens: Sequence[str] = ()) -> str:
    """
    Drop ``strip_tokens`` (token-wise) and de-BPE the remaining subword stream into a WORD string.

    :param tokens: emitted label strings (BPE subwords, possibly interleaved with special tokens)
    :param strip_tokens: special tokens to remove as whole tokens before de-BPE (blank/silence/bos/eos)
    :return: a whitespace-normalized word string ("@@" continuation markers removed)
    """
    strip = set(strip_tokens)
    kept = [t for t in tokens if t not in strip]
    # subword-nmt de-BPE: "@@ " joins a continued subword to the next; the trailing bare "@@" guard
    # handles a final incomplete subword (rare).
    out = " ".join(kept).replace("@@ ", "").replace("@@", "")
    return " ".join(out.split())
