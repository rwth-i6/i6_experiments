"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/prior_gap.py (the helpers only).

The window replay and word-LM helpers :class:`~.word_lm.LexiconTrieBuildJob` calls: the phonemization
lexicon, the counted/held split of the prior's window, the replay of the window's word lines, the
KenLM word LM and the Witten-Bell phone score of one string.

Cut from the source: ``PriorGapAnalysisJob`` and everything only it reaches (the Step 0 prior-gap
read, the neural phone LM rows, the segmentation scorer, the verdict and report helpers) -- analysis
of a finished phase, not used by any in-scope run.
"""

from __future__ import annotations

import gzip
import math
import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "LN10",
    "SIL_SPELLINGS",
    "DISCOUNT_FALLBACK",
    "witten_bell_utt_log_prob",
    "KenLMWordLM",
    "window_line_flags",
    "load_phonemization_lexicon",
    "restrict_to_word_lm",
    "count_kept_lines",
    "replay_window_texts",
]

LN10 = math.log(10.0)
#: the corpus spells silence ``<SIL>`` and every read in this campaign spells it ``SIL``; both
#: spellings are the SIL token
SIL_SPELLINGS = ("<SIL>", "SIL")
#: ``lmplz`` discount fallback, the campaign's canonical value
DISCOUNT_FALLBACK = (0.5, 1.0, 1.5)


def witten_bell_utt_log_prob(prior, tokens: Sequence[str], order: int) -> float:
    """``phone_prior.PhoneNgramPrior.log_prob`` of one string, in nats (the live estimator's convention)."""
    from .phone_prior import PHONE2ID

    if not tokens:
        return 0.0
    return float(prior.log_prob([PHONE2ID[t] for t in tokens], order=order))


class KenLMWordLM:
    """The word LM of the lexicalised prior: ``<s>`` context, ``BaseScore`` per word, no ``</s>``."""

    def __init__(self, path: str):
        import kenlm

        self._kenlm = kenlm
        self.model = kenlm.Model(path)
        self.order = int(self.model.order)

    def begin_state(self):
        state = self._kenlm.State()
        self.model.BeginSentenceWrite(state)
        return state

    def score(self, state, word: str) -> Tuple[float, object]:
        """``(log p(word | state) in nats, the successor state)``."""
        out = self._kenlm.State()
        return float(self.model.BaseScore(state, word, out) * LN10), out

    def __contains__(self, word: str) -> bool:
        return word in self.model


def window_line_flags(
    window_path: str, *, n_lines: int, held_stride: int
) -> Tuple[List[bool], List[bool]]:
    """``phone_prior.count_ngrams``' own split of the window: ``(counted, held)``, one flag per line.

    An empty line is neither counted nor held (``count_ngrams`` skips it before the stride test),
    so it is ``False`` in both and contributes to no model.
    """
    counted: List[bool] = []
    held: List[bool] = []
    opener = gzip.open if str(window_path).endswith(".gz") else open
    with opener(window_path, "rt") as fh:
        for i, line in enumerate(fh):
            if i >= n_lines:
                break
            toks = line.split()
            is_held = bool(toks) and bool(held_stride) and i % held_stride == 0
            counted.append(bool(toks) and not is_held)
            held.append(is_held)
    assert len(counted) == n_lines, f"{window_path} has {len(counted)} lines, expected {n_lines}"
    return counted, held


def load_phonemization_lexicon(bliss_lexicon: str, g2p_lexicon: Optional[str]) -> Dict[str, List[str]]:
    """``PhonemizeWithSilJob``'s own word -> pronunciation map (bliss first, g2p for the rest)."""
    from .lexicon import _load_g2p_lexicon, _load_lexicon_word_to_phon

    w2p = _load_lexicon_word_to_phon(bliss_lexicon)
    if g2p_lexicon is not None:
        for word, pron in _load_g2p_lexicon(g2p_lexicon).items():
            w2p.setdefault(word, pron)
    return {w: p.split() for w, p in w2p.items()}


def restrict_to_word_lm(word_to_phones: Mapping[str, Sequence[str]], word_lm) -> Dict[str, List[str]]:
    """The trie's word set: only the lexicon words the word LM has in its VOCABULARY.

    Without it, ``BaseScore`` prices an unseen word as one ``<unk>`` transition whatever its phone
    length, so a long unseen pronunciation is a cheap cover for any phone span.  The escape word
    replaces that path at a per-phone price.
    """
    return {w: list(p) for w, p in word_to_phones.items() if w in word_lm}


def count_kept_lines(text_path: str, vocab) -> int:
    """The number of lines ``PhonemizeWithSilJob`` keeps: every word of the line is in the lexicon."""
    opener = gzip.open if str(text_path).endswith(".gz") else open
    kept = 0
    with opener(text_path, "rt") as fh:
        for line in fh:
            words = line.split()
            if words and all(w in vocab for w in words):
                kept += 1
    return kept


def replay_window_texts(
    *,
    word_corpus: str,
    window_phn: str,
    word_to_phones: Mapping[str, Sequence[str]],
    counted: Sequence[bool],
    out_dir: str,
    n_window_lines: int,
    sample_seed: int,
) -> Dict[str, object]:
    """Write the counted window lines as phones and as words, VERIFYING they are the same lines.

    ``PhonemizeWithSilJob`` writes one phone line per kept text line, and ``SampleLinesJob`` then
    takes ``sample_line_indices(n_kept, n_window_lines, sample_seed)`` of those, so window line
    ``j`` is the ``keep[j]``-th kept line of ``word_corpus``.  Every one of the ``n_window_lines``
    recovered word lines is checked against the window's own phone line (its SIL tokens removed
    must equal the concatenation of the lexicon pronunciations); only the lines flagged in
    ``counted`` are written.  ``<SIL>`` is written as ``SIL``.
    """
    from .phone_text import sample_line_indices

    vocab = set(word_to_phones)
    n_kept = count_kept_lines(word_corpus, vocab)
    keep = iter(sample_line_indices(n_kept, n_window_lines, sample_seed))
    want = next(keep, None)

    phones_path = os.path.join(out_dir, "window.phn.txt")
    words_path = os.path.join(out_dir, "window.words.txt")
    opener = gzip.open if str(word_corpus).endswith(".gz") else open
    win_opener = gzip.open if str(window_phn).endswith(".gz") else open
    n_written = n_checked = 0
    with opener(word_corpus, "rt") as src, win_opener(window_phn, "rt") as win, \
            open(phones_path, "w") as pf, open(words_path, "w") as wf:
        kept_index = 0
        j = 0
        for line in src:
            if want is None:
                break
            words = line.split()
            if not (words and all(w in vocab for w in words)):
                continue
            if kept_index == want:
                phone_line = win.readline().split()
                recon = [p for w in words for p in word_to_phones[w]]
                assert [p for p in phone_line if p not in SIL_SPELLINGS] == recon, (
                    f"window line {j} does not phonemize to its word line: "
                    f"{phone_line[:12]} vs {recon[:12]}"
                )
                n_checked += 1
                if counted[j]:
                    pf.write(" ".join("SIL" if p in SIL_SPELLINGS else p for p in phone_line) + "\n")
                    wf.write(" ".join(words) + "\n")
                    n_written += 1
                j += 1
                want = next(keep, None)
            kept_index += 1
    assert n_checked == n_window_lines, f"replayed {n_checked} of {n_window_lines} window lines"
    assert n_written == sum(counted), f"wrote {n_written} counted lines, expected {sum(counted)}"
    return {"phones": phones_path, "words": words_path, "kept_corpus_lines": int(n_kept),
            "window_lines": int(n_window_lines), "counted_lines": int(n_written)}
