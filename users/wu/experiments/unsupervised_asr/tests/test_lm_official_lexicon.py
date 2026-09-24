"""T3.6 (test plan 2026-09-24): the official-lexicon reader behind the k2 arms' ``L``
(``lm/lexlat_k2_official.py``): stress stripped, SIL / out-of-inventory entries dropped and counted,
several pronunciations per word kept, restriction to ``G``'s vocabulary, and the null's
derangement moving whole pronunciation blocks with no fixed point.  No k2 needed."""

from __future__ import annotations

import gzip
from collections import Counter

import pytest

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import lexlat_k2_official as O
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import PHONE2ID

LEXICON = """\
A  AH0
A  EY1
ABOUT  AH0 B AW1 T
READ  R EH1 D
READ  R IY1 D
READ  R EH2 D
HELLO  HH AH0 L OW1
HELLO  HH EH0 L OW1
FOO  F UW1 XX
BAR  B AA1 SIL
EMPTY
NOTINLM  N AA1 T
<s>  S IY1

ZED  Z EH1 D
"""
# (duplicate READ R EH2 D -> R EH D after stripping; FOO has a phone outside the inventory; BAR
#  carries SIL; EMPTY has no pronunciation; NOTINLM is not in G; <s> is a non-lexical marker)

EXPECT_ENTRIES = [
    ("A", ("AH",)), ("A", ("EY",)), ("ABOUT", ("AH", "B", "AW", "T")), ("READ", ("R", "EH", "D")),
    ("READ", ("R", "IY", "D")), ("HELLO", ("HH", "AH", "L", "OW")), ("HELLO", ("HH", "EH", "L", "OW")),
    ("NOTINLM", ("N", "AA", "T")), ("<s>", ("S", "IY")), ("ZED", ("Z", "EH", "D")),
]
LM_WORDS = ["<s>", "</s>", "<UNK>", "A", "ABOUT", "HELLO", "READ", "ZED", "UNSPOKEN"]


@pytest.fixture(params=["txt", "gz"])
def lexicon_path(tmp_path, request):
    if request.param == "gz":
        p = tmp_path / "librispeech-lexicon.txt.gz"
        with gzip.open(p, "wt") as fh:
            fh.write(LEXICON)
    else:
        p = tmp_path / "librispeech-lexicon.txt"
        p.write_text(LEXICON)
    return str(p)


def test_read_official_lexicon(lexicon_path):
    entries, stats = O.read_official_lexicon(lexicon_path)
    assert entries == EXPECT_ENTRIES
    assert stats == {"lines": 14, "no_pron": 1, "oov_phone": 1, "has_sil": 1, "duplicate": 1,
                     "kept": len(EXPECT_ENTRIES)}
    assert stats["lines"] == stats["no_pron"] + stats["oov_phone"] + stats["has_sil"] + stats["duplicate"] + stats["kept"]
    assert all(p in PHONE2ID and p != "SIL" and not p[-1].isdigit() for _w, pron in entries for p in pron)


def test_official_prons_restricted_to_g(lexicon_path):
    entries, _ = O.read_official_lexicon(lexicon_path)
    prons, stats = O.official_prons(entries, LM_WORDS)
    wid = {w: i for i, w in enumerate(LM_WORDS)}
    expect = sorted((wid[w], [PHONE2ID[p] for p in pron]) for w, pron in EXPECT_ENTRIES
                    if w in wid and w not in O.NON_LEXICAL_WORDS)
    assert prons == expect
    assert stats == {"entries": len(EXPECT_ENTRIES), "oov_word": 1, "dropped_marker": 1,
                     "words_with_pron": 5, "kept": 8}
    # several pronunciations per word survive
    per_word = Counter(w for w, _ in prons)
    assert per_word[wid["A"]] == 2 and per_word[wid["READ"]] == 2 and per_word[wid["HELLO"]] == 2
    # never a pronunciation for a sentence marker or the unknown word; a G word without an entry
    # simply has none
    assert not {wid["<s>"], wid["</s>"], wid["<UNK>"], wid["UNSPOKEN"]} & set(per_word)


def _blocks(prons, words):
    out = {}
    for w, pron in prons:
        out.setdefault(words[w], []).append(tuple(pron))
    return {w: sorted(v) for w, v in out.items()}


def test_derange_official_prons_moves_whole_blocks(lexicon_path):
    entries, _ = O.read_official_lexicon(lexicon_path)
    prons, _ = O.official_prons(entries, LM_WORDS)
    before = _blocks(prons, LM_WORDS)
    out, stats = O.derange_official_prons(prons, LM_WORDS)
    after = _blocks(out, LM_WORDS)
    assert out == sorted(out, key=lambda t: (t[0], t[1]))
    assert set(after) == set(before)  # the same words carry pronunciations
    # every new block IS one old block (whole-block moves) and the blocks are a permutation
    assert sorted(after.values()) == sorted(before.values())
    assert sorted(tuple(p) for _w, p in out) == sorted(tuple(p) for _w, p in prons)
    # no fixed point: all blocks are distinct here, so no word keeps its own
    assert all(after[w] != before[w] for w in before)
    assert stats["fixed_points"] == 0 and stats["kept_by_homophony"] == 0
    assert stats["n_pronunciations"] == len(prons) and stats["n_words_with_pronunciation"] == len(before)
    assert stats["seed"] == O.DERANGEMENT_SEED == 0
    # one cycle (Sattolo): following word -> source word visits every word
    src = {w: next(v for v, b in before.items() if b == after[w]) for w in before}
    w0 = sorted(before)[0]
    cyc, w = [w0], src[w0]
    while w != w0:
        cyc.append(w)
        w = src[w]
    assert len(cyc) == len(before)
    # deterministic in the seed
    assert O.derange_official_prons(prons, LM_WORDS)[0] == out
    other = _blocks(O.derange_official_prons(prons, LM_WORDS, seed=1)[0], LM_WORDS)
    assert all(other[w] != before[w] for w in before)


def test_derange_official_prons_homophone_residual_is_reported():
    words = ["X", "Y", "Z"]
    prons = [(0, [PHONE2ID["AA"]]), (1, [PHONE2ID["AA"]]), (2, [PHONE2ID["B"]])]
    out, stats = O.derange_official_prons(prons, words)
    after = _blocks(out, words)
    before = _blocks(prons, words)
    kept = sum(after[w] == before[w] for w in words)
    assert stats["kept_by_homophony"] == kept
