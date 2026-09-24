"""Tests of ``lm/phone_text.py`` on tiny inputs: the SIL-augmented phonemization (the per-line draw
protocol, OOV dropping, the job end to end on a tiny bliss + g2p lexicon), the seeded uniform line
sample and the content-based line filter."""

import gzip
import math
import random

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_text as T

W2P = {"A": "AH", "B": "B IY", "C": "S IY"}


def test_phonemize_line_draw_protocol():
    """One ``random_sample(W - 1)`` per kept line, SIL after word i iff draw i < sil_prob."""
    draws = np.random.RandomState(0).random_sample(2)  # [0.5488..., 0.7151...]
    assert draws[0] < 0.6 <= draws[1]
    out = T.phonemize_line(["A", "B", "C"], W2P, 0.6, True, np.random.RandomState(0))
    assert out == ["<SIL>", "AH", "<SIL>", "B", "IY", "S", "IY", "<SIL>"]
    out = T.phonemize_line(["A", "B", "C"], W2P, 0.5, False, np.random.RandomState(0))
    assert out == ["AH", "B", "IY", "S", "IY"]
    # sil_prob = 0 and a one-word line draw nothing
    rng = np.random.RandomState(0)
    assert T.phonemize_line(["A"], W2P, 0.5, True, rng) == ["<SIL>", "AH", "<SIL>"]
    assert T.phonemize_line(["A", "B"], W2P, 0.0, True, rng) == ["<SIL>", "AH", "B", "IY", "<SIL>"]
    assert rng.random_sample() == np.random.RandomState(0).random_sample()


def test_phonemize_line_drops_oov_and_empty():
    rng = np.random.RandomState(0)
    assert T.phonemize_line(["A", "ZZZ"], W2P, 0.5, True, rng) is None
    assert T.phonemize_line([], W2P, 0.5, True, rng) is None
    # a dropped line consumes no draw
    assert rng.random_sample() == np.random.RandomState(0).random_sample()


BLISS = """<?xml version="1.0" encoding="utf-8"?>
<lexicon>
  <lemma special="silence"><orth>[SILENCE]</orth><phon>SIL</phon></lemma>
  <lemma><orth>A</orth><phon>AH</phon><phon>EY</phon></lemma>
  <lemma><orth>B</orth><phon>B IY</phon></lemma>
</lexicon>
"""
G2P = "B\t0\t0.9\tB AH\nC\t0\t0.9\tS IY\nC\t1\t0.1\tK IY\n"


def test_phonemize_job_on_a_tiny_lexicon(tmp_path):
    (tmp_path / "lex.xml").write_text(BLISS)
    (tmp_path / "g2p.txt").write_text(G2P)
    with gzip.open(tmp_path / "text.gz", "wt") as fh:
        fh.write("A B C\nA D\nC A\n\nB\n")
    job = T.PhonemizeWithSilJob(text_file=tk.Path(str(tmp_path / "text.gz")),
                                bliss_lexicon=tk.Path(str(tmp_path / "lex.xml")),
                                g2p_lexicon=tk.Path(str(tmp_path / "g2p.txt")))
    job.out_text = tk.Path(str(tmp_path / "out.phn.gz"))
    job.out_stats = tk.Path(str(tmp_path / "stats.txt"))
    job.run()
    # bliss first (A -> its FIRST pron, B -> bliss not g2p), g2p for the rest (C -> first entry)
    w2p = {"A": "AH", "B": "B IY", "C": "S IY"}
    rng = np.random.RandomState(0)
    expect = [T.phonemize_line(l.split(), w2p, 0.5, True, rng) for l in ("A B C", "A D", "C A", "", "B")]
    expect = [" ".join(p) for p in expect if p is not None]
    with gzip.open(tmp_path / "out.phn.gz", "rt") as fh:
        assert fh.read().splitlines() == expect
    stats = dict(l.split("=") for l in (tmp_path / "stats.txt").read_text().split())
    assert stats["lines_in"] == "5" and stats["lines_out"] == "3" and stats["dropped_oov"] == "2"


def test_sample_line_indices():
    keep = T.sample_line_indices(100, 10, 0)
    assert keep == sorted(random.Random(0).sample(range(100), 10))
    assert len(set(keep)) == 10 and keep == sorted(keep)
    assert T.sample_line_indices(100, 10, 0) == keep != T.sample_line_indices(100, 10, 1)
    assert T.sample_line_indices(5, 5, 3) == [0, 1, 2, 3, 4]
    with pytest.raises(AssertionError):
        T.sample_line_indices(5, 6, 0)


def test_iter_sampled_streams_in_order():
    lines = [f"l{i}" for i in range(10)]
    assert list(T.iter_sampled(iter(lines), [1, 4, 9])) == ["l1", "l4", "l9"]
    with pytest.raises(AssertionError):
        list(T.iter_sampled(iter(lines), [4, 1]))


def test_sample_lines_job(tmp_path):
    lines = [f"<SIL> {'AA' if i % 3 else 'IY'} B <SIL>" for i in range(50)]
    with gzip.open(tmp_path / "corpus.gz", "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    job = T.SampleLinesJob(corpus=tk.Path(str(tmp_path / "corpus.gz")), n_out=7, seed=0)
    job.out_corpus = tk.Path(str(tmp_path / "sample.gz"))
    job.out_stats = tk.Path(str(tmp_path / "stats.txt"))
    job.run()
    with gzip.open(tmp_path / "sample.gz", "rt") as fh:
        got = fh.read().splitlines()
    assert got == [lines[i] for i in T.sample_line_indices(50, 7, 0)]
    assert T.count_lines(str(tmp_path / "corpus.gz")) == 50


# ---------------------------------------------------------------------------------------------------
# T3.5 (test plan 2026-09-24): SIL insertion statistics of the job at its defaults (sil_prob 0.5,
# surround, seed 0) on 20,000 lines, and the count pass's mapping of ``<SIL>`` onto SIL_ID.
# ---------------------------------------------------------------------------------------------------
# one-phone words, so every token position is a word or a boundary and SILs can be attributed
_BLISS_1PH = """<?xml version="1.0" encoding="utf-8"?>
<lexicon>
  <lemma><orth>W1</orth><phon>AA</phon></lemma>
  <lemma><orth>W2</orth><phon>B</phon></lemma>
  <lemma><orth>W3</orth><phon>IY</phon></lemma>
</lexicon>
"""


@pytest.fixture(scope="module")
def sil_corpus(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sil")
    (tmp / "lex.xml").write_text(_BLISS_1PH)
    rng = random.Random(0)
    lines = [" ".join(rng.choice(["W1", "W2", "W3"]) for _ in range(rng.randint(1, 30))) for _ in range(20000)]
    with gzip.open(tmp / "text.gz", "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    job = T.PhonemizeWithSilJob(text_file=tk.Path(str(tmp / "text.gz")), bliss_lexicon=tk.Path(str(tmp / "lex.xml")))
    assert (job.sil_prob, job.surround, job.seed) == (0.5, True, 0)  # the defaults the corpus uses
    job.out_text = tk.Path(str(tmp / "out.phn.gz"))
    job.out_stats = tk.Path(str(tmp / "stats.txt"))
    job.run()
    with gzip.open(tmp / "out.phn.gz", "rt") as fh:
        out = [l.split() for l in fh.read().splitlines()]
    return lines, out, str(tmp / "out.phn.gz")


def test_sil_insertion_statistics(sil_corpus):
    lines, out, _ = sil_corpus
    assert len(out) == len(lines) == 20000
    n_boundaries = n_sil = 0
    for words, toks in zip(lines, out):
        words = words.split()
        assert toks[0] == T.SIL and toks[-1] == T.SIL  # the line edges: always SIL
        inner = toks[1:-1]
        assert inner[0] != T.SIL and inner[-1] != T.SIL  # no second SIL at an edge
        assert [t for t in inner if t != T.SIL] == [{"W1": "AA", "W2": "B", "W3": "IY"}[w] for w in words]
        assert all(not (a == T.SIL and b == T.SIL) for a, b in zip(inner, inner[1:]))  # at most one per boundary
        n_boundaries += len(words) - 1
        n_sil += sum(t == T.SIL for t in inner)
    p = 0.5
    sigma = math.sqrt(n_boundaries * p * (1 - p))
    assert n_boundaries > 100000
    assert abs(n_sil - p * n_boundaries) <= 4 * sigma, (n_sil, n_boundaries, sigma)


def test_count_ngrams_maps_sil_token(sil_corpus):
    from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as PP

    _, out, path = sil_corpus
    counts, held = PP.count_ngrams(PP.read_phone_lines(path))
    assert held == [] and counts.n_lines == len(out)
    n_sil = sum(t == "<SIL>" for toks in out for t in toks)
    assert int(counts.unigram[PP.SIL_ID]) == n_sil
    assert int(counts.unigram.sum()) == counts.n_tokens == sum(len(t) for t in out)
    # every line opens with <SIL> after the two BOS pads
    assert int(counts.bigram[PP.BOS_ID, PP.SIL_ID]) == len(out) == int(counts.bigram[PP.BOS_ID].sum())
    assert int(counts.trigram[PP.BOS_ID * PP.N_CTX + PP.BOS_ID, PP.SIL_ID]) == len(out)
    # the other spellings of silence land on the same id; an unknown token is an error
    c2, _ = PP.count_ngrams([["<SIL>", "AA", "[SIL]", "sil", "SIL"]])
    assert int(c2.unigram[PP.SIL_ID]) == 4 and int(c2.unigram[PP.PHONE2ID["AA"]]) == 1
    with pytest.raises(KeyError):
        PP.count_ngrams([["<SIL>", "ZZ"]])
