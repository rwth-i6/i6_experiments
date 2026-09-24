"""Tests of ``lm/phone_text.py`` on tiny inputs: the SIL-augmented phonemization (the per-line draw
protocol, OOV dropping, the job end to end on a tiny bliss + g2p lexicon), the seeded uniform line
sample and the content-based line filter."""

import gzip
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
