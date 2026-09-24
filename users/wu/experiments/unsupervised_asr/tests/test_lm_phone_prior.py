"""Tests of ``lm/phone_prior.py`` on a tiny corpus: the Witten-Bell fit (normalisation, the unigram
by hand, the trigram's back-off context), the held-out split, the job end to end and the
``prior.npz`` round trip."""

import gzip

import numpy as np
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as P

CORPUS = ["<SIL> AA B <SIL>", "<SIL> AA B AA <SIL>", "B IY", "<SIL> IY IY B <SIL>"] * 30


def _write(tmp_path, lines):
    path = tmp_path / "corpus.phn.gz"
    with gzip.open(path, "wt") as fh:
        fh.write("\n".join(lines) + "\n")
    return str(path)


def test_rows_are_distributions(tmp_path):
    prior, _ = P.fit_prior(_write(tmp_path, CORPUS), n_count_lines=100, n_held_lines=10,
                           held_stride=11)
    for table in (prior.log_uni[None, :], prior.log_bi, prior.log_tri):
        assert np.allclose(np.exp(table).sum(axis=-1), 1.0, atol=1e-12)


def test_unigram_by_hand(tmp_path):
    counts, held = P.count_ngrams(([t for t in l.split()] for l in CORPUS), held_stride=0)
    assert held == []
    prior = P.PhoneNgramPrior.from_counts(counts)
    uni = counts.unigram.astype(float)
    n, t = uni.sum(), float((uni > 0).sum())
    expect = (uni + t / P.N_TYPES) / (n + t)
    assert np.allclose(np.exp(prior.log_uni), expect, rtol=0, atol=1e-15)
    # "<SIL>" is counted as SIL
    assert counts.unigram[P.SIL_ID] == 30 * 6


def test_trigram_backs_off_on_h1(tmp_path):
    """An unseen trigram context falls back to p2[h1] -- the row's MOST RECENT phone."""
    counts, _ = P.count_ngrams([l.split() for l in CORPUS])
    prior = P.PhoneNgramPrior.from_counts(counts)
    h2, h1 = P.PHONE2ID["IY"], P.PHONE2ID["AA"]  # "IY AA" never occurs
    assert counts.trigram[h2 * P.N_CTX + h1].sum() == 0
    assert np.array_equal(prior.log_tri[h2 * P.N_CTX + h1], prior.log_bi[h1])


def test_held_split_and_job(tmp_path):
    corpus = _write(tmp_path, CORPUS)
    counts, held = P.count_ngrams(P.read_phone_lines(corpus, limit=110), held_stride=11)
    assert len(held) == 10 and counts.n_lines == 100
    job = P.PhoneNgramPriorJob(corpus_phn=tk.Path(corpus), n_count_lines=100, n_held_lines=10)
    job.out_prior = tk.Path(str(tmp_path / "prior.npz"))
    job.out_stats = tk.Path(str(tmp_path / "prior.stats.txt"))
    job.out_json = tk.Path(str(tmp_path / "prior.json"))
    job.run()
    loaded = P.PhoneNgramPrior.load(str(tmp_path / "prior.npz"))
    ref, stats = P.fit_prior(corpus, n_count_lines=100, n_held_lines=10)
    for key in ("log_uni", "log_bi", "log_tri", "tri_counts"):
        assert np.array_equal(getattr(loaded, key), getattr(ref, key)), key
    assert loaded.meta["lines_counted"] == stats["lines_counted"]
    assert np.isfinite(stats["held_ppl_order3"]) and stats["held_ppl_order3"] <= stats["held_ppl_order1"]
    with np.load(str(tmp_path / "prior.npz")) as d:
        assert sorted(d.files) == ["log_bi", "log_tri", "log_uni", "meta", "phones", "tri_counts"]


def test_log_prob_has_no_end_term(tmp_path):
    counts, _ = P.count_ngrams([l.split() for l in CORPUS])
    prior = P.PhoneNgramPrior.from_counts(counts)
    seq = [P.PHONE2ID[p] for p in ("SIL", "AA", "B")]
    expect = (prior.log_tri[P.BOS_ID * P.N_CTX + P.BOS_ID, seq[0]]
              + prior.log_tri[P.BOS_ID * P.N_CTX + seq[0], seq[1]]
              + prior.log_tri[seq[0] * P.N_CTX + seq[1], seq[2]])
    assert prior.log_prob(seq, order=3) == expect
