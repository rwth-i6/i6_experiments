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


# --- oracle tests (test plan 2026-09-24, T1.12 and T1.13) -------------------------------------------

import random  # noqa: E402

import pytest  # noqa: E402

from wb_oracle import WittenBellOracle  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.model import emc_model as EM  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L  # noqa: E402

# CORPUS plus lines that exercise a one-token line, a SIL-only line, repeated phones and a context
# (IY IY) whose trigram row differs from its bigram back-off
ORACLE_CORPUS = CORPUS + ["AA", "AA AA B", "<SIL>", "B IY IY IY"] * 7
_ALIAS = {"<SIL>": "SIL"}


def _ids(line):
    """The oracle's own token -> id map (the inventory order is the only shared constant)."""
    return [P.PHONE2ID[_ALIAS.get(t, t)] for t in line.split()]


def _oracle_prior():
    counts, _ = P.count_ngrams([l.split() for l in ORACLE_CORPUS])
    return P.PhoneNgramPrior.from_counts(counts), counts


def test_witten_bell_tables_match_independent_oracle():
    """T1.12: every entry of log_uni, log_bi (41 x 40) and log_tri (1681 x 40), BOS and unseen rows
    included, equals the dict-based interpolated Witten-Bell oracle to 1e-12."""
    prior, counts = _oracle_prior()
    oracle = WittenBellOracle([_ids(l) for l in ORACLE_CORPUS], n_types=P.N_TYPES, bos=P.BOS_ID)
    uni, bi, tri = (np.asarray(t) for t in oracle.tables())
    assert prior.log_bi.shape == bi.shape == (P.N_CTX, P.N_TYPES)
    assert prior.log_tri.shape == tri.shape == (P.N_CTX * P.N_CTX, P.N_TYPES)
    assert np.abs(prior.log_uni - uni).max() <= 1e-12
    assert np.abs(prior.log_bi - bi).max() <= 1e-12
    assert np.abs(prior.log_tri - tri).max() <= 1e-12
    # the instance is not vacuous: seen, unseen and BOS rows all occur, and a trigram row differs
    # from its bigram back-off
    assert counts.trigram[P.BOS_ID * P.N_CTX + P.BOS_ID].sum() == len(ORACLE_CORPUS)
    iy = P.PHONE2ID["IY"]
    assert not np.allclose(prior.log_tri[iy * P.N_CTX + iy], prior.log_bi[iy])
    # no context crosses a line: a history (h2 != BOS, h1 = BOS) is never counted
    tri = counts.trigram.reshape(P.N_CTX, P.N_CTX, P.N_TYPES)
    assert tri[: P.N_TYPES, P.BOS_ID].sum() == 0


_SINGLE_TOKEN_XFAIL = pytest.mark.xfail(
    strict=True,
    reason="FINDING model/prior.py:246: per_token_log_probs(order=3) of a ONE-token sequence returns "
           "two terms (h2 = [BOS, BOS] + ids[:-2] has length 2), so log_prob/perplexity count "
           "log P(y0 | BOS, BOS) twice",
)


def _held_split(stride=11):
    counts, held = P.count_ngrams([l.split() for l in ORACLE_CORPUS], held_stride=stride)
    counted = [_ids(l) for i, l in enumerate(ORACLE_CORPUS) if i % stride != 0]
    held_oracle = [_ids(l) for i, l in enumerate(ORACLE_CORPUS) if i % stride == 0]
    assert held == held_oracle and len(held) > 5
    return P.PhoneNgramPrior.from_counts(counts), WittenBellOracle(counted, n_types=P.N_TYPES, bos=P.BOS_ID), held


@pytest.mark.parametrize("order", [1, 2, pytest.param(3, marks=_SINGLE_TOKEN_XFAIL)])
def test_perplexity_matches_oracle_on_held_lines(order):
    """T1.12: perplexity() on the held-out lines equals the oracle's, with the oracle fitted on the
    counted (non-held) lines only. The held lines include one-token lines ("AA", "<SIL>")."""
    prior, oracle, held = _held_split()
    assert any(len(l) == 1 for l in held)
    got, expect = prior.perplexity(held, order), oracle.perplexity(held, order)
    assert abs(got - expect) <= 1e-12 * expect, (order, got, expect)


def test_perplexity_order3_matches_oracle_on_held_lines_of_two_or_more_tokens():
    """T1.12 at order 3 without the one-token lines: pins that the defect above is confined to them."""
    prior, oracle, held = _held_split()
    multi = [l for l in held if len(l) >= 2]
    assert len(multi) > 5
    got, expect = prior.perplexity(multi, 3), oracle.perplexity(multi, 3)
    assert abs(got - expect) <= 1e-12 * expect, (got, expect)


def _random_strings(n=500, seed=0):
    rng = random.Random(seed)
    out = [[P.SIL_ID], [P.SIL_ID, P.SIL_ID], [0, 0, 0]]  # SIL-only and repeated strings, pinned
    while len(out) < n:
        out.append([rng.randrange(P.N_TYPES) for _ in range(rng.randint(1, 8))])
    return out


def _walk(prior, name, order, strings):
    """T1.13 body: walk the lattice history over each string and compare with the prior's scorer."""
    cfg = L.LatticeConfig()  # 40 phones, SIL = 39, BOS = 40: the bed's inventory
    assert cfg.n_phones == P.N_TYPES and cfg.sil_id == P.SIL_ID and cfg.bos_id == P.BOS_ID
    table = EM.SaeEmcModelV1._prior_table(prior, name)
    assert table is (prior.log_tri if name == "trigram" else prior.log_bi)
    hist = L.build_prior_history(cfg, name)
    assert int(hist.last[hist.start]) == P.BOS_ID
    for y in strings:
        h, h2, h1, terms = hist.start, P.BOS_ID, P.BOS_ID, []
        for k in y:
            # the row the definition wants: (h2, h1) for the trigram, h1 for the bigram
            assert h == (h2 * P.N_CTX + h1 if order == 3 else h1), (name, y, h)
            terms.append(table[h, k])
            h = int(hist.next[h, k])
            h2, h1 = h1, k
        assert int(hist.last[h]) == y[-1]
        assert list(prior.per_token_log_probs(y, order=order)) == terms, (name, y)
        assert float(np.asarray(terms).sum()) == prior.log_prob(y, order=order), (name, y)


@pytest.mark.parametrize("name, order", [("trigram", 3), ("bigram", 2)])
def test_prior_table_layout_in_the_lattice_history(name, order):
    """T1.13: the lattice's trigram / bigram table is prior.log_tri / log_bi; walking
    h <- hist.next[h, k] from hist.start reads exactly the per-token log-probs of
    prior.log_prob(y, order) and ends in a history whose last(h) is the last token.
    One-token strings at order 3 are split off into the xfail below."""
    prior, _ = _oracle_prior()
    strings = [y for y in _random_strings() if order != 3 or len(y) >= 2]
    assert len(strings) > 400
    _walk(prior, name, order, strings)


@_SINGLE_TOKEN_XFAIL
def test_prior_table_layout_one_token_strings_trigram():
    """T1.13 on the one-token strings at order 3 (the lattice walk reads one term; the scorer two)."""
    prior, _ = _oracle_prior()
    _walk(prior, "trigram", 3, [y for y in _random_strings() if len(y) == 1])
