"""Word-LM tables: parsers, pruning, compaction (test plan 2026-09-24, T1.23; item S3).

The fixture LMs (``tests/k2_oracle.py``) are NORMALISED back-off LMs of order 3 and 4 whose
back-off weights are derived so every context sums to one; the ARPA text is read independently by
``k2_oracle.ArpaScorer`` (dicts + the textbook back-off recursion) and the CSR tables are walked
independently by ``k2_oracle.csr_walk``.

S3: ``lexlat_k2.prune_lm_tables`` re-derives each state's back-off weight against the UNPRUNED
lower order (``lexlat_k2.py:715-725``) while it prunes the lower-order arcs in the same pass, so a
context of order >= 3 whose back-off path loses an arc no longer sums to one.  The docstring
(``lexlat_k2.py:679-685``) claims it does.  Values the plan says to RECORD are printed with a
``[record]`` label.
"""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

import k2_oracle as O  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import lexlat_k2_official as OFF  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2 as K  # noqa: E402

pytestmark = pytest.mark.k2

THETAS = (0.5, 2.0, 5.0)
S3_REASON = ("S3 CONFIRMED: prune_lm_tables re-derives back-off weights from the UNPRUNED lower "
             "order (lexlat_k2.py:715-725) while pruning lower-order arcs, so order>=3 contexts "
             "whose back-off path lost an arc do not sum to one (docstring lexlat_k2.py:679-685)")


def _silent(*_a, **_k):
    pass


@pytest.fixture(scope="module")
def lms(tmp_path_factory):
    root = tmp_path_factory.mktemp("lm_prune")
    a3 = O.write_arpa(str(root / "lm3.arpa"), O.LM3_EXPLICIT)
    a4 = O.write_arpa(str(root / "lm4.arpa"), O.LM4_EXPLICIT)
    return {
        3: {"arpa": a3, "scorer": O.ArpaScorer(a3), "inhouse": lexlat.parse_arpa_word_lm(a3),
            "official": OFF.parse_arpa_lm(a3, log=_silent)},
        4: {"arpa": a4, "scorer": O.ArpaScorer(a4), "official": OFF.parse_arpa_lm(a4, log=_silent)},
    }


def _tables(lms, order):
    """The table the production path prunes: the in-house parser at order 3, the official one at 4."""
    return lms[order]["inhouse"] if order == 3 else lms[order]["official"]


def _histories(max_len):
    for n in range(max_len + 1):
        yield from itertools.product(O.VOCAB, repeat=n)


def _close(a, b):
    return a == pytest.approx(b, rel=1e-6, abs=1e-6)


# ================================================================================================
# (a) parsers against the independent back-off scorer
# ================================================================================================
def test_t123a_fixture_lms_are_normalised(lms):
    for order in (3, 4):
        sc = lms[order]["scorer"]
        for h in _histories(order - 1):
            total = sum(math.exp(sc.logp(h, w)) for w in O.VOCAB)
            assert abs(total - 1.0) <= 1e-9, (order, h, total)


@pytest.mark.parametrize("which", ["inhouse", "official"])
def test_t123a_order3_parsers_match_scorer(lms, which):
    tables, sc = lms[3][which], lms[3]["scorer"]
    wid = {w: i for i, w in enumerate(tables["words"])}
    assert [str(w) for w in tables["words"]] == list(O.VOCAB)
    for h in _histories(2):
        s = O.csr_state(tables, [wid[w] for w in h])
        lookup, _ = K._lookup(tables, np.full(len(O.VOCAB), s), np.arange(len(O.VOCAB)))
        for j, w in enumerate(O.VOCAB):
            want = sc.logp(h, w)
            assert _close(float(lookup[j]), want), (which, h, w, float(lookup[j]), want)
            assert _close(O.csr_walk(tables, s, j)[0], want), (which, h, w)


def test_t123a_order3_parsers_agree(lms):
    a, b = lms[3]["inhouse"], lms[3]["official"]
    for key in ("arc_key", "arc_next", "backoff_state"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
    for key in ("arc_logp", "backoff"):
        assert np.array_equal(np.asarray(a[key], dtype=np.float32),
                              np.asarray(b[key], dtype=np.float32)), key
    assert int(a["begin_state"]) == int(b["begin_state"])


def test_t123a_order4_parser_matches_scorer_at_levels_4(lms):
    tables, sc = lms[4]["official"], lms[4]["scorer"]
    assert int(tables["order"]) == 4
    wid = {w: i for i, w in enumerate(tables["words"])}
    for h in _histories(3):
        s = O.csr_state(tables, [wid[w] for w in h])
        lookup, _ = K._lookup(tables, np.full(len(O.VOCAB), s), np.arange(len(O.VOCAB)), levels=4)
        for j, w in enumerate(O.VOCAB):
            assert _close(float(lookup[j]), sc.logp(h, w)), (h, w)


def test_t123a_levels3_on_order4_returns_zero_for_a_unigram_only_word(lms):
    """lexlat.lm_score's documented silent failure (lexlat.py:512-515, 522-528): three levels from
    an order-4 context cannot reach a word that only has a unigram arc, and it returns log p = 0."""
    tables, sc = lms[4]["official"], lms[4]["scorer"]
    wid = {w: i for i, w in enumerate(tables["words"])}
    h, w = ("<s>", "a", "bc"), "ab"  # 'ab' has no arc at (<s> a bc), (a bc) or (bc)
    s = O.csr_state(tables, [wid[x] for x in h])
    res = SimpleNamespace(
        n_words=int(tables["n_words"]),
        arc_key=torch.as_tensor(np.asarray(tables["arc_key"]), dtype=torch.int64),
        arc_logp=torch.as_tensor(np.asarray(tables["arc_logp"]), dtype=torch.float64),
        arc_next=torch.as_tensor(np.asarray(tables["arc_next"]), dtype=torch.int64),
        backoff=torch.as_tensor(np.asarray(tables["backoff"]), dtype=torch.float64),
        backoff_state=torch.as_tensor(np.asarray(tables["backoff_state"]), dtype=torch.int64))
    st, wd = torch.tensor([s]), torch.tensor([wid[w]])
    lp3, nx3 = lexlat.lm_score(res, st, wd, levels=3)
    lp4, _ = lexlat.lm_score(res, st, wd, levels=4)
    assert float(lp3[0]) == 0.0 and int(nx3[0]) == 0
    assert _close(float(lp4[0]), sc.logp(h, w)) and float(lp4[0]) < -1.0
    print(f"[record] T1.23a levels=3 on the order-4 table: log p(ab | <s> a bc) = {float(lp3[0])} "
          f"(true {sc.logp(h, w):.6f})")


# ================================================================================================
# (b) pruning
# ================================================================================================
def _arc_set(tables, contexts):
    v = int(tables["n_words"])
    key = np.asarray(tables["arc_key"], dtype=np.int64)
    return {contexts[int(k // v)] + (O.VOCAB[int(k % v)],) for k in key}


def _expected_kept(sc, theta):
    """Every unigram, and every higher-order n-gram with |logp - (bo(h) + log p(w | h[1:]))| >=
    theta, recomputed from the ARPA text."""
    kept = set()
    margins = []
    for ngram, lp in sc.lp.items():
        h, w = ngram[:-1], ngram[-1]
        if not h:
            kept.add(ngram)
            continue
        gap = abs(lp - (sc.bo.get(h, 0.0) + sc.logp(h[1:], w)))
        margins.append(abs(gap - theta))
        if gap >= theta:
            kept.add(ngram)
    return kept, min(margins)


def _state_sums(tables):
    v = int(tables["n_words"])
    return np.array([sum(math.exp(O.csr_walk(tables, s, w)[0]) for w in range(v))
                     for s in range(int(tables["n_states"]))])


@pytest.mark.parametrize("order", [3, 4])
def test_t123b_theta0_is_identity(lms, order):
    tables = _tables(lms, order)
    out, stats = K.prune_lm_tables(tables, 0.0)
    for key in ("arc_key", "arc_logp", "arc_next", "backoff", "backoff_state"):
        assert out[key] is tables[key] or np.array_equal(out[key], tables[key]), key
    assert stats["arcs_after"] == stats["arcs_before"]


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("order", [3, 4])
def test_t123b_kept_arcs_follow_the_criterion(lms, order, theta):
    tables, sc = _tables(lms, order), lms[order]["scorer"]
    contexts = sc.state_contexts()
    assert len(contexts) == int(tables["n_states"])
    assert _arc_set(tables, contexts) == set(sc.lp)  # the numbering reads the ARPA back exactly
    out, stats = K.prune_lm_tables(tables, theta)
    want, margin = _expected_kept(sc, theta)
    assert margin > 1e-4, f"an n-gram sits within {margin} nats of theta: the fixture is ambiguous"
    assert _arc_set(out, contexts) == want
    assert stats["arcs_after"] == len(want)


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("order", [3, 4])
def test_t123b_low_order_contexts_sum_to_one(lms, order, theta):
    tables = _tables(lms, order)
    out, _ = K.prune_lm_tables(tables, theta)
    sums = _state_sums(out)
    st_order = K.state_orders(out)
    low = st_order <= 2
    assert np.all(np.abs(sums[low] - 1.0) <= 1e-6), sums[low]
    # _lookup (the production walk) and the independent walk agree on the pruned tables
    v = int(out["n_words"])
    s_all = np.repeat(np.arange(int(out["n_states"])), v)
    w_all = np.tile(np.arange(v), int(out["n_states"]))
    lp, _ = K._lookup(out, s_all, w_all)
    mine = np.array([O.csr_walk(out, s, w)[0] for s, w in zip(s_all, w_all)])
    assert np.allclose(lp, mine, atol=1e-9, rtol=0)


def _high_order_record(lms, order, theta):
    tables, sc = _tables(lms, order), lms[order]["scorer"]
    contexts = sc.state_contexts()
    out, _ = K.prune_lm_tables(tables, theta)
    sums = _state_sums(out)
    high = np.nonzero(K.state_orders(out) >= 3)[0]
    dev = np.abs(sums[high] - 1.0)
    worst = int(high[int(np.argmax(dev))])
    return out, sums, high, dev, contexts[worst], contexts


@pytest.mark.parametrize("theta", THETAS)
@pytest.mark.parametrize("order", [3, 4])
def test_t123b_record_high_order_normalisation(lms, order, theta):
    """RECORD max |sum_w p(w | s) - 1| over contexts of order >= 3 after pruning (S3)."""
    tables, sc = _tables(lms, order), lms[order]["scorer"]
    out, sums, high, dev, worst_ctx, contexts = _high_order_record(lms, order, theta)
    kept_before = _arc_set(tables, contexts)
    kept_after = _arc_set(out, contexts)
    # the fixture's precondition: a context of order >= 3 that keeps an explicit arc while a
    # context on its back-off path lost one
    def lost(h):
        return any(g[:-1] == h and g not in kept_after for g in kept_before)

    precondition = any(
        len(g) >= 3 and any(lost(g[i:-1]) for i in range(1, len(g) - 1))
        for g in kept_after)
    assert precondition, "the fixture no longer prunes an arc on a kept context's back-off path"
    assert np.all(np.isfinite(sums)) and np.all(sums > 0)
    print(f"[record] T1.23b order {order} theta {theta}: max |sum - 1| over {high.size} contexts of "
          f"order >= 3 = {float(dev.max()):.3e} at context {worst_ctx} (fixture precondition "
          f"holds: a kept context of order >= 3 whose back-off context lost an arc)")


@pytest.mark.xfail(strict=True, reason=S3_REASON)
@pytest.mark.parametrize("order,theta", [(3, 0.5), (4, 5.0)])
def test_t123b_docstring_claim_every_state_sums_to_one(lms, order, theta):
    """The claim of lexlat_k2.py:679-685 at the in-house ladder's first rung (order 3, 0.5) and at
    the default arm's official 4-gram rung (theta 5.0)."""
    _out, _sums, _high, dev, _ctx, _c = _high_order_record(lms, order, theta)
    assert float(dev.max()) <= 1e-6, float(dev.max())


# ================================================================================================
# (c) compaction
# ================================================================================================
def _reach(tables):
    """States reachable from begin_state over n-gram arcs and back-off arcs, by BFS."""
    v = int(tables["n_words"])
    key = np.asarray(tables["arc_key"], dtype=np.int64)
    nxt = np.asarray(tables["arc_next"], dtype=np.int64)
    bs = np.asarray(tables["backoff_state"], dtype=np.int64)
    out_arcs = {}
    for k, n in zip(key, nxt):
        out_arcs.setdefault(int(k // v), []).append(int(n))
    seen = {int(tables["begin_state"])}
    todo = list(seen)
    while todo:
        s = todo.pop()
        for t in out_arcs.get(s, []) + [int(bs[s])]:
            if t not in seen:
                seen.add(t)
                todo.append(t)
    return sorted(seen)


@pytest.mark.parametrize("theta", (0.0,) + THETAS)
@pytest.mark.parametrize("order", [3, 4])
def test_t123c_compaction_keeps_every_reachable_lookup(lms, order, theta):
    tables = _tables(lms, order)
    pruned, _ = K.prune_lm_tables(tables, theta)
    if order == 3:
        pruned = dict(pruned, order=3)
    out, stats = OFF.compact_lm_tables(pruned)
    reach = _reach(pruned)
    new_id = {s: i for i, s in enumerate(reach)}
    assert stats["states_after"] == len(reach) == int(out["n_states"])
    assert reach[0] == 0 and new_id[0] == 0  # the null context stays state 0
    assert int(out["begin_state"]) == new_id[int(pruned["begin_state"])]
    key = np.asarray(out["arc_key"], dtype=np.int64)
    assert np.all(np.diff(key) > 0)
    v = int(pruned["n_words"])
    for s in reach:
        for w in range(v):
            lp_a, nx_a = O.csr_walk(pruned, s, w)
            lp_b, nx_b = O.csr_walk(out, new_id[s], w)
            assert lp_a == lp_b and new_id[nx_a] == nx_b, (s, w)
    print(f"[record] T1.23c order {order} theta {theta}: states {stats['states_before']} -> "
          f"{stats['states_after']}")
