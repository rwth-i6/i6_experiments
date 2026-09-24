"""k2 lexicon term: numerics against a brute-force oracle (test plan 2026-09-24, T1.18-T1.22).

L_lex = log Z_HLG(e / tau) - log Z_H(e / tau) (``model/lexlat_k2_train.py``).  The oracle
(``tests/k2_oracle.py``) enumerates every L o G path of the tiny fixture explicitly and every frame
string a in 4^T, never calling the package's graph or scoring code; the package builds the fixture's
graph through its production CLI (``lexlat_k2.main(["build-hlg", ...])``), which is under test.

Items settled here: S4 (tropical epsilon removal, T1.22 / T1.19c), S5 (escape multiplicity and the
sign of L_lex, T1.19d-e), S6 (segment order, T1.21), S7 (the term that runs, T1.20).  Values the
plan says to RECORD are printed with a ``[record]`` label (run with ``-s`` to see them).
"""

from __future__ import annotations

import json
import math
import os
import statistics

import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

import k2_oracle as O  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2 as K  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2_train as KT  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.lattice import LatticeConfig  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.lexlat_train import lexlat_lambda  # noqa: E402

pytestmark = pytest.mark.k2

#: T1.19's operating point: pruning effectively off on the tiny graph
WIDE = dict(max_active=10000, search_beam=1000.0, output_beam=1000.0)
#: k2 tolerance of the plan (section 0): 1e-5, relative on log Z, absolute on gradients
TOL = 1e-5

S4_REASON = ("S4 CONFIRMED for backoff_loops='all': compile_hlg's k2.remove_epsilon "
             "(lexlat_k2.py:874) is tropical and merges the parallel epsilon routes the 'all' "
             "placement creates; the log-semiring total drops (up to 2.7 nats per string, T1.22)")


# ------------------------------------------------------------------------------------------------
# the fixture: resources and graphs built once through the production path
# ------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def fx(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("k2_fixture"))
    res = O.build_resources(root)
    graphs = {
        "word_boundary": O.build_hlg(res["npz"], os.path.join(root, "wb")),
        "all": O.build_hlg(res["npz"], os.path.join(root, "all"), backoff_loops="all"),
        "strict": O.build_hlg(res["npz"], os.path.join(root, "strict"), strict=True),
        "shuffled": O.build_hlg(res["npz"], os.path.join(root, "shuf"), shuffled=True),
    }
    g = O.GTables(res["npz"])
    caches = {
        "word_boundary": O.PathCache(g, placement="word_boundary"),
        "all": O.PathCache(g, placement="all"),
        "strict": O.PathCache(g, placement="word_boundary", escape=False),
        "shuffled": O.PathCache(g, placement="word_boundary", lexicon=res["shuffled_lexicon"]),
    }
    return {"root": root, "res": res, "graphs": graphs, "g": g, "caches": caches}


def runtime(fx, variant="word_boundary", **kw):
    b = fx["graphs"][variant]
    args = dict(WIDE)
    args.update(kw)
    return KT.LexlatK2Runtime(hlg=b["hlg"], stats=b["stats"], resources=fx["res"]["npz"], **args)


def bed_cfg(**kw):
    base = dict(n_phones=O.N_PHONES, sil_id=O.SIL_ID, topology="blankfree",
                recognizer_stride=O.RECOGNIZER_STRIDE)
    base.update(kw)
    return LatticeConfig(**base)


def random_log_q(b, t, seed):
    gen = torch.Generator().manual_seed(seed)
    return torch.log_softmax(torch.randn(b, t, O.N_PHONES, generator=gen) * 1.5, dim=-1)


def oracle_z_hlg(log_q, lens, tau, cache):
    """Per-utterance brute-force log Z_HLG and its gradient w.r.t. ``log_q`` (float64 autograd)."""
    e = log_q.detach().to(torch.float64).clone().requires_grad_(True)
    z = torch.stack([O.brute_force(e[i, :int(lens[i])], tau, cache) for i in range(e.shape[0])])
    return z, e


def closed_form_z_h(e, lens, tau):
    """log Z_H at min_frames 1: H accepts every frame string once, so it is sum_t logsumexp."""
    return torch.stack([torch.logsumexp(e[i, :int(lens[i])] / tau, dim=-1).sum()
                        for i in range(e.shape[0])])


# ================================================================================================
# T1.18  H topology and Z_H
# ================================================================================================
def test_t118_emission_min_frames():
    assert K.emission_min_frames(2, 3) == 1
    assert K.emission_min_frames(2, 1) == 2
    assert K.emission_min_frames(3, 3) == 1
    assert K.emission_min_frames(4, 3) == 2


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t118_log_z_h_closed_form_unsorted(fx, tau):
    rt = runtime(fx)
    lens = torch.tensor([4, 2, 3])
    log_q = random_log_q(3, 4, seed=1).requires_grad_(True)
    z = rt.log_z_h(rt.dense(log_q, tau), lens)
    e = log_q.detach().to(torch.float64)
    want = closed_form_z_h(e, lens, tau)
    assert z.detach().cpu().tolist() == pytest.approx(want.tolist(), rel=1e-6, abs=1e-6)
    z.sum().backward()
    grad_want = torch.zeros_like(e)
    for i, n in enumerate(lens.tolist()):
        grad_want[i, :n] = torch.softmax(e[i, :n] / tau, dim=-1) / tau
    assert torch.allclose(log_q.grad.to(torch.float64), grad_want, atol=1e-6, rtol=0)


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t118_min_frames_2_brute_force(fx, tau, tmp_path):
    """H with min_frames 2 accepts exactly the frame strings whose runs are all >= 2 frames."""
    import itertools

    stats = json.load(open(fx["graphs"]["word_boundary"]["stats"]))
    stats["min_frames"] = 2
    path = tmp_path / "build_min2.json"
    path.write_text(json.dumps(stats))
    rt = KT.LexlatK2Runtime(hlg=fx["graphs"]["word_boundary"]["hlg"], stats=str(path),
                            resources=fx["res"]["npz"], **WIDE)
    lens = torch.tensor([4, 2, 3])
    log_q = random_log_q(3, 4, seed=2)
    z = rt.log_z_h(rt.dense(log_q, tau), lens)
    e = log_q.to(torch.float64)
    want = []
    for i, n in enumerate(lens.tolist()):
        vals = []
        for a in itertools.product(range(O.N_PHONES), repeat=n):
            runs = [len(list(g)) for _k, g in itertools.groupby(a)]
            if min(runs) >= 2:
                vals.append(sum(float(e[i, t, a[t]]) for t in range(n)) / tau)
        want.append(float(torch.logsumexp(torch.tensor(vals, dtype=torch.float64), 0)))
    assert z.detach().tolist() == pytest.approx(want, rel=1e-6, abs=1e-6)


# ================================================================================================
# T1.19  HLG total vs the brute-force word-graph score
# ================================================================================================
def _check_variant(fx, variant, tau, seed=3):
    rt = runtime(fx, variant)
    lens = torch.tensor([3, 5, 4])  # T in {3, 4, 5}, unsorted
    log_q = random_log_q(3, 5, seed=seed).requires_grad_(True)
    z, _chunks = rt.log_z_hlg(rt.dense(log_q, tau), lens, tau)
    z_or, e = oracle_z_hlg(log_q, lens, tau, fx["caches"][variant])
    assert z.detach().tolist() == pytest.approx(z_or.detach().tolist(), rel=TOL), variant
    z.sum().backward()
    z_or.sum().backward()
    assert torch.allclose(log_q.grad.to(torch.float64), e.grad, atol=TOL, rtol=0), (
        variant, float((log_q.grad.to(torch.float64) - e.grad).abs().max()))


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t119_hlg_total_and_gradient_word_boundary(fx, tau):
    """(a) + (b) on the default arm's graph (escape, word_boundary back-off loops)."""
    _check_variant(fx, "word_boundary", tau)


@pytest.mark.parametrize("tau", [1.0, 2.0])
@pytest.mark.parametrize("variant", [
    "strict",
    "shuffled",
    pytest.param("all", marks=pytest.mark.xfail(strict=True, reason=S4_REASON)),
])
def test_t119_variants_match_their_oracles(fx, variant, tau):
    """(c) the strict, shuffled and 'all'-placement graphs against their own oracles."""
    _check_variant(fx, variant, tau)


@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t119_all_ge_word_boundary(fx, tau):
    """(c) the 'all' placement only ADDS routes: its total is >= word_boundary's, in the graph and
    in the oracle."""
    lens = torch.tensor([3, 5, 4])
    log_q = random_log_q(3, 5, seed=3)
    z = {}
    for v in ("word_boundary", "all"):
        rt = runtime(fx, v)
        z[v] = rt.log_z_hlg(rt.dense(log_q, tau), lens, tau)[0].detach()
    assert bool(torch.all(z["all"] >= z["word_boundary"] - 1e-6)), z
    zo = {v: oracle_z_hlg(log_q, lens, tau, fx["caches"][v])[0].detach()
          for v in ("word_boundary", "all")}
    assert bool(torch.all(zo["all"] >= zo["word_boundary"])), zo
    print(f"[record] T1.19c tau={tau}: graph Z_all - Z_wb = "
          f"{(z['all'] - z['word_boundary']).tolist()}, oracle Z_all - Z_wb = "
          f"{(zo['all'] - zo['word_boundary']).tolist()}")


def test_t119d_record_escape_multiplicity_and_sign(fx):
    """(d) S5: max_y log W_tau(y) for tau in {1, 2, 8}; the mass of parses with two adjacent escape
    spans; L_lex at a posterior peaked on the argmax string (the runtime's own number)."""
    cache = fx["caches"]["word_boundary"]
    ys = O.token_strings(5)
    rec = {}
    for tau in (1.0, 2.0, 8.0):
        lw = {y: O.log_w(cache(y), tau) for y in ys}
        y_max = max(lw, key=lw.get)
        rec[tau] = (lw[y_max], y_max)
        assert all(math.isfinite(v) for v in lw.values())  # escape: every string has a parse
        print(f"[record] T1.19d tau={tau}: max_y log W_tau(y) = {lw[y_max]:.6f} at y = "
              f"{[O.PHONES[k] for k in y_max]} (over {len(ys)} strings, |y| <= 5); strings with "
              f"log W_tau > 0: {sum(v > 0 for v in lw.values())}")
    shares = []
    for y in ys:
        paths = cache(y)
        w = O.log_w(paths, 1.0)
        wa = O.log_w(paths, 1.0, weight=lambda p: 1.0 if p.adjacent_escape else 0.0)
        shares.append((math.exp(wa - w) if math.isfinite(wa) else 0.0, y))
    share_max, y_share = max(shares)
    assert 0.0 <= share_max <= 1.0
    log_q = random_log_q(1, 5, seed=3)
    e = log_q.to(torch.float64)
    z = O.brute_force(e[0], 1.0, cache)
    za = O.brute_force(e[0], 1.0, cache, weight=lambda p: 1.0 if p.adjacent_escape else 0.0)
    print(f"[record] T1.19d tau=1: adjacent-escape share of W_1(y): max {share_max:.4f} at "
          f"{[O.PHONES[k] for k in y_share]}, median {statistics.median(s for s, _ in shares):.4f}; "
          f"share of Z_HLG on a random T=5 posterior: {math.exp(float(za - z)):.4f}")
    # the runtime's L_lex at a posterior peaked on the argmax string of each tau
    for tau in (1.0, 2.0, 8.0):
        y = rec[tau][1]
        t = len(y)
        lq = torch.full((1, t, O.N_PHONES), math.log(1e-30), dtype=torch.float32)
        for i, k in enumerate(y):
            lq[0, i, k] = 0.0
        rt = runtime(fx)
        dense = rt.dense(lq, tau)
        l_lex = float(rt.log_z_hlg(dense, torch.tensor([t]), tau)[0][0] - rt.log_z_h(dense, torch.tensor([t]))[0])
        assert math.isfinite(l_lex)
        print(f"[record] T1.19d tau={tau}: runtime L_lex at a posterior peaked on that string = "
              f"{l_lex:.6f} (oracle log W_tau = {rec[tau][0]:.6f}); L_lex > 0: {l_lex > 0}")


def test_t119e_record_overcount_vs_proper_lm(fx):
    """(e) the graph's W_1(y) against a single-route proper-LM W (exact back-off on the word
    history), per token, over every y with |y| <= 5; split off the trailing back-off moves."""
    cache = fx["caches"]["word_boundary"]
    scorer = O.ArpaScorer(fx["res"]["arpa"])
    over, trail = [], []
    for y in O.token_strings(5):
        paths = cache(y)
        lw = O.log_w(paths, 1.0)
        lw_no_trail = O.log_w(paths, 1.0, weight=lambda p: 1.0 if p.n_final_backoff == 0 else 0.0)
        proper = O.proper_paths(y, scorer)
        lp = O.log_w([O.Path(s, 0, 0, False, 0) for s in proper], 1.0)
        assert lw >= lp - 1e-9, (y, lw, lp)  # the graph's routes contain the proper route
        over.append((lw - lp) / len(y))
        trail.append((lw - lw_no_trail) / len(y))
    over_sorted = sorted(over)
    p95 = over_sorted[int(0.95 * (len(over_sorted) - 1))]
    print(f"[record] T1.19e over-count per token (log W_graph - log W_proper) / |y| over "
          f"{len(over)} strings: median {statistics.median(over):.4f}, p95 {p95:.4f}, "
          f"max {max(over):.4f}, min {min(over):.2e}; of which trailing back-off moves before "
          f"acceptance: median {statistics.median(trail):.4f}, max {max(trail):.4f} nats/token")


# ================================================================================================
# T1.20  the runtime step, empty lattices, bed checks
# ================================================================================================
@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t120_step_term_gradient_monitors(fx, tau):
    rt = runtime(fx)
    lens = torch.tensor([4, 3, 5])
    retained = torch.tensor([12.0, 9.0, 15.0])
    keep = torch.tensor([1.0, 1.0, 0.0])
    log_q = random_log_q(3, 5, seed=4).requires_grad_(True)
    epoch = rt.spec.onset
    term, mon = rt.step(log_q, feat_lens=lens, retained=retained, keep=keep, epoch=epoch,
                        temperature=tau, cfg=bed_cfg())
    cache = fx["caches"]["word_boundary"]
    z_hlg, e = oracle_z_hlg(log_q, lens, tau, cache)
    z_h = closed_form_z_h(e, lens, tau)
    l_lex = z_hlg - z_h
    scored = [0, 1]
    want = sum(-l_lex[i] / float(retained[i]) for i in scored) / float(keep.sum())
    assert float(term) == pytest.approx(float(want), rel=TOL)
    term.backward()
    want.backward()
    assert torch.allclose(log_q.grad.to(torch.float64), e.grad, atol=TOL, rtol=0), float(
        (log_q.grad.to(torch.float64) - e.grad).abs().max())
    assert float(mon["lexlat_k2_term_mean"]) == pytest.approx(
        sum(float(l_lex[i]) / float(retained[i]) for i in scored) / len(scored), rel=TOL)
    ew, ee = [], []
    with torch.no_grad():
        ed = e.detach()
        for i in scored:
            n = int(lens[i])
            ew.append(math.exp(float(O.brute_force(ed[i, :n], tau, cache, weight=lambda p: p.n_words)
                                     - z_hlg[i])))
            ee.append(math.exp(float(O.brute_force(ed[i, :n], tau, cache, weight=lambda p: p.n_unk)
                                     - z_hlg[i])))
    assert float(mon["lexlat_k2_expected_words"]) == pytest.approx(sum(ew) / 2, rel=TOL)
    assert float(mon["lexlat_k2_expected_escape_words"]) == pytest.approx(sum(ee) / 2, rel=TOL)
    assert float(mon["lexlat_k2_n_empty"]) == 0.0
    assert float(mon["lexlat_k2_lam"]) == pytest.approx(lexlat_lambda(epoch, onset=8, ramp=3))


def _forced_b_batch():
    """Row 0 forces the frame string B B B (y = "B": no strict parse); rows 1-2 are random."""
    log_q = random_log_q(3, 4, seed=5)
    log_q[0] = -math.inf
    log_q[0, :, O.PHONE2ID["B"]] = 0.0
    return log_q, torch.tensor([3, 4, 2]), torch.tensor([9.0, 12.0, 6.0]), torch.ones(3)


def test_t120_empty_lattice_counted_and_excluded(fx, tmp_path):
    # the wide operating point, so the scored rows can be held to the unpruned oracle
    rt = runtime(fx, "strict", empty_raise_frac=0.5, abort_dir=str(tmp_path))
    log_q, lens, retained, keep = _forced_b_batch()
    log_q.requires_grad_(True)
    term, mon = rt.step(log_q, feat_lens=lens, retained=retained, keep=keep, epoch=rt.spec.onset,
                        temperature=1.0, cfg=bed_cfg())
    assert float(mon["lexlat_k2_n_empty"]) == 1.0
    assert float(mon["lexlat_k2_empty_frac"]) == pytest.approx(1.0 / 3.0)
    cache = fx["caches"]["strict"]
    z_hlg, e = oracle_z_hlg(log_q, lens, 1.0, cache)
    assert not math.isfinite(float(z_hlg[0]))  # the oracle agrees: "B" has no strict parse
    z_h = closed_form_z_h(e[1:], lens[1:], 1.0)
    num = sum(-(z_hlg[i] - z_h[i - 1]) / float(retained[i]) for i in (1, 2))
    want = num / float(keep.sum())  # the denominator stays keep.sum() = 3
    assert float(term) == pytest.approx(float(want), rel=TOL)
    term.backward()
    assert bool(torch.isfinite(log_q.grad).all())
    assert bool(torch.all(log_q.grad[0] == 0))  # the empty row sends no gradient
    assert not os.path.exists(os.path.join(str(tmp_path), KT.ABORT_MARKER))


def test_t120_empty_above_limit_raises_and_writes_marker(fx, tmp_path):
    abort_dir = tmp_path / "arm"
    abort_dir.mkdir()
    # NEVER create learning_rates here: that path calls os._exit(0)
    assert not (abort_dir / KT.LEARNING_RATES_FILE).exists()
    rt = runtime(fx, "strict", abort_dir=str(abort_dir))
    assert rt.spec.empty_raise_frac == pytest.approx(0.10)
    log_q, lens, retained, keep = _forced_b_batch()
    with pytest.raises(RuntimeError, match="EMPTY"):
        rt.step(log_q, feat_lens=lens, retained=retained, keep=keep, epoch=rt.spec.onset,
                temperature=1.0, cfg=bed_cfg(), global_step=7)
    marker = json.load(open(abort_dir / KT.ABORT_MARKER))
    assert marker["n_empty"] == 1 and marker["batch_size"] == 3 and marker["step"] == 7


def test_t120_check_bed_accepts_the_bed(fx):
    rt = runtime(fx, expected_build={"theta": 0.0, "backoff_loops": "word_boundary",
                                     "escape": True, "sil_prob": O.SIL_PROB})
    rt.check_bed(bed_cfg())


@pytest.mark.parametrize("cfg_kw", [dict(d_min=3), dict(recognizer_stride=1), dict(topology="ctc")],
                         ids=["d_min3", "stride1", "ctc"])
def test_t120_check_bed_refuses(fx, cfg_kw):
    with pytest.raises(AssertionError):
        runtime(fx).check_bed(bed_cfg(**cfg_kw))


def test_t120_check_bed_refuses_determinized(fx, tmp_path):
    stats = json.load(open(fx["graphs"]["word_boundary"]["stats"]))
    stats["determinized"] = True
    path = tmp_path / "build_det.json"
    path.write_text(json.dumps(stats))
    rt = KT.LexlatK2Runtime(hlg=fx["graphs"]["word_boundary"]["hlg"], stats=str(path),
                            resources=fx["res"]["npz"], **WIDE)
    with pytest.raises(AssertionError, match="DETERMINIZED"):
        rt.check_bed(bed_cfg())


def test_t120_check_expected_build_refuses_theta(fx):
    rt = runtime(fx, expected_build={"theta": 5.0})
    with pytest.raises(AssertionError, match="theta"):
        rt.check_expected_build()


@pytest.mark.parametrize("onset,ramp", [(8, 3), (5, 3)])
def test_t120_lam_and_active_follow_lexlat_lambda(fx, onset, ramp):
    rt = runtime(fx, onset=onset, ramp=ramp)
    for e in range(1, 21):
        lam = lexlat_lambda(e, onset=onset, ramp=ramp, full=1.0)
        assert rt.lam(e) == lam
        assert rt.active(e) == (lam > 0.0)


# ================================================================================================
# T1.21  chunking and segment order (S6)
# ================================================================================================
@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t121_chunking_and_segment_order(fx, tau):
    lens = torch.tensor([2, 5, 3, 4])
    base = random_log_q(4, 5, seed=6)
    results = {}
    layouts = {"as_is": [0, 1, 2, 3], "permuted": [2, 0, 3, 1],
               "sorted": sorted(range(4), key=lambda i: -int(lens[i]))}
    for name, perm in layouts.items():
        for chunk in (1, 2, 16):
            if name != "as_is" and chunk != 16:
                continue
            rt = runtime(fx, chunk_seqs=chunk)
            lq = base[perm].clone().requires_grad_(True)
            z, _ = rt.log_z_hlg(rt.dense(lq, tau), lens[perm], tau)
            z.sum().backward()
            inv = torch.argsort(torch.tensor(perm))
            results[(name, chunk)] = (z.detach()[inv].to(torch.float64),
                                      lq.grad[inv].to(torch.float64))
    ref_z, ref_g = results[("as_is", 16)]
    for key, (z, g) in results.items():
        assert torch.allclose(z, ref_z, atol=1e-9, rtol=0), (key, (z - ref_z).abs().max())
        assert torch.allclose(g, ref_g, atol=1e-7, rtol=0), (key, (g - ref_g).abs().max())
    z_or, _ = oracle_z_hlg(base, lens, tau, fx["caches"]["word_boundary"])
    assert ref_z.tolist() == pytest.approx(z_or.detach().tolist(), rel=TOL)


# ================================================================================================
# T1.22  epsilon removal keeps the log-semiring total (S4)
# ================================================================================================
def _lg_stages(fx, placement):
    """compile_hlg's stages up to LG, replayed on the fixture: (before, after) epsilon removal.

    ``before`` has the disambiguation tokens relabelled to 0 (so a linear token FSA can match it)
    but keeps its epsilon arcs, which ``k2.intersect(treat_epsilons_specially=True)`` follows;
    ``after`` is ``connect(remove_epsilon(before))`` -- the object compile_hlg composes with H.
    """
    lm = K.read_lm_tables(fx["res"]["npz"])
    n_phones, sil_id, n_words = int(lm["n_phones"]), int(lm["sil_id"]), int(lm["n_words"])
    prons = K.prons_from_trie(lm["child"], lm["word_start"], lm["word_id"])
    entries, max_disambig = K.add_lex_disambig(prons)
    esc = K.escape_prices(lm["escape_phone_log_prob"], n_phones=n_phones, sil_id=sil_id)
    L = K.lexicon_to_fst(entries, n_phones=n_phones, sil_id=sil_id, n_words=n_words,
                         max_disambig=max_disambig, sil_prob=O.SIL_PROB,
                         escape_word=int(lm["unk_word"]), escape_price=esc,
                         backoff_loops=placement)
    G = K.g_fsa(lm, drop_words=K.NON_EMITTABLE_WORDS)
    LG = k2.connect(k2.compose(k2.arc_sort(L), k2.arc_sort(G)))
    LG.labels[LG.labels >= K.first_token_disambig_id(n_phones)] = 0
    LG.__dict__["_properties"] = None
    before = LG
    after = k2.connect(k2.remove_epsilon(LG.clone()))
    return before, after


def _string_total(fsa, y):
    lin = k2.linear_fsa([int(t) + 1 for t in y])
    r = k2.top_sort(k2.connect(k2.intersect(k2.arc_sort(fsa), lin, treat_epsilons_specially=True)))
    if r.num_arcs == 0:
        return -math.inf
    return float(k2.create_fsa_vec([r]).get_tot_scores(log_semiring=True, use_double_scores=True)[0])


def _t122_rows(fx, placement):
    before, after = _lg_stages(fx, placement)
    cache = fx["caches"][placement]
    return [(y, _string_total(before, y), _string_total(after, y), O.log_w(cache(y), 1.0))
            for y in O.token_strings(5)]


@pytest.mark.parametrize("placement", ["word_boundary", "all"])
def test_t122_lg_before_epsilon_removal_matches_oracle(fx, placement):
    rows = _t122_rows(fx, placement)
    worst = max(abs(b - o) for _y, b, _a, o in rows)
    assert worst <= 1e-6, worst


@pytest.mark.parametrize("placement", [
    "word_boundary",
    pytest.param("all", marks=pytest.mark.xfail(strict=True, reason=S4_REASON)),
])
def test_t122_epsilon_removal_keeps_log_total(fx, placement):
    rows = _t122_rows(fx, placement)
    diffs = [(a - b, y) for y, b, a, _o in rows]
    worst, y_worst = min(diffs)
    n_bad = sum(abs(d) > 1e-6 for d, _y in diffs)
    print(f"[record] T1.22 {placement}: {len(rows)} strings; after - before: min {worst:.3e} at "
          f"{[O.PHONES[k] for k in y_worst]}, max {max(d for d, _ in diffs):.3e}; "
          f"{n_bad} strings off by > 1e-6")
    assert max(abs(a - o) for _y, _b, a, o in rows) <= 1e-6
    assert n_bad == 0


# ================================================================================================
# G0.V (SAE_i6_P0.md): CPU-vs-CUDA parity of log Z_HLG and log Z_H on the T1.19 fixture
# ================================================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("tau", [1.0, 2.0])
def test_t119_cpu_cuda_parity_log_z_hlg_and_log_z_h(fx, tau):
    """The same runtime, graph and emissions on the CPU and on CUDA: log Z_HLG and log Z_H agree to
    TOL (1e-5, relative).  Run on a gpu_48gb node (the k2 build's sm_86 SASS on the L40S)."""
    rt = runtime(fx)
    lens = torch.tensor([3, 5, 4])  # T1.19's batch: T in {3, 4, 5}, unsorted
    log_q = random_log_q(3, 5, seed=3)
    z = {}
    for device in ("cpu", "cuda"):
        dense = rt.dense(log_q.to(device), tau)
        z_hlg, _chunks = rt.log_z_hlg(dense, lens, tau)
        z_h = rt.log_z_h(dense, lens)
        assert z_hlg.device.type == device and z_h.device.type == device, (z_hlg.device, z_h.device)
        z[device] = (z_hlg.detach().cpu().tolist(), z_h.detach().cpu().tolist())
    assert z["cuda"][0] == pytest.approx(z["cpu"][0], rel=TOL), ("log_z_hlg", z)
    assert z["cuda"][1] == pytest.approx(z["cpu"][1], rel=TOL), ("log_z_h", z)
    print(f"[record] G0.V parity tau={tau}: log_z_hlg cpu {z['cpu'][0]} cuda {z['cuda'][0]}; "
          f"log_z_h cpu {z['cpu'][1]} cuda {z['cuda'][1]}")
