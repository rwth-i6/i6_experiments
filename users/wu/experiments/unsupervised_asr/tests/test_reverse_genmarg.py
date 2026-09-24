"""Tests of ``reverse_model/genmarg.py`` (the label-free L2-1 reads and selection) without running a
job: the selection rule on synthetic records, the bed helper, the cut options and the forward jobs'
resources."""

import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg as G
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf

TAGS = ["a", "b", "c"]


def _rec(value, shuffle=None, impossible=()):
    return {"dataset": "cv_holdout", "settings": {"shuffle_seed": shuffle},
            "summary": {"expected_nonsil_rate": 1.0, "n_impossible": len(impossible)},
            "per_utterance": {t: {"nll_tau1_per_frame": value, "impossible": t in impossible,
                                  "expected_nonsil_tokens": 10.0, "original_frames": 50} for t in TAGS}}


def _dec(rate):
    return {"dataset": "cv_holdout", "settings": {"shuffle_seed": None},
            "per_utterance": {t: {} for t in TAGS}, "emitted_nonsil_rate": {"pooled_hz": rate, "n": 3}}


def _inputs(null_values=(6.0, 6.05, 6.1, 6.02), rates=None):
    restarts = {pf.arm_name(s): _rec(5.0 + 0.01 * s) for s in range(1, 17)}
    rates = rates or {k: 10.0 for k in restarts}
    return dict(
        restarts=restarts,
        nulls={pf.arm_name(s, "null"): _rec(v, shuffle=0) for s, v in enumerate(null_values, 1)},
        nulls_real={pf.arm_name(s, "null"): _rec(7.0) for s in range(1, 5)},
        phi_c={pf.arm_name(s, "phic"): _rec(5.5) for s in (1, 2)},
        reruns={"em_s01": _rec(5.01 + 0.004), "em_s02": _rec(5.02)},
        restart_decodes={k: _dec(r) for k, r in rates.items()},
    )


def test_decide_signal_with_void_restart():
    kw = _inputs()
    kw["restart_decodes"]["em_s01"] = _dec(20.0)  # outside [5.80, 14.49]: VOID
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["void"]["em_s01"] and out["eligible"] == 15
    assert out["selected"] == "em_s02"
    assert out["best_null"] == "null_s01" and abs(out["null_range"] - 0.1) < 1e-12
    assert abs(out["identity_band"] - 0.004) < 1e-12 and abs(out["margin"] - 0.1) < 1e-12
    assert abs(out["gap"] - (6.0 - 5.02)) < 1e-12 and out["verdict"] == "SIGNAL"
    assert out["reference_reading"] == "BEYOND PRIVATE CODE"


def test_decide_no_signal_and_no_eligible():
    out = G.GenMargSelectionJob.decide(**_inputs(null_values=(5.02, 5.06, 5.07, 5.08)))
    assert out["selected"] == "em_s01" and out["verdict"] == "NO SIGNAL"
    kw = _inputs()
    kw["restart_decodes"] = {k: _dec(1.0) for k in kw["restarts"]}
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["selected"] is None and out["verdict"] == "NO ELIGIBLE RESTART"


def test_decide_refuses_a_real_null():
    kw = _inputs()
    kw["nulls"]["null_s01"] = _rec(6.0)  # nulls are gated on the permuted holdout
    with pytest.raises(AssertionError):
        G.GenMargSelectionJob.decide(**kw)


def test_decide_pairing_set_drops_impossible():
    kw = _inputs()
    kw["nulls"]["null_s02"] = _rec(6.05, shuffle=0, impossible=("c",))
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["pairing_utterances"] == 2 and out["utterances"] == 3


def test_select_utterances():
    tags = [f"t{i}" for i in range(20)]
    a = G.select_utterances(tags, 5)
    assert a == G.select_utterances(list(reversed(tags)), 5) and len(set(a)) == 5
    assert G.GenMargSampleJob.select(tags, None, 0) == sorted(tags)


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def test_reads_bed_and_forward_jobs():
    reads = pf.reads_bed(_data())
    bed = reads["bed"]
    assert list(bed["model_args"]) == list(G.BED_MODEL_ARG_KEYS)
    assert bed["model_args"]["temperature_schedule"] == list(G.READ_TEMPERATURE_SCHEDULE)
    assert "recognizer_checkpoint_path" not in bed["model_args"]
    jobs = G.genmarg_reads(tk.Path("/nonexistent/phi.pt"), "x", alias=None, **reads)["cv_holdout"]
    assert set(jobs) == {"sample", "marginal", "decode"}
    assert jobs["sample"].n is None
    assert jobs["marginal"].rqmt == {"gpu": 1, "cpu": 4, "mem": 32, "time": 2}


def test_cut_options_raise():
    reads = pf.reads_bed(_data())
    phi = tk.Path("/nonexistent/phi.pt")
    for bad in ({"gap": True}, {"report": True}, {"datasets": ("dev-other",)}):
        with pytest.raises(ValueError):
            G.genmarg_reads(phi, "x", alias=None, **reads, **bad)
    with pytest.raises(AssertionError):
        G.genmarg_reads("/nonexistent/phi.pt", "x", alias=None, **reads)


# ===================================================================================================
# Priority-2 test (test plan 2026-09-24, T2.10): genmarg's marginals against the brute-force oracle.
#
# The oracle (tests/lattice_oracle.py) enumerates the latent set of the bed's model, whose
# prior_history is built for the "ctc" topology (model/emc_model.py:367, pinned by
# test_model_lattice.py T1.6), so SIL runs may split into several SIL tokens: ``sil_split=True``.
# The prior of every latent is the model's float32 buffer walked by the oracle's own BOS-padded
# token history (never PhoneNgramPrior.per_token_log_probs, model/prior.py:246).
# ===================================================================================================

import math  # noqa: E402
import os  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import lattice_oracle as O  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.lattice import build_segment_table  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg_steps as GS  # noqa: E402

T210_TAGS, T210_S, T210_ORIG = ["u0", "u1", "u2"], [6, 5, 1], [11, 5, 4]


def _t2_10_model(tmp):
    rng = np.random.RandomState(0)
    n = 40
    prior = PhoneNgramPrior(
        np.log(np.ones(n) / n), np.log(rng.dirichlet(np.ones(n), size=n + 1)),
        np.log(rng.dirichlet(0.3 * np.ones(n), size=(n + 1) * (n + 1))), meta={"test": True})
    prior_path, eta_path = os.path.join(tmp, "prior.npz"), os.path.join(tmp, "eta.npz")
    prior.save(prior_path)
    np.savez_compressed(eta_path, tags=np.array(T210_TAGS), eta=rng.randn(3, 4).astype(np.float32))
    torch.manual_seed(0)
    return BM.SaeBlankfreeModelV1(
        temperature_schedule=[1.0], anchor_weight_schedule=0.0, lam_agg=0.1, count_ema_decay=0.99,
        band=25, prior_weight=1.0, prior_npz_path=prior_path, eta_table_path=eta_path,
        reverse_kwargs={"n_units": 20, "eta_dim": 4, "d_model": 16, "d_ff": 16},
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True)


def test_t2_10_generative_marginals_vs_oracle(tmp_path):
    """log_z[tau] = the lattice with log_q = 0 = the oracle (1e-10) at tau 1 and 2; expected_nonsil =
    the oracle's E[#non-SIL tokens] under the tau = 1 posterior; the uniform-q offset of each row is
    (T_rec / tau) log 40, i.e. -log Z at log_q = -log 40 (float64) = the zero-q value + the offset."""
    model = _t2_10_model(str(tmp_path))
    units = torch.randint(0, 20, (3, 6), generator=torch.Generator().manual_seed(4))
    lens = torch.tensor(T210_S)
    res = GS.generative_marginals(model, units, lens, T210_TAGS)
    rows = GS.marginal_rows(res, lens, T210_TAGS, original_lens=T210_ORIG)
    cfg = model.lattice_cfg
    topo = O.Topology(cfg.n_phones, cfg.sil_id, cfg.band, cfg.d_min, cfg.d_max, cfg.d_max_sil, cfg.recognizer_stride)
    eta = model.lookup_eta(T210_TAGS, "cpu")
    with torch.no_grad():
        seg = build_segment_table(model.reverse, units, eta).double()
    prior = model.prior_log_bi.double()
    assert res["t_rec"].tolist() == [math.ceil(s / 3) for s in T210_S] == [2, 2, 1]
    uniform = {}
    for tau in (1.0, 2.0):
        log_q = torch.full((3, 2, 40), -math.log(40), dtype=torch.float64)
        uniform[tau] = L.forward_log_z(log_q, seg, prior, res["t_rec"], lens, cfg, temperature=tau,
                                       prior_weight=1.0, history=model.prior_history, reduction="matmul")
    for b, (S, T) in enumerate(zip(T210_S, res["t_rec"].tolist())):
        enum = O.enumerate_utterance(T, S, topo, "trigram", s_cols=seg.shape[-1], sil_split=True)
        row = rows[b]
        if enum.n == 0:
            assert row["impossible"] and row["nll_tau1"] is None
            continue
        assert not row["impossible"]
        zero = torch.zeros(T, 40, dtype=torch.float64)
        for tau in (1.0, 2.0):
            ref = float(O.log_z(enum, zero, seg[b], prior, tau=tau, beta=1.0))
            got = float(res["log_z"][tau][b])
            print(f"T2.10 utt {b} tau {tau}: log Z genmarg {got!r} oracle {ref!r}")
            assert abs(got - ref) <= 1e-10 * max(1.0, abs(ref))
            offset = row["uniform_q_offset_tau1"] if tau == 1.0 else row["uniform_q_offset_tau2"]
            assert offset == T * math.log(40) / tau
            value = row["nll_tau1"] if tau == 1.0 else row["l_tau2"]
            assert abs(-float(uniform[tau][b]) - (value + offset)) <= 1e-10 * max(1.0, abs(value + offset))
        st = O.statistics(enum, zero, seg[b], prior, tau=1.0, beta=1.0, n_symbols=40)
        e_n = float(st["expected_nonsil"])
        assert abs(float(res["expected_nonsil"][b]) - e_n) <= 1e-10 * max(1.0, e_n)
        assert abs(row["expected_nonsil_rate_original_hz"] - 50.0 * e_n / T210_ORIG[b]) <= 1e-9
        assert row["free_energy_tau2"] == -2.0 * float(res["log_z"][2.0][b])
