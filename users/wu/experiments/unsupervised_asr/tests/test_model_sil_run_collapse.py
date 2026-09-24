"""The ``sil_run_collapse`` model option and its arm ``ctrl_20_rc`` (SAE_i6_P0.md, G0.RC; the fix of the
SIL-run split pinned by test_model_lattice.py T1.6).

* config / arm: the option enters ``get_model``'s hashed arguments only when True, and ``ctrl_20_rc``
  is ``ctrl_20`` plus that one key;
* the band's cap on one token's run, and what enforcing run collapse on SIL does to long silences, on
  the DP at the bed's band and durations (W 25, d_min 2, D 25, D_sil 50, stride 3) with a 3-symbol
  inventory: the cap does not depend on the inventory size, and the tiny inventory keeps the
  independent feasibility check below exhaustive.

The model-level checks (history, train step) are in test_model_lattice.py (T1.6 rc) and
test_model_blankfree.py (T2.1 rc).  CPU, no job runs.
"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import pytest
import torch
from sisyphus import tk
from sisyphus.hash import sis_hash_helper

import lattice_oracle as O

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L
from i6_experiments.users.wu.experiments.unsupervised_asr.model import rate_term as RT
from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms, config, jobs


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")],
        train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")],
        dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")],
        dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"),
        dev_segments=p("cv.segments"),
        prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"),
        flat_checkpoint=p("flat_init.pt"),
    )


@pytest.fixture
def private_jobs(monkeypatch):
    """A private job registry, so jobs built here do not merge with other tests' jobs."""
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})


# ================================================================================================
# config and arm
# ================================================================================================
def test_config_key_only_when_true():
    base = config.build_train_config(**_data())
    off = config.build_train_config(**_data(), sil_run_collapse=False)
    on = config.build_train_config(**_data(), sil_run_collapse=True)
    assert "sil_run_collapse" not in jobs.get_model_args(base)
    assert "sil_run_collapse" not in jobs.get_model_args(off)
    assert sis_hash_helper(off) == sis_hash_helper(base)
    assert jobs.get_model_args(on)["sil_run_collapse"] is True
    assert sis_hash_helper(on) != sis_hash_helper(base)
    undone = copy.deepcopy(on)
    del jobs.get_model_args(undone)["sil_run_collapse"]
    assert sis_hash_helper(undone) == sis_hash_helper(base)


@pytest.mark.parametrize("n", [20, 60])
def test_ctrl_20_rc_is_ctrl_20_plus_the_option(private_jobs, n):
    ctrl = arms.ctrl_20(data=_data(), num_sub_epochs=n)
    rc = arms.ctrl_20_rc(data=_data(), num_sub_epochs=n)
    assert rc.name == ("ctrl_20_rc" if n == 20 else "ctrl_20_rc_x60")
    assert (rc.num_epochs, rc.keep_epochs) == (ctrl.num_epochs, ctrl.keep_epochs)
    assert arms.ARM_PRESETS["ctrl_20_rc"] is arms.ctrl_20_rc and "ctrl_20_rc" in arms.__all__
    a_rc, a_ctrl = jobs.get_model_args(rc.returnn_config), jobs.get_model_args(ctrl.returnn_config)
    assert set(a_rc) - set(a_ctrl) == {"sil_run_collapse"} and set(a_ctrl) <= set(a_rc)
    assert a_rc["sil_run_collapse"] is True
    assert sis_hash_helper(rc.returnn_config.config) == sis_hash_helper(ctrl.returnn_config.config)
    undone = copy.deepcopy(rc.returnn_config)
    del jobs.get_model_args(undone)["sil_run_collapse"]
    assert sis_hash_helper(undone) == sis_hash_helper(ctrl.returnn_config)
    j_ctrl = jobs.train_arm(**ctrl.job_kwargs())
    j_rc = jobs.train_arm(**rc.job_kwargs())
    assert j_rc is not j_ctrl and j_rc._sis_id() != j_ctrl._sis_id()
    assert j_rc.rqmt == j_ctrl.rqmt and sorted(j_rc.out_checkpoints) == sorted(j_ctrl.out_checkpoints)


# ================================================================================================
# the DP at the bed's band / durations: the per-token run cap and long silences under run collapse
# ================================================================================================
K, SIL = 3, 2
CFG = L.LatticeConfig(n_phones=K, sil_id=SIL, band=25, d_min=2, d_max=25, d_max_sil=50,
                      topology="blankfree", recognizer_stride=3)
TOPO = O.Topology(K, SIL, 25, 2, 25, 50, 3)
#: the history the default (banked) model hands the train step, and the one the option hands it
H_DEFAULT = L.build_prior_history(replace(CFG, topology="ctc"), "trigram")
H_RC = L.build_prior_history(CFG, "trigram")


def _dp(log_q, S, hist, *, seg=None, prior=None, tau=1.0):
    T = log_q.shape[0]
    seg = torch.zeros(1, K, 50, S + 1, dtype=torch.float64) if seg is None else seg.unsqueeze(0)
    prior = torch.zeros((K + 1) ** 2, K, dtype=torch.float64) if prior is None else prior
    return L.lattice_forward_backward(
        log_q.unsqueeze(0), seg, prior, torch.tensor([T]), torch.tensor([S]), CFG,
        temperature=tau, prior_weight=1.0, history=hist, reduction="matmul", checkpoint=0)


def _forced(path, other):
    """log_q that is 0 on ``path`` and ``other`` everywhere else."""
    lq = torch.full((len(path), K), float(other), dtype=torch.float64)
    lq[torch.arange(len(path)), torch.tensor(path)] = 0.0
    return lq


def _feasible_run_collapse(path, S) -> bool:
    """Independent of the DP: has the run-collapse reading of ``path`` an admissible segmentation of
    ``S`` unit frames (the oracle's band, ``lattice_oracle._in_band``)?  Reachable token ends, token by
    token."""
    tokens, emits = O.run_collapse(path)
    reach = {0}
    for u, k in enumerate(tokens):
        reach = {s0 + d for s0 in reach for d in range(TOPO.d_min, TOPO.d_max_of(k) + 1)
                 if s0 + d <= S and O._in_band(s0 + d, u, emits, len(path), TOPO, "code")}
    return S in reach


def _any_s_feasible(path) -> bool:
    T = len(path)
    return any(_feasible_run_collapse(path, S) for S in (3 * T - 2, 3 * T - 1, 3 * T))


def test_band_caps_one_token_run():
    """Band arithmetic: every frame of a run keeps the token's end s within W of 3t, so one token
    spans at most floor(2W / 3) + 1 = 17 recognizer frames (~1.02 s) inside an utterance, whatever
    its type; a trailing run at most 10 (s = S ~ 3T at the end).  The cap is the band's, not D_sil's."""
    alt = (0, 1) * 3
    for R in range(10, 24):
        assert _any_s_feasible(alt + (SIL,) * R + alt[::-1]) == (R <= 17), R
        assert _any_s_feasible(alt + (0,) * R + (1,) + alt[::-1][1:]) == (R <= 17), R  # a phone run: same cap
        assert _any_s_feasible((SIL,) * R + alt) == (R <= 17), R  # leading
        assert _any_s_feasible(alt + (SIL,) * R) == (R <= 10), R  # trailing


@pytest.mark.parametrize("R", [16, 17, 18, 20])
def test_long_sil_run_has_no_reading_under_run_collapse(R):
    """The DP agrees with the independent check: a SIL run longer than the cap is an infeasible path
    under the option (its mass needs a deviation from the forced path), while the default reads it as
    several SIL tokens.  Scores are 0 on the path and -500 off it (within float64's _logmm floor)."""
    alt = (0, 1) * 3
    path = alt + (SIL,) * R + alt[::-1]
    T = len(path)
    S = 3 * T
    lq = _forced(path, -500.0)
    rc, default = _dp(lq, S, H_RC), _dp(lq, S, H_DEFAULT)
    feasible = _feasible_run_collapse(path, S)
    assert feasible == (R <= 17)
    assert (float(rc.log_z[0]) > -100.0) == feasible, float(rc.log_z[0])
    assert float(default.log_z[0]) > -100.0
    assert not bool(rc.z_zero[0]) and not bool(default.z_zero[0])


@pytest.mark.parametrize("S,n_nonsil", [(20, 0), (28, 0), (29, 1), (50, 1), (75, 1), (100, 2), (150, 3)])
def test_all_silence_is_broken_by_non_sil_tokens(S, n_nonsil):
    """An utterance the recognizer calls silence throughout (SIL 0, every other symbol -30 nats):
    under the option one SIL token covers at most d <= W + 3 = 28 units at the start, so a longer
    silence is read with non-SIL tokens inserted (about one per 50 units); the default reads it as
    several SIL tokens and inserts none.  Z > 0 either way: the utterance is never a z_zero row."""
    T = math.ceil(S / 3)
    lq = _forced((SIL,) * T, -30.0)
    rc, default = _dp(lq, S, H_RC), _dp(lq, S, H_DEFAULT)
    e_rc = float(RT.expected_nonsil_tokens(rc, CFG)[0])
    e_def = float(RT.expected_nonsil_tokens(default, CFG)[0])
    print(f"all-SIL S={S}: log Z rc {float(rc.log_z[0]):.3f} default {float(default.log_z[0]):.3f}; "
          f"E[non-SIL] rc {e_rc:.4f} default {e_def:.2e}; E[tokens] rc {float(rc.seg_post.sum()):.3f} "
          f"default {float(default.seg_post.sum()):.3f}")
    assert abs(e_rc - n_nonsil) <= 1e-6
    assert e_def <= 1e-6
    assert not bool(rc.z_zero[0]) and not bool(default.z_zero[0])
    assert float(rc.log_z[0]) < float(default.log_z[0])


def test_z_zero_rows_unchanged_and_log_z_a_subset():
    """Random inputs, S = 1..60: the option leaves the z_zero rows (the rows the step masks out of
    l_tau and rate) exactly the default's -- S = 1 only -- and never raises log Z (a subset of the
    latents)."""
    g = torch.Generator().manual_seed(0)
    zz = []
    for S in range(1, 61):
        T = math.ceil(S / 3)
        lq = torch.log_softmax(2.0 * torch.randn(T, K, generator=g, dtype=torch.float64), -1)
        seg = torch.randn(K, 50, S + 1, generator=g, dtype=torch.float64) - 2.0
        prior = torch.log_softmax(torch.randn((K + 1) ** 2, K, generator=g, dtype=torch.float64), -1)
        rc = _dp(lq, S, H_RC, seg=seg, prior=prior, tau=2.0)
        default = _dp(lq, S, H_DEFAULT, seg=seg, prior=prior, tau=2.0)
        assert bool(rc.z_zero[0]) == bool(default.z_zero[0]), S
        if bool(rc.z_zero[0]):
            zz.append(S)
        else:
            assert float(rc.log_z[0]) <= float(default.log_z[0]) + 1e-12, S
    assert zz == [1]
