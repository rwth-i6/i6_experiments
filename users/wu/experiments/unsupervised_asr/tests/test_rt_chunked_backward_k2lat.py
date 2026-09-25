"""The per-chunk k2 backward (``reverse_model/rt_chunked_backward.py``) against the HELD path at P0
``k2lat_20_ma3000``'s k2 settings, on the T1.19 fixture graph (``test_model_lexlat_k2.fx``).

``tests/test_rt_chunked_backward.py`` pins exactness at the fixture's wide operating point (max_active
10000, beams 1000), where the stability read returns ``nan`` (its reference rung 10000 is not above the
rung) and a chunk of 16 covers the whole 5-sequence batch.  Here the runtime carries k2lat's settings
(the job's own print of its spec, ``ReturnnTrainingJob.jcKXbLMDk4hl/log.run.1``): rung 3000, search beam
20, output beam 8, min active states 30, on-set 8, ramp 3, full lam 1, reference rung 10000 (so the
stability read RUNS and is emitted), 16 sequences per ``intersect_dense_pruned`` call, and a
40-sequence batch (chunks of 16, 16, 8); tau 2.0, the arm's temperature from sub-epoch 4 on.
Pinned as in the sibling file: log Z_HLG to 1e-9, the term, the total and every monitor EQUAL (the
timing monitors excepted), the gradients to ``log_q`` and to a parameter upstream of it to 1e-7.
Deviations are printed with a ``[record]`` label (run with ``-s``).
"""

from __future__ import annotations

import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

from test_model_lexlat_k2 import fx, random_log_q, runtime  # noqa: E402,F401
from test_rt_chunked_backward import G_TOL, Z_TOL, _backend, _compare, _path  # noqa: E402,F401

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2 as K  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2_train as KT  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import rt_chunked_backward as RC  # noqa: E402

pytestmark = pytest.mark.k2

#: k2lat_20_ma3000's k2 runtime spec, as the live job prints it (log.run.1: "k2 lexicon term: {...}")
K2LAT = dict(max_active=3000, search_beam=20.0, output_beam=8.0, min_active_states=30, onset=8, ramp=3,
             full_lam=1.0, stability_reference_max_active=10000, chunk_seqs=16)
#: k2lat's temperature at every k2 sub-epoch (its schedule is 2.0 from sub-epoch 4 on)
TAU = 2.0
#: the batch: more sequences than one k2lat chunk, so the per-chunk backward runs over 3 chunks
N_SEQS, N_FRAMES = 40, 6


def test_k2lat_settings_are_the_package_defaults():
    """The spec entries the k2lat config does not state are the runtime's defaults (the job's print)."""
    assert KT.STABILITY_REFERENCE_MAX_ACTIVE == K2LAT["stability_reference_max_active"]
    assert K.CHUNK_SEQS == K2LAT["chunk_seqs"]
    assert (K.SEARCH_BEAM, K.OUTPUT_BEAM, K.MIN_ACTIVE_STATES) == (
        K2LAT["search_beam"], K2LAT["output_beam"], K2LAT["min_active_states"])
    assert len(K.chunk_bounds(N_SEQS, K2LAT["chunk_seqs"])) == 3


def _batch():
    gen = torch.Generator().manual_seed(23)
    lens = torch.randint(2, N_FRAMES + 1, (N_SEQS,), generator=gen)
    retained = (3 * lens + torch.randint(0, 3, (N_SEQS,), generator=gen)).to(torch.float64)
    keep = torch.ones(N_SEQS, dtype=torch.float64)
    keep[[3, 17, 33]] = 0.0  # a dropped row in each chunk
    return random_log_q(N_SEQS, N_FRAMES, seed=29), lens, retained, keep


def _rt(fx, tmp_path, **kw):
    return runtime(fx, abort_dir=str(tmp_path), **{**K2LAT, **kw})


@pytest.mark.parametrize("epoch", [8, 9, 10, 11])
def test_chunked_backward_equals_held_k2lat(fx, tmp_path, epoch):
    base, lens, retained, keep = _batch()
    held_rt = _rt(fx, tmp_path)
    mine_rt = RC.install_chunked_backward(_rt(fx, tmp_path))
    assert mine_rt.spec.chunk_seqs == 16 and mine_rt.spec.max_active == 3000

    def _held_leg_forbidden(*a, **k):
        raise AssertionError("the per-chunk runtime called the held log_z_hlg")

    mine_rt.log_z_hlg = _held_leg_forbidden  # the comparison is not vacuous: the new leg runs
    held = _path(held_rt, base, lens, retained, keep, epoch=epoch, tau=TAU)
    mine = _path(mine_rt, base, lens, retained, keep, epoch=epoch, tau=TAU)
    # not vacuous: the gradients are far above the tolerance (smaller than in the sibling file, whose
    # 5-sequence batch divides the term by fewer kept rows)
    g_mag = (float(held["g_log_q"].abs().max()), float(held["g_bias"].abs().max()))
    assert min(g_mag) > 100 * G_TOL, g_mag
    assert held["mon"]["lexlat_k2_lam"] == pytest.approx(min(1.0, (epoch - 7) / 3.0))
    assert held["mon"]["lexlat_k2_n_empty"] == 0.0
    # the reference rung 10000 is above the rung 3000: the stability read runs, is finite, and is
    # emitted by both paths (so _compare holds its value equal)
    stab = held["mon"]["lexlat_k2_stability"]
    assert stab == stab and mine["mon"]["lexlat_k2_stability"] == stab
    mon_dev, dev = _compare(held, mine, ("k2lat", epoch))

    # log Z_HLG at the k2lat rung and beams, held leg vs per-chunk leg, on the same dense tensor
    log_q = torch.log_softmax(base.to(torch.float64), dim=-1)
    plain = _rt(fx, tmp_path)
    dense = plain.dense(log_q, TAU)
    z_held = plain.log_z_hlg(dense, lens, TAU)[0].detach()
    z_mine = RC.install_chunked_backward(_rt(fx, tmp_path)).log_z_hlg_chunked_backward(
        dense.detach(), lens, TAU, upstream=None)[0]
    z_dev = float((z_held - z_mine).abs().max())
    assert torch.allclose(z_mine, z_held, atol=Z_TOL, rtol=0), z_dev
    # informative: how far the k2lat rung / beams move log Z from the wide point on this fixture
    wide = runtime(fx, chunk_seqs=16)
    z_wide = wide.log_z_hlg(wide.dense(log_q, TAU), lens, TAU)[0].detach()
    prune = float((z_wide - z_held).abs().max())
    print(f"[record] k2lat settings, epoch {epoch} (lam {held['mon']['lexlat_k2_lam']:.4f}), "
          f"{N_SEQS} seqs in chunks of 16: max |dlogZ| {z_dev:.3e}, max |dmonitor| {mon_dev:.3e}, "
          f"max |dgrad log_q| {dev['g_log_q']:.3e}, max |dgrad param| {dev['g_bias']:.3e}, "
          f"term {mine['term']!r} (held {held['term']!r}), stability {stab!r}, "
          f"max |logZ(k2lat) - logZ(wide)| {prune:.3e}, max |grad log_q|, |grad param| "
          f"{g_mag[0]:.3e}, {g_mag[1]:.3e}")


def test_chunked_backward_no_grad_pass_k2lat(fx, tmp_path):
    """RETURNN's dev pass (no_grad) at k2lat's settings: the same value and monitors, no backward."""
    base, lens, retained, keep = _batch()
    held = _path(_rt(fx, tmp_path), base, lens, retained, keep, epoch=11, tau=TAU, grad=False)
    mine = _path(RC.install_chunked_backward(_rt(fx, tmp_path)), base, lens, retained, keep, epoch=11,
                 tau=TAU, grad=False)
    mon_dev, _ = _compare(held, mine, "k2lat no_grad")
    print(f"[record] k2lat settings, no_grad pass: term {mine['term']!r} (held {held['term']!r}), "
          f"max |dmonitor| {mon_dev:.3e}")
