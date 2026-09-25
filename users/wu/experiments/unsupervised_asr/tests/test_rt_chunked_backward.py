"""The per-chunk k2 backward of the P1 rt arms (``reverse_model/rt_chunked_backward.py``) against the
HELD path (``model/lexlat_k2_train.LexlatK2Runtime.step``), T1.21-style (SAE_i6_P1.md, G1.M).

Both paths run on the T1.19 fixture graph (``test_model_lexlat_k2.fx``) through RETURNN's own
``mark_as_loss(scale = lam)`` / ``total_loss().backward()``.  Pinned, per chunk size (1, 2, 3, 16),
temperature and ramp sub-epoch: log Z_HLG to 1e-9, the term and every monitor EQUAL (the two timing /
memory monitors excepted), the gradient to ``log_q`` and to a parameter upstream of it to 1e-7; the
empty-lattice rule (kept and aborting), the no-grad (dev) pass; and one full blank-free
``train_step`` (theta and phi, every loss) with the fixture leg behind the model's recognizer.
Deviations are printed with a ``[record]`` label (run with ``-s``).
"""

from __future__ import annotations

import copy
import json
import math
import os

import pytest

k2 = pytest.importorskip("k2")
torch = pytest.importorskip("torch")

import returnn.frontend as rf  # noqa: E402
from returnn.frontend._backend import select_backend_torch  # noqa: E402

import k2_oracle as O  # noqa: E402
from test_model_blankfree import inputs  # noqa: E402,F401  (the blank-free bed fixture)
from test_model_lexlat_k2 import _forced_b_batch, bed_cfg, fx, random_log_q, runtime  # noqa: E402,F401

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lexlat_k2_train as KT  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import rt_chunked_backward as RC  # noqa: E402

pytestmark = pytest.mark.k2

#: the brief's tolerances (SAE_i6_P1.md, Task B Design): log Z 1e-9, gradients 1e-7
Z_TOL = 1e-9
G_TOL = 1e-7
#: monitors that time / measure the leg; the per-chunk path includes its backwards in them
TIMING = {"lexlat_k2_sec", "lexlat_k2_peak_reserved_gib"}
#: the G1.M memory monitors the per-chunk path ADDS (the held path has none of them); 0.0 on CPU
MEMORY = ("lexlat_k2_pre_peak_allocated_gib", "lexlat_k2_pre_peak_reserved_gib",
          "lexlat_k2_device_used_gib")
#: the rt arms' ramp: on-set 1, ramp 3 -> lam 1/3, 2/3, 1 at sub-epochs 1, 2, 3
ONSET, RAMP = 1, 3
#: the stability read's monitor: the per-chunk path OMITS it when the held read is non-finite (a
#: missing read), so RETURNN's non-finite stop cannot end the arm on it (rt_chunked_backward doc)
STABILITY = "lexlat_k2_stability"


def _held_keys(held_mon):
    """The held path's monitor keys the per-chunk path emits: all of them, less a non-finite stability
    read.  ``held_mon`` maps key -> float or tensor."""
    return [k for k, v in held_mon.items() if not (k == STABILITY and not math.isfinite(float(v)))]


@pytest.fixture(scope="module", autouse=True)
def _backend():
    select_backend_torch()


def _batch():
    lens = torch.tensor([2, 5, 3, 4, 5])
    retained = torch.tensor([6.0, 15.0, 9.0, 12.0, 14.0], dtype=torch.float64)
    keep = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    return random_log_q(5, 5, seed=11), lens, retained, keep


def _path(rt, base, lens, retained, keep, *, epoch, tau, device="cpu", variant_cfg=None, grad=True):
    """One step of ``rt`` through RETURNN's loss plumbing; log_q = log_softmax(base + bias) in float64
    (the train step hands the leg ``dp_log_q = log_q.double()``), ``bias`` a parameter."""
    bias = torch.nn.Parameter(torch.zeros(base.shape[-1], dtype=torch.float64, device=device))
    rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=epoch)
    ctx = rf.get_run_ctx()
    with torch.set_grad_enabled(grad):
        log_q = torch.log_softmax(base.to(device=device, dtype=torch.float64) + bias, dim=-1)
        if grad:
            log_q.retain_grad()
        term, mon = rt.step(log_q, feat_lens=lens, retained=retained.to(device), keep=keep.to(device),
                            epoch=epoch, temperature=tau, cfg=variant_cfg or bed_cfg(), global_step=3)
        ctx.mark_as_loss(term, "lexlat_k2", scale=rt.lam(epoch), use_normalized_loss=False)
        total = ctx.total_loss()
        total = total.raw_tensor if hasattr(total, "raw_tensor") else total
    out = {"term": float(term), "total": float(total), "dtype": term.dtype,
           "mon": {k: float(v) for k, v in mon.items()}}
    if grad:
        total.backward()
        out["g_log_q"] = log_q.grad.detach().cpu()
        out["g_bias"] = bias.grad.detach().cpu()
    return out


def _z_hlg(rt, base, lens, tau, device="cpu"):
    """The held and the per-chunk log Z_HLG on the same dense tensor."""
    log_q = torch.log_softmax(base.to(device=device, dtype=torch.float64), dim=-1)
    dense = rt.dense(log_q, tau)
    held = rt.log_z_hlg(dense, lens, tau)[0].detach().cpu()
    chunked_rt = RC.install_chunked_backward(copy.copy(rt))
    mine = chunked_rt.log_z_hlg_chunked_backward(dense.detach(), lens, tau, upstream=None)[0].cpu()
    return held, mine


def _compare(held, mine, label):
    assert mine["dtype"] == held["dtype"], label
    assert mine["term"] == held["term"], (label, mine["term"], held["term"])
    assert mine["total"] == held["total"], (label, mine["total"], held["total"])
    assert set(MEMORY) <= set(mine["mon"]) and not set(MEMORY) & set(held["mon"]), label
    shared = [k for k in mine["mon"] if k not in MEMORY]
    assert list(shared) == _held_keys(held["mon"]), label  # the column order RETURNN logs
    mon_dev = 0.0
    for key, value in held["mon"].items():
        if key in TIMING or key not in _held_keys(held["mon"]):
            continue
        other = mine["mon"][key]
        both_nan = math.isnan(value) and math.isnan(other)
        assert both_nan or other == value, (label, key, other, value)
        if not both_nan:
            mon_dev = max(mon_dev, abs(other - value))
    dev = {}
    for key in ("g_log_q", "g_bias"):
        if key in held:
            a, b = held[key].to(torch.float64), mine[key].to(torch.float64)
            assert bool(torch.isfinite(b).all()), (label, key)
            dev[key] = float((a - b).abs().max())
            assert torch.allclose(b, a, atol=G_TOL, rtol=0), (label, key, dev[key])
    return mon_dev, dev


# ================================================================================================
# the leg alone, against the held path
# ================================================================================================
@pytest.mark.parametrize("tau", [1.0, 2.0])
@pytest.mark.parametrize("epoch", [1, 2, 3])
@pytest.mark.parametrize("chunk", [1, 2, 3, 16])
def test_chunked_backward_equals_held(fx, chunk, epoch, tau):
    base, lens, retained, keep = _batch()
    held_rt = runtime(fx, chunk_seqs=chunk, onset=ONSET, ramp=RAMP)
    mine_rt = RC.install_chunked_backward(runtime(fx, chunk_seqs=chunk, onset=ONSET, ramp=RAMP))
    assert type(mine_rt) is RC.ChunkedBackwardLexlatK2Runtime

    def _held_leg_forbidden(*a, **k):
        raise AssertionError("the per-chunk runtime called the held log_z_hlg")

    mine_rt.log_z_hlg = _held_leg_forbidden  # the comparison is not vacuous: the new leg runs
    held = _path(held_rt, base, lens, retained, keep, epoch=epoch, tau=tau)
    mine = _path(mine_rt, base, lens, retained, keep, epoch=epoch, tau=tau)
    assert float(held["g_log_q"].abs().max()) > 1e-3 and float(held["g_bias"].abs().max()) > 1e-6
    z_held, z_mine = _z_hlg(runtime(fx, chunk_seqs=chunk), base, lens, tau)
    z_dev = float((z_held - z_mine).abs().max())
    assert torch.allclose(z_mine, z_held, atol=Z_TOL, rtol=0), z_dev
    mon_dev, dev = _compare(held, mine, (chunk, epoch, tau))
    # the reference chunk (16 = one call here): the per-chunk path at every chunk size against it
    ref = _path(runtime(fx, chunk_seqs=16, onset=ONSET, ramp=RAMP), base, lens, retained, keep,
                epoch=epoch, tau=tau)
    assert abs(mine["term"] - ref["term"]) <= 1e-12 * max(1.0, abs(ref["term"]))
    assert torch.allclose(mine["g_log_q"], ref["g_log_q"], atol=G_TOL, rtol=0)
    assert torch.allclose(mine["g_bias"], ref["g_bias"], atol=G_TOL, rtol=0)
    assert held["mon"]["lexlat_k2_lam"] == pytest.approx(min(1.0, epoch / 3.0))
    print(f"[record] chunk {chunk} epoch {epoch} tau {tau}: max |dlogZ| {z_dev:.3e}, "
          f"max |dmonitor| {mon_dev:.3e}, max |dgrad log_q| {dev['g_log_q']:.3e}, "
          f"max |dgrad param| {dev['g_bias']:.3e}, term {mine['term']!r} (held {held['term']!r})")


@pytest.mark.parametrize("chunk", [1, 2, 16])
def test_chunked_backward_empty_lattice_kept(fx, tmp_path, chunk):
    """Row 0 has no strict parse (empty lattice): counted, excluded, no gradient; same as held."""
    base, lens, retained, keep = _forced_b_batch()
    retained, keep = retained.to(torch.float64), keep.to(torch.float64)
    kw = dict(chunk_seqs=chunk, onset=ONSET, ramp=RAMP, empty_raise_frac=0.5, abort_dir=str(tmp_path))
    held = _path(runtime(fx, "strict", **kw), base, lens, retained, keep, epoch=2, tau=1.0)
    mine = _path(RC.install_chunked_backward(runtime(fx, "strict", **kw)), base, lens, retained, keep,
                 epoch=2, tau=1.0)
    assert held["mon"]["lexlat_k2_n_empty"] == 1.0
    mon_dev, dev = _compare(held, mine, ("empty", chunk))
    assert bool(torch.all(mine["g_log_q"][0] == 0))
    assert not os.path.exists(os.path.join(str(tmp_path), KT.ABORT_MARKER))
    print(f"[record] empty lattice, chunk {chunk}: max |dmonitor| {mon_dev:.3e}, "
          f"max |dgrad log_q| {dev['g_log_q']:.3e}, max |dgrad param| {dev['g_bias']:.3e}")


@pytest.mark.parametrize("chunk", [1, 16])
def test_chunked_backward_empty_lattice_aborts_as_held(fx, tmp_path, chunk):
    """Above empty_raise_frac (per batch): the same marker and the same raise as held."""
    base, lens, retained, keep = _forced_b_batch()
    markers = {}
    for name, make in (("held", lambda rt: rt), ("mine", RC.install_chunked_backward)):
        work = tmp_path / name
        work.mkdir()
        rt = make(runtime(fx, "strict", chunk_seqs=chunk, onset=ONSET, ramp=RAMP, abort_dir=str(work)))
        with pytest.raises(RuntimeError, match="EMPTY"):
            _path(rt, base, lens, retained.to(torch.float64), keep.to(torch.float64), epoch=1, tau=1.0)
        m = json.load(open(work / KT.ABORT_MARKER))
        m.pop("time")
        m.pop("work_dir")
        markers[name] = m
    assert markers["mine"] == markers["held"]
    assert markers["mine"]["n_empty"] == 1 and markers["mine"]["step"] == 3


def test_chunked_backward_no_grad_pass(fx):
    """RETURNN's dev pass runs the train step under no_grad: the same value, no backward."""
    base, lens, retained, keep = _batch()
    held = _path(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP), base, lens, retained, keep,
                 epoch=3, tau=2.0, grad=False)
    mine = _path(RC.install_chunked_backward(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP)), base,
                 lens, retained, keep, epoch=3, tau=2.0, grad=False)
    _compare(held, mine, "no_grad")


def test_memory_monitors_emitted_and_inert(fx):
    """G1.M: the three memory monitors exist (0.0 on CPU, where no CUDA call is made) and leave the
    term, the total loss and every gradient bit-identical to the held path, which has none of them."""
    base, lens, retained, keep = _batch()
    kw = dict(chunk_seqs=2, onset=ONSET, ramp=RAMP)
    held = _path(runtime(fx, **kw), base, lens, retained, keep, epoch=2, tau=2.0)
    mine = _path(RC.install_chunked_backward(runtime(fx, **kw)), base, lens, retained, keep, epoch=2,
                 tau=2.0)
    keys = list(mine["mon"])
    assert keys.index("lexlat_k2_peak_reserved_gib") + 1 == keys.index(MEMORY[0])
    assert keys[keys.index(MEMORY[0]):keys.index(MEMORY[0]) + len(MEMORY)] == list(MEMORY)
    for key in MEMORY:
        assert key not in held["mon"] and mine["mon"][key] == 0.0, key
    assert mine["term"] == held["term"] and mine["total"] == held["total"]
    for key in ("g_log_q", "g_bias"):
        assert torch.equal(mine[key], held[key]), key
    _compare(held, mine, "memory monitors")


def test_upstream_is_the_held_autograd():
    """z_hlg_upstream against the hand form -((lam / n_keep) / retained_i) on kept rows, 0 elsewhere."""
    keep = torch.tensor([1.0, 0.0, 1.0, 1.0], dtype=torch.float64)
    retained = torch.tensor([7.0, 3.0, 11.0, 13.0], dtype=torch.float64)
    for lam in (1.0 / 3.0, 2.0 / 3.0, 1.0):
        g = RC.z_hlg_upstream(keep=keep, retained=retained, lam=lam, term_dtype=torch.float64)
        want = torch.where(keep > 0, -((lam / keep.sum()) / retained), torch.zeros_like(retained))
        assert torch.equal(g, want), (lam, g, want)


def test_inject_grad_is_zero_valued_and_exact():
    dense = torch.randn(2, 3, 5, dtype=torch.float32, requires_grad=True)
    grad = torch.randn(2, 3, 5, dtype=torch.float32)
    grad[0, 0, 0] = 0.0
    with torch.no_grad():
        dense[0, 0, 0] = -math.inf  # an impossible emission: the addend's value must stay 0
    lam = 2.0 / 3.0
    lam_t = torch.tensor(lam, dtype=torch.float64)
    out = RC._InjectGrad.apply(dense, grad, lam_t)
    assert float(out) == 0.0 and out.dtype == torch.float64
    (out * lam).backward()
    assert torch.equal(dense.grad, grad)


# ================================================================================================
# selection: the train-step override and the config epilog
# ================================================================================================
def test_train_step_installs_and_delegates(fx, monkeypatch):
    from types import SimpleNamespace

    from i6_experiments.users.wu.experiments.unsupervised_asr.model import train_step as TS

    calls = []
    monkeypatch.setattr(TS, "train_step", lambda **kw: calls.append(kw) or "held")
    model = SimpleNamespace(lexlat_k2=runtime(fx, chunk_seqs=4))
    assert RC.train_step(model=model, extern_data="data", x=1) == "held"
    assert type(model.lexlat_k2) is RC.ChunkedBackwardLexlatK2Runtime
    assert model.lexlat_k2.spec.chunk_seqs == 4
    assert calls == [{"model": model, "extern_data": "data", "x": 1}]
    with pytest.raises(AssertionError, match="without the k2"):
        RC.train_step(model=SimpleNamespace(lexlat_k2=None), extern_data="data")


def test_epilog_rebinds_train_step():
    ns = {"train_step": object()}
    exec(RC.EPILOG, ns)  # noqa: S102
    assert ns["train_step"] is RC.train_step


# ================================================================================================
# one full blank-free train step (theta AND phi), the fixture leg behind the model's recognizer
# ================================================================================================
class _FixtureLeg:
    """The fixture's 4-phone graph behind the model's 40-phone recognizer: the leg reads
    ``log_softmax(log_q[..., :4])`` against the fixture bed.  A test adapter, nothing else."""

    def __init__(self, rt):
        self.rt = rt

    def active(self, epoch):
        return self.rt.active(epoch)

    def lam(self, epoch):
        return self.rt.lam(epoch)

    def step(self, log_q, *, feat_lens, retained, keep, epoch, temperature, cfg, global_step=None):
        lq = torch.log_softmax(log_q[..., :O.N_PHONES], dim=-1)
        return self.rt.step(lq, feat_lens=feat_lens, retained=retained, keep=keep, epoch=epoch,
                            temperature=temperature, cfg=bed_cfg(), global_step=global_step)


@pytest.mark.parametrize("chunk", [1, 16])
@pytest.mark.parametrize("epoch", [1, 3])
def test_full_train_step_theta_and_phi(fx, inputs, chunk, epoch):
    import test_model_blankfree as TB

    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import train_step as TS

    model = BM.get_model(epoch=1, step=0, device="cpu", **TB._kwargs(inputs))
    models = {"held": model, "mine": copy.deepcopy(model)}
    rt_kw = dict(chunk_seqs=chunk, onset=ONSET, ramp=RAMP)
    models["held"].lexlat_k2 = _FixtureLeg(runtime(fx, **rt_kw))
    models["mine"].lexlat_k2 = _FixtureLeg(RC.install_chunked_backward(runtime(fx, **rt_kw)))
    rec = {}
    for name, m in models.items():
        losses, total = TB._run(m, TS.train_step, TB._extern_data(), epoch=epoch)
        rec[name] = {"total": total,
                     "losses": {k: v.loss.raw_tensor.detach().clone() for k, v in losses.items()},
                     "scale": {k: v.scale for k, v in losses.items()},
                     "as_error": {k: v.as_error for k, v in losses.items()},
                     "grads": {k: (None if p.grad is None else p.grad.detach().clone())
                               for k, p in m.named_parameters()}}
    held, mine = rec["held"], rec["mine"]
    assert "lexlat_k2" in held["losses"] and held["scale"]["lexlat_k2"] == pytest.approx(min(1.0, epoch / 3))
    assert mine["total"] == held["total"]
    held_keys = _held_keys({k: v.sum() for k, v in held["losses"].items()})
    assert [k for k in mine["losses"] if k not in MEMORY] == held_keys
    for key in MEMORY:  # the G1.M monitors: reported as errors, never part of the total
        assert mine["as_error"][key] and float(mine["losses"][key]) == 0.0, key
    skip = TIMING | {"blankfree_frames_per_sec"}
    for key, value in held["losses"].items():
        if key in skip or key not in held_keys:
            continue
        a, b = mine["losses"][key], value
        assert a.dtype == b.dtype and (torch.equal(a, b) or bool(torch.isnan(a).all() and torch.isnan(b).all())), (key, a, b)
    worst = {"recognizer": 0.0, "reverse": 0.0, "other": 0.0}
    for key, g in held["grads"].items():
        other = mine["grads"][key]
        assert (g is None) == (other is None), key
        if g is None:
            continue
        d = float((g.double() - other.double()).abs().max())
        part = key.split(".")[0] if key.split(".")[0] in worst else "other"
        worst[part] = max(worst[part], d)
        assert torch.allclose(other, g, atol=G_TOL, rtol=0), (key, d)
    assert any(k.startswith("reverse.") and g is not None for k, g in held["grads"].items())
    assert any(k.startswith("recognizer.") and g is not None for k, g in held["grads"].items())
    print(f"[record] full train step chunk {chunk} epoch {epoch}: total {mine['total']!r} (held "
          f"{held['total']!r}); max |dgrad| theta {worst['recognizer']:.3e}, phi {worst['reverse']:.3e}, "
          f"other {worst['other']:.3e}")


# ================================================================================================
# RETURNN's non-finite stop (``stop_on_nonfinite_train_score = True``, the rt configs): a non-finite
# stability read (a diagnostic) must not end the arm; a non-finite total loss or other monitor must
# ================================================================================================
#: a reference rung above the fixture's max_active 10000, so the stability read actually runs
REF_RUNG = 20000
#: the case the Engine's train step runs; module level, as RETURNN's own engine tests keep theirs
_ENGINE_CASE = {}


class _EngineLegModel(torch.nn.Module):
    """RETURNN's ``get_model``: one parameter upstream of the fixture leg's ``log_q``."""

    def __init__(self, **_kwargs):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(O.N_PHONES, dtype=torch.float64))


def _engine_train_step(*, model, extern_data, **_kwargs):
    """The fixture leg on a fixed batch; the leg's term and monitors marked as the held train step
    marks them (``model/train_step.py``: ``mark_as_loss(term, "lexlat_k2", scale=lam)`` and, per
    monitor, ``mark_as_loss(torch.tensor([float(value)]), key, as_error=True,
    use_normalized_loss=False)``).  ``extern_data`` only paces the steps."""
    ctx = rf.get_run_ctx()
    rt = _ENGINE_CASE["rt"]
    epoch = int(ctx.epoch)
    base, lens, retained, keep = _batch()
    log_q = torch.log_softmax(base.to(torch.float64) + model.bias, dim=-1)
    term, mon = rt.step(log_q, feat_lens=lens, retained=retained, keep=keep, epoch=epoch, temperature=1.0,
                        cfg=bed_cfg(), global_step=int(ctx.step))
    if _ENGINE_CASE.get("nan_total"):
        # the scored term's VALUE is nan (so is the total loss); its gradient stays finite, so the
        # parameters stay finite and RETURNN's debug re-run of the step reaches its own raise
        term = term + torch.tensor(float("nan"), dtype=term.dtype)
    if _ENGINE_CASE.get("nan_monitor"):
        mon[_ENGINE_CASE["nan_monitor"]] = torch.tensor(float("nan"), dtype=torch.float64)
    ctx.mark_as_loss(term, "lexlat_k2", scale=rt.lam(epoch), use_normalized_loss=False)
    for key, value in mon.items():
        ctx.mark_as_loss(torch.tensor([float(value)], device=log_q.device), key,
                         as_error=True, use_normalized_loss=False)


def _force_reference_rung(monkeypatch, mode):
    """The stability read's REFERENCE-rung call (``max_active_states == REF_RUNG``; the training leg
    runs at 10000) fails (``failed``: an OOM, caught by the held read) or scores no utterance
    (``empty``: every total -inf).  Either way the held read returns ``nan``."""
    orig = KT.K.chunked_tot_scores

    def patched(*args, **kw):
        if int(kw.get("max_active_states", -1)) != REF_RUNG:
            return orig(*args, **kw)
        if mode == "failed":
            raise torch.cuda.OutOfMemoryError("forced: the reference rung does not fit")
        tot, chunks, sec = orig(*args, **kw)
        return torch.full_like(tot, -math.inf), chunks, sec

    monkeypatch.setattr(KT.K, "chunked_tot_scores", patched)


def _run_engine(tmp_path, rt, **case):
    """One sub-epoch of RETURNN's torch Engine on CPU with the rt configs' non-finite stop.
    :return: the sub-epoch's ``learning_rates`` error dict and the file's text"""
    from returnn.config import Config, global_config_ctx
    from returnn.datasets import init_dataset
    from returnn.log import log
    from returnn.torch.engine import Engine

    if not getattr(log, "initialized", False):
        log.initialize(verbosity=[3])
    _ENGINE_CASE.clear()
    _ENGINE_CASE.update(rt=rt, **case)
    lr_file = str(tmp_path / "learning_rates")
    config = Config(dict(
        task="train", device="cpu", num_epochs=1, model=str(tmp_path / "epoch"),
        learning_rate_file=lr_file, stop_on_nonfinite_train_score=True,
        extern_data={"data": {"dim": 9}, "classes": {"dim": 2, "sparse": True}},
        get_model=_EngineLegModel, train_step=_engine_train_step, batch_size=500,
        optimizer={"class": "adam"}, learning_rate=1e-3,
        torch_dataloader_opts={"num_workers": 0}))  # a forked loader worker segfaults under pytest here
    dataset = init_dataset({"class": "Task12AXDataset", "num_seqs": 12, "name": "train"})
    dataset.init_seq_order(epoch=1)
    try:
        with global_config_ctx(config):
            engine = Engine(config=config)
            engine.init_train_from_config(train_data=dataset)
            engine.train()
            error = dict(engine.learning_rate_control.epoch_data[1].error)
            n_steps = engine.learning_rate_control.epoch_data[1].meta["epoch_num_train_steps"]
    finally:
        _ENGINE_CASE.clear()
    assert n_steps >= 2, n_steps  # the cached read is emitted (or omitted) on more than one step
    return error, open(lr_file).read()


@pytest.mark.parametrize("mode", ["failed", "empty"])
def test_nonfinite_stability_does_not_stop_training(fx, tmp_path, monkeypatch, capfd, mode):
    """A failed / empty stability read: the sub-epoch completes, the read's print line says so, and
    the learning-rate file has every other lexlat_k2 column but no lexlat_k2_stability (not 0)."""
    _force_reference_rung(monkeypatch, mode)
    rt = RC.install_chunked_backward(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP,
                                             stability_reference_max_active=REF_RUNG))
    error, text = _run_engine(tmp_path, rt)
    out = capfd.readouterr().out
    want = ("stability read at sub-epoch 1 FAILED (OutOfMemoryError" if mode == "failed"
            else "stability at sub-epoch 1: median nan nats per retained frame over 0 of 5")
    assert want in out, out[-2000:]
    assert "Inf/nan score" not in out
    assert "train_loss_lexlat_k2_lam" in error and "train_loss_lexlat_k2_term_mean" in error
    assert f"train_loss_{STABILITY}" not in error and STABILITY not in text, error
    assert all(math.isfinite(v) for v in error.values()), error
    print(f"[record] engine, {mode} stability read: sub-epoch 1 completed; learning_rates keys "
          f"{sorted(k for k in error if 'lexlat_k2' in k)}")


def test_finite_stability_is_logged(fx, tmp_path, capfd):
    """A real read (reference rung above the arm's): the column is there, with the printed median."""
    rt = RC.install_chunked_backward(runtime(fx, chunk_seqs=2, onset=ONSET, ramp=RAMP,
                                             stability_reference_max_active=REF_RUNG))
    error, text = _run_engine(tmp_path, rt)
    out = capfd.readouterr().out
    value = error[f"train_loss_{STABILITY}"]
    assert math.isfinite(value) and value >= 0.0, value
    assert f"stability at sub-epoch 1: median {value:.4f} nats per retained frame over 5 of 5" in out
    assert f"train_loss_{STABILITY}" in text
    print(f"[record] engine, real stability read: {STABILITY} = {value!r}")


@pytest.mark.parametrize("case", ["held_stability_nan", "total_nan", "other_monitor_nan"])
def test_nonfinite_stop_still_fires(fx, tmp_path, monkeypatch, case):
    """The guard is unchanged for everything else: a NaN total loss or another NaN monitor still ends
    the run; and the HELD runtime's NaN stability read does (the defect this module removes)."""
    _force_reference_rung(monkeypatch, "failed")
    kw = dict(chunk_seqs=2, onset=ONSET, ramp=RAMP, stability_reference_max_active=REF_RUNG)
    held_rt = runtime(fx, **kw)
    rt = held_rt if case == "held_stability_nan" else RC.install_chunked_backward(held_rt)
    extra = {"total_nan": dict(nan_total=True),
             "other_monitor_nan": dict(nan_monitor="lexlat_k2_term_mean")}.get(case, {})
    with pytest.raises(Exception, match="Inf/nan score in step 0"):
        _run_engine(tmp_path, rt, **extra)


# ================================================================================================
# GPU (the arms' device): the same comparison on CUDA
# ================================================================================================
@pytest.mark.gpu
@pytest.mark.parametrize("chunk", [1, 4])
def test_chunked_backward_equals_held_cuda(fx, chunk):
    base, lens, retained, keep = _batch()
    kw = dict(chunk_seqs=chunk, onset=ONSET, ramp=RAMP)
    held = _path(runtime(fx, **kw), base, lens, retained, keep, epoch=2, tau=2.0, device="cuda")
    mine = _path(RC.install_chunked_backward(runtime(fx, **kw)), base, lens, retained, keep, epoch=2,
                 tau=2.0, device="cuda")
    z_held, z_mine = _z_hlg(runtime(fx, chunk_seqs=chunk), base, lens, 2.0, device="cuda")
    assert torch.allclose(z_mine, z_held, atol=Z_TOL, rtol=0)
    assert mine["term"] == pytest.approx(held["term"], rel=1e-12, abs=1e-12)
    # lexlat_k2_stability is the held read's NaN "no read" here (the fixture's max_active 10000 is
    # not below the reference rung 10000): the per-chunk path omits it
    assert [k for k in mine["mon"] if k not in MEMORY] == _held_keys(held["mon"])
    for key, value in held["mon"].items():
        if key not in TIMING and key in _held_keys(held["mon"]):
            other = mine["mon"][key]
            # equal, or both NaN
            assert (math.isnan(value) and math.isnan(other)) or other == pytest.approx(
                value, rel=1e-9, abs=1e-12), (key, other, value)
    for key in ("g_log_q", "g_bias"):
        assert torch.allclose(mine[key], held[key], atol=G_TOL, rtol=0), key
    mem = {key: mine["mon"][key] for key in MEMORY}
    assert all(math.isfinite(v) and v > 0.0 for v in mem.values()), mem
    print(f"[record] cuda chunk {chunk} memory monitors (GiB): {mem}")
    print(f"[record] cuda chunk {chunk}: max |dlogZ| {float((z_held - z_mine).abs().max()):.3e}, "
          f"max |dgrad log_q| {float((held['g_log_q'] - mine['g_log_q']).abs().max()):.3e}, "
          f"max |dgrad param| {float((held['g_bias'] - mine['g_bias']).abs().max()):.3e}")
