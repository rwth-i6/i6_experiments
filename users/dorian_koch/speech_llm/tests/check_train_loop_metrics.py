"""Guard: the training loop must LOG what an LR/collapse investigation needs, and must not lie.

Two bugs this pins down, both found 2026-08-05:

1.  ``metrics.train.jsonl`` carried ``lr_scale`` (the schedule multiplier) but never the actual
    learning rate. Two arms with different ``lr`` render an IDENTICAL ``lr_scale`` curve, so the one
    axis backlog A3/A4 exists to sweep was the one axis not recorded. Same for the gradient norm:
    ``clip_grad_norm_`` already computes it and the return value was discarded, so there was no way
    to tell whether ``grad_clip`` was active -- and when it is, the effective LR is not the
    configured one, which silently changes what an LR sweep measures.

2.  A non-finite gradient norm was stepped straight into the weights. Every later loss is then nan,
    which reads as a modelling problem rather than one bad batch.

The loop is exercised for real here -- a tiny nn.Module, a real AdamW, real backward passes -- rather
than by asserting on a hand-built record. A guard that builds its own idealised inputs is how
``check_mixed_loader`` missed a live DDP bug for months (see CLAUDE.md).

Run: ``CUDA_HOME=/usr .venv/bin/python recipe/speech_llm/full_duplex/moshi_family/tests/...`` -- or
from the setup root via the path used by the other checks.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from speech_llm.full_duplex.moshi_family.train_loop import (  # noqa: E402
    OTHER_BUCKET,
    TrainConfig,
    classify_param,
    run_training,
)


def _run(*, max_steps, grad_accum=2, log_every=1, grad_clip=1.0, scale_loss=1.0, base_lrs=None):
    """Drive run_training over a 2-group toy model and return the parsed metrics rows."""
    torch.manual_seed(0)
    lin_a = torch.nn.Linear(4, 4)
    lin_b = torch.nn.Linear(4, 4)
    params_a = list(lin_a.parameters())
    params_b = list(lin_b.parameters())
    base_lrs = base_lrs if base_lrs is not None else [1e-3, 2e-3]
    opt = torch.optim.AdamW([{"params": params_a, "lr": base_lrs[0]}, {"params": params_b, "lr": base_lrs[1]}])

    def batches():
        while True:
            yield torch.randn(2, 4)

    def loss_step(batch):
        return (lin_b(lin_a(batch)) * scale_loss).pow(2).mean()

    out_dir = tempfile.mkdtemp()
    run_training(
        optimizer=opt,
        base_lrs=base_lrs,
        group_names=("temporal", "depth"),
        trainable_params=params_a + params_b,
        batch_iter=batches(),
        loss_step=loss_step,
        save_fn=lambda step, final: None,
        cfg=TrainConfig(
            max_steps=max_steps,
            grad_accum=grad_accum,
            warmup_steps=max(1, max_steps // 2),
            grad_clip=grad_clip,
            save_every=0,
            log_every=log_every,
        ),
        out_dir=out_dir,
        log=lambda m: None,
    )
    path = Path(out_dir) / "metrics.train.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_lr_is_logged():
    rows = _run(max_steps=4, base_lrs=[1e-3, 2e-3])
    assert rows, "no metrics written"
    for r in rows:
        for key in ("lr", "lr_scale", "lr_by_group", "grad_norm", "grad_clip", "grad_clipped"):
            assert key in r, f"missing {key} in {r}"
        # lr must be the ACTUAL rate, i.e. base * schedule -- not the bare multiplier.
        assert abs(r["lr"] - 1e-3 * r["lr_scale"]) < 1e-12, r
        assert set(r["lr_by_group"]) == {"temporal", "depth"}, r["lr_by_group"]
        assert abs(r["lr_by_group"]["depth"] - 2e-3 * r["lr_scale"]) < 1e-12, r
        # Distinct base LRs must stay distinct -- except at lr_scale==0, where the cosine schedule
        # legitimately drives every group to zero at the final step.
        if r["lr_scale"] > 0:
            assert r["lr_by_group"]["depth"] > r["lr_by_group"]["temporal"], r

    other = _run(max_steps=4, base_lrs=[1e-5, 2e-5])
    assert [r["lr_scale"] for r in rows] == [r["lr_scale"] for r in other], (
        "lr_scale is identical across base LRs -- which is exactly why it is not enough"
    )
    assert [r["lr"] for r in rows] != [r["lr"] for r in other], "lr must separate the two arms"
    print("PASS  lr logged as the real rate, per group, and separates arms lr_scale cannot")


def check_grad_norm_and_clipping():
    # Tiny loss -> tiny grads -> below the clip threshold.
    small = _run(max_steps=3, grad_clip=1e6, scale_loss=1.0)
    assert all(r["grad_norm"] >= 0.0 for r in small), small
    assert not any(r["grad_clipped"] for r in small), "nothing should clip at grad_clip=1e6"
    # Large loss scale -> large grads -> clipping must be REPORTED, not just silently applied.
    big = _run(max_steps=3, grad_clip=1e-8, scale_loss=50.0)
    assert all(r["grad_clipped"] for r in big), big
    assert all(r["grad_norm"] > 1e-8 for r in big), "grad_norm must be the PRE-clip value"
    assert all("grad_norm_by_group" in r and r["grad_norm_by_group"] for r in big), big
    assert set(big[0]["grad_norm_by_group"]) == {"temporal", "depth"}, big[0]
    print("PASS  grad_norm is pre-clip, per group, and clipping is reported")


def check_non_finite_grad_is_fatal():
    """A nan/inf gradient must stop the run, not be stepped into the weights."""
    try:
        _run(max_steps=3, scale_loss=float("inf"))
    except AssertionError as e:
        assert "non-finite gradient norm" in str(e), f"wrong assertion fired: {e}"
        print("PASS  non-finite gradient norm raises instead of poisoning the weights")
        return
    raise SystemExit("FAIL: a non-finite gradient norm did not stop training")


def check_step_timing_present():
    rows = _run(max_steps=2)
    assert all("step_seconds" in r and r["step_seconds"] >= 0 for r in rows), rows
    print("PASS  step_seconds logged")


def _named_toy():
    """A toy module whose parameter names match the four real MODULE_BUCKETS prefixes.

    Names taken from a real trained LoRA checkpoint (a8-fast): ``text_linear.*``, ``linears.N.*``,
    ``depformer.layers.N.*`` / ``depformer_in.N.*``, ``transformer.layers.N.*``.
    """
    m = torch.nn.Module()
    m.text_linear = torch.nn.Linear(4, 4)
    m.linears = torch.nn.ModuleList([torch.nn.Linear(4, 4)])
    m.depformer = torch.nn.Linear(4, 4)
    m.depformer_in = torch.nn.Linear(4, 4)
    m.transformer = torch.nn.Linear(4, 4)
    return m


def check_module_classification():
    """Every real checkpoint name must land in its intended bucket, and nothing in `other`."""
    real = [
        ("text_linear.lora_A.weight", "text_head"),
        ("linears.3.lora_B.weight", "audio_heads"),
        ("depformer.layers.2.self_attn.in_projs.0.lora_A.weight", "depformer"),
        ("depformer_in.1.lora_B.weight", "depformer"),
        ("transformer.layers.7.gating.linear_out.lora_A.weight", "temporal"),
    ]
    for name, want in real:
        got = classify_param(name)
        assert got == want, f"{name!r} -> {got!r}, expected {want!r}"
    # A renamed module must be REPORTED, not absorbed into a neighbour.
    assert classify_param("brand_new_stack.0.weight") == OTHER_BUCKET
    print("PASS  module classification covers the real checkpoint names; drift lands in `other`")


def _run_named(*, max_steps=4, aux=False, track_weight_delta=True):
    torch.manual_seed(0)
    m = _named_toy()
    named = [(n, p) for n, p in m.named_parameters()]
    params = [p for _, p in named]
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-2}])

    def batches():
        while True:
            yield torch.randn(2, 4)

    def loss_step(batch):
        h = m.transformer(batch)
        h = m.depformer(m.depformer_in(h))
        loss = m.text_linear(h).pow(2).mean() + m.linears[0](h).pow(2).mean()
        if aux:
            return loss, {"text_loss": 1.25, "audio_loss": 0.5}
        return loss

    out_dir = tempfile.mkdtemp()
    run_training(
        optimizer=opt,
        base_lrs=[1e-2],
        group_names=("all",),
        trainable_params=params,
        named_trainable=named,
        batch_iter=batches(),
        loss_step=loss_step,
        save_fn=lambda step, final: None,
        cfg=TrainConfig(
            max_steps=max_steps,
            grad_accum=2,
            warmup_steps=1,
            grad_clip=1e6,
            save_every=0,
            log_every=1,
            track_weight_delta=track_weight_delta,
        ),
        out_dir=out_dir,
        log=lambda msg: None,
    )
    path = Path(out_dir) / "metrics.train.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def check_per_module_norms():
    rows = _run_named()
    want = {"text_head", "audio_heads", "depformer", "temporal"}
    for r in rows:
        g = r["grad_norm_by_module"]
        assert set(g) == want, f"buckets {set(g)} != {want}"
        assert all(v > 0 for v in g.values()), g
    print("PASS  grad_norm_by_module reports all four stacks separately")


def check_weight_delta_grows():
    rows = _run_named(max_steps=5)
    d0 = rows[0]["weight_delta_by_module"]
    dN = rows[-1]["weight_delta_by_module"]
    assert set(d0) == set(dN), (d0, dN)
    # The whole point: the delta is measured from theta_0, so it must GROW as training moves.
    for b in d0:
        assert dN[b] > d0[b], f"{b}: delta did not grow ({d0[b]} -> {dN[b]})"
    assert rows[0]["weight_delta_baseline_step"] == 0
    # And it must be separable from the gradient norm -- a stack can be pushed hard and not move.
    assert rows[-1]["grad_norm_by_module"].keys() == dN.keys()
    print("PASS  weight_delta_by_module grows from the theta_0 baseline, per stack")


def check_weight_delta_can_be_disabled():
    rows = _run_named(track_weight_delta=False)
    assert all(r["weight_delta_by_module"] is None for r in rows), rows[0]
    assert all(r["grad_norm_by_module"] for r in rows), "grad norms must survive the opt-out"
    print("PASS  track_weight_delta=False drops the snapshot but keeps grad norms")


def check_loss_components_merged():
    """loss_step may return (loss, aux); aux must reach the record, averaged over microbatches."""
    rows = _run_named(aux=True)
    for r in rows:
        assert abs(r["text_loss"] - 1.25) < 1e-9, r
        assert abs(r["audio_loss"] - 0.5) < 1e-9, r
    # Backward compatibility: a bare-scalar loss_step must still work and simply add no keys.
    plain = _run_named(aux=False)
    assert "text_loss" not in plain[0], plain[0]
    print("PASS  loss components merged into every log row; scalar loss_step still supported")


if __name__ == "__main__":
    check_lr_is_logged()
    check_grad_norm_and_clipping()
    check_non_finite_grad_is_fatal()
    check_step_timing_present()
    check_module_classification()
    check_per_module_norms()
    check_weight_delta_grows()
    check_weight_delta_can_be_disabled()
    check_loss_components_merged()
    print("ALL PASS")
