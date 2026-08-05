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
    TrainConfig,
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
    opt = torch.optim.AdamW(
        [{"params": params_a, "lr": base_lrs[0]}, {"params": params_b, "lr": base_lrs[1]}]
    )

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


if __name__ == "__main__":
    check_lr_is_logged()
    check_grad_norm_and_clipping()
    check_non_finite_grad_is_fatal()
    check_step_timing_present()
    print("ALL PASS")
