"""Guard: small-LR updates to bf16 trainable weights are not rounded away (``TrainConfig.fp32_master_weights``).

WHY. Found 2026-09-22 on p0 (PersonaPlex LoRA r128, lr 1e-6/2e-6): AdamW stepped the bf16 adapter in
place, so an update of ~lr is far below half a bf16 ulp of a weight of |w| ~ 1e-2 and rounds away --
between steps 100 and 600 only 1.8% of lora_A elements changed at all, and the optimizer moments were
bf16. The reference moshi_finetune steps fp32 master copies (``finetune/mixed_precision.py``).

Drives the real ``train_loop.run_training`` on a bf16 linear layer with |w| ~ 1e-2 at lr 1e-6:
  1. without masters the weights do not move at all (the bug, reproduced -- this is what makes the guard able
     to fail: if the flag stopped working, [2] would look like [1]);
  2. with masters they track an all-fp32 run of the same loop to within bf16 rounding, and the AdamW
     moments are fp32.

  CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_fp32_master_weights.py
"""

import sys
import tempfile
from pathlib import Path

import torch

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))

from moshi_family.train_loop import TrainConfig, run_training  # noqa: E402

# 400 steps of ~lr each add up to several bf16 ulps of a |w| ~ 1e-2 weight (ulp 6e-5), while each
# single step is far below half an ulp -- the regime p0 trained in.
STEPS, LR = 400, 1e-6


def train(dtype, master: bool):
    torch.manual_seed(0)
    layer = torch.nn.Linear(256, 64, bias=False)
    torch.nn.init.normal_(layer.weight, std=1e-2)
    layer = layer.to(dtype)
    w0 = layer.weight.detach().float().clone()  # AFTER the cast: measure training, not the rounding
    x, y = torch.randn(32, 256), torch.randn(32, 64)

    def batches():
        while True:
            yield x.to(dtype), y.to(dtype)

    def loss_step(batch):
        xb, yb = batch
        return ((layer(xb).float() - yb.float()) ** 2).mean()

    opt = torch.optim.AdamW([layer.weight], lr=LR, betas=(0.9, 0.95), weight_decay=0.0)
    run_training(
        optimizer=opt,
        base_lrs=[LR],
        trainable_params=[layer.weight],
        batch_iter=batches(),
        loss_step=loss_step,
        save_fn=lambda step, final: None,
        cfg=TrainConfig(
            max_steps=STEPS,
            grad_accum=1,
            warmup_steps=0,
            save_every=10**9,
            log_every=10**9,
            fp32_master_weights=master,
        ),
        out_dir=tempfile.mkdtemp(prefix="check_fp32_master_"),
        log=lambda m: None,
    )
    moments = {k: v.dtype for k, v in opt.state[layer.weight].items() if k != "step"}
    return layer.weight.detach().float() - w0, moments


ref, _ = train(torch.float32, master=False)
bug, _ = train(torch.bfloat16, master=False)
fix, moments = train(torch.bfloat16, master=True)

moved_bug = float((bug != 0).float().mean())
moved_fix = float((fix != 0).float().mean())
# Compare the UPDATE, not the weights: bf16 storage rounds each weight by up to half an ulp, so the
# fixed run's stored delta is the fp32 delta plus that rounding. Relative to the fp32 run's update:
err_bug = float((bug - ref).norm() / ref.norm())
err_fix = float((fix - ref).norm() / ref.norm())
print(f"[1] bf16, no masters: {moved_bug:.1%} of weights moved; update error vs fp32 run {err_bug:.2f}")
print(f"[2] bf16 + fp32 masters: {moved_fix:.1%} moved; update error vs fp32 run {err_fix:.2f}; moments {moments}")
assert moved_bug < 0.05 and err_bug > 0.9, (
    "the bf16 bug did not reproduce -- the guard cannot tell the fix from its absence"
)
assert moved_fix > 0.5 and err_fix < 0.3, "fp32 masters did not recover the fp32 update"
assert all(d == torch.float32 for d in moments.values()), moments
print("OK")
