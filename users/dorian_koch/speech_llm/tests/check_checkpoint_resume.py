"""Guard: the shared checkpoint/resume round-trips, and refuses to resume from a partial checkpoint.

Resume is the one feature whose bugs are invisible in the happy path: a run that never gets preempted
never touches this code, and a run that *does* get preempted silently restarts from step 1 (wasting an
allocation) or -- worse -- resumes with half-restored state and trains a different experiment under
the same name. Neither shows up as a crash. So exercise it directly.

Three of the five launchers (moshirag, personaplex, audex) had no resume at all before 2026-08-01 and
restarted from scratch on every preemption; the other two had two independent implementations. These
helpers are the union, so this check is also what says the newly-granted capability actually works.

Run from the setup root, no GPU needed:
    CUDA_HOME=/usr .venv/bin/python \\
        recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_checkpoint_resume.py
"""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import torch  # noqa: E402

from moshi_family.train_loop import (  # noqa: E402
    LEGACY_STATE_NAME,
    TRAINER_STATE_NAME,
    find_latest_checkpoint,
    is_lora_param,
    prune_optimizer_state,
    resume_from_checkpoint,
    restore_trained_params,
    save_trained_params,
    save_trainer_state,
)


class Tiny(torch.nn.Module):
    """A stand-in for a LoRA-wrapped model: two adapter params plus a frozen base."""

    def __init__(self):
        super().__init__()
        self.base_weight = torch.nn.Parameter(torch.randn(4, 4), requires_grad=False)
        self.layer_lora_A = torch.nn.Parameter(torch.randn(4, 2))
        self.layer_lora_B = torch.nn.Parameter(torch.randn(2, 4))

    def forward(self, x):
        return x @ self.base_weight + (x @ self.layer_lora_A) @ self.layer_lora_B


def ckpt_dir(root: Path, step: int) -> Path:
    return root / "checkpoints" / f"checkpoint_{step:06d}" / "consolidated"


def lora_keys(m) -> set:
    return {n for n, _ in m.named_parameters() if is_lora_param(n)}


torch.manual_seed(0)
tmp = Path(tempfile.mkdtemp(prefix="ckpt-resume-"))

# --- what gets saved -----------------------------------------------------------------------------
model = Tiny()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
saved = save_trained_params(ckpt_dir(tmp, 100) / "lora.safetensors", model, predicate=is_lora_param)
assert set(saved) == {"layer_lora_A", "layer_lora_B"}, sorted(saved)
assert "base_weight" not in saved, "the predicate let a frozen base weight into a LoRA checkpoint"
print(f"[ok] save_trained_params writes only what the predicate selects  ({sorted(saved)})")

try:
    save_trained_params(tmp / "empty.safetensors", model, predicate=lambda n: False)
    raise AssertionError("a predicate matching nothing must not write an empty checkpoint")
except AssertionError as e:
    assert "empty checkpoint" in str(e), e
print("[ok] a predicate matching nothing fails loudly, not silently")

# --- take real optimizer steps so the moments are non-trivial ------------------------------------
for _ in range(3):
    opt.zero_grad()
    model(torch.randn(2, 4)).sum().backward()
    opt.step()

save_trained_params(ckpt_dir(tmp, 200) / "lora.safetensors", model, predicate=is_lora_param)
save_trainer_state(ckpt_dir(tmp, 200), step=200, optimizer=opt, rng_extra={"numpy": "state"})
reference = {n: p.detach().clone() for n, p in model.named_parameters() if is_lora_param(n)}
exp_avg = opt.state[model.layer_lora_A]["exp_avg"].clone()
assert exp_avg.abs().sum() > 0, "test bug: optimizer moments are still zero, nothing to verify"
next_draw = torch.randn(3)

# --- step 100 has weights but NO sidecar, so it must be skipped ----------------------------------
found = find_latest_checkpoint(tmp, weights_name="lora.safetensors")
assert found is not None and found[0] == 200, f"expected step 200, got {found}"
print("[ok] a checkpoint with weights but no optimizer state is skipped, not silently resumed")

# --- resume into a *different* model, then compare ------------------------------------------------
torch.manual_seed(999)
fresh = Tiny()
fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-3)
assert not torch.allclose(fresh.layer_lora_A, reference["layer_lora_A"]), "test bug: models identical"

step, state = resume_from_checkpoint(
    tmp,
    module=fresh,
    optimizer=fresh_opt,
    weights_name="lora.safetensors",
    expect=lora_keys(fresh),
    log=lambda m: None,
)
assert step == 200, step
for name, want in reference.items():
    got = dict(fresh.named_parameters())[name]
    assert torch.equal(got, want), f"{name} did not round-trip"
print(f"[ok] weights restored exactly                         (step {step})")

restored_avg = fresh_opt.state[fresh.layer_lora_A]["exp_avg"]
assert torch.equal(restored_avg, exp_avg), "Adam moments did not round-trip -- resume would restart them"
print("[ok] optimizer moments restored exactly")

assert state["extra"] == {"numpy": "state"}, state["extra"]
print("[ok] caller-owned RNG state (rng_extra) round-trips")

assert torch.equal(torch.randn(3), next_draw), "torch RNG not restored -- resume would redraw data"
print("[ok] torch RNG restored: the next draw matches the pre-save one")

# --- a mismatched key set must be loud ------------------------------------------------------------
try:
    restore_trained_params(ckpt_dir(tmp, 200) / "lora.safetensors", fresh, expect=lora_keys(fresh) | {"layer_lora_C"})
    raise AssertionError("a checkpoint whose key set differs from the model's must not load")
except AssertionError as e:
    assert "param set" in str(e), e
print("[ok] a rank/config mismatch fails instead of half-restoring")

# --- the legacy sidecar name still resumes --------------------------------------------------------
legacy = tmp / "legacy"
save_trained_params(ckpt_dir(legacy, 50) / "lora.safetensors", model, predicate=is_lora_param)
torch.save({"optimizer": opt.state_dict(), "step": 50}, ckpt_dir(legacy, 50) / LEGACY_STATE_NAME)
assert not (ckpt_dir(legacy, 50) / TRAINER_STATE_NAME).exists()
step, _ = resume_from_checkpoint(
    legacy,
    module=Tiny(),
    optimizer=torch.optim.AdamW(Tiny().parameters(), lr=1e-3),
    weights_name="lora.safetensors",
    log=lambda m: None,
)
assert step == 50, step
print(f"[ok] a pre-unification checkpoint ({LEGACY_STATE_NAME}) still resumes")

# --- nothing to resume from -----------------------------------------------------------------------
step, state = resume_from_checkpoint(
    tmp / "nonexistent",
    module=Tiny(),
    optimizer=torch.optim.AdamW(Tiny().parameters(), lr=1e-3),
    weights_name="lora.safetensors",
    log=lambda m: None,
)
assert (step, state) == (0, None), (step, state)
print("[ok] an empty out_dir starts fresh at step 0")

# --- pruning old optimizer state: keep every checkpoint's WEIGHTS, drop the resume sidecars --------
# Two thirds of a LoRA checkpoint is AdamW moments (1552 MB against 776 MB of weights on a real r128
# arm), and only the newest complete checkpoint is ever resumed from. Dropping the rest is what makes
# a dense save_every affordable -- which is what lets a later experiment test the model DURING a
# collapse instead of only at the two checkpoints that happened to straddle it.
prune_root = Path(tempfile.mkdtemp(prefix="ckpt-prune-"))
m_p = Tiny()
opt_p = torch.optim.AdamW(m_p.parameters(), lr=1e-3)
m_p(torch.randn(2, 4)).sum().backward()
opt_p.step()  # give AdamW real moments, so the sidecar is not trivially empty
for s in (50, 100, 150, 200):
    d = ckpt_dir(prune_root, s)
    d.mkdir(parents=True, exist_ok=True)
    save_trained_params(d / "lora.safetensors", m_p, predicate=is_lora_param)
    save_trainer_state(d, step=s, optimizer=opt_p)

removed = prune_optimizer_state(prune_root, keep_last=2, log=lambda m: None)
assert removed == 2, removed
# The weights survive EVERYWHERE. That is the whole point: a pruned checkpoint must still be
# loadable by an evaluation invented months later.
for s in (50, 100, 150, 200):
    assert (ckpt_dir(prune_root, s) / "lora.safetensors").exists(), s
for s in (50, 100):
    assert not (ckpt_dir(prune_root, s) / TRAINER_STATE_NAME).exists(), s
for s in (150, 200):
    assert (ckpt_dir(prune_root, s) / TRAINER_STATE_NAME).exists(), s
print("[ok] pruning keeps every checkpoint's weights and only the newest resume sidecars")

found = find_latest_checkpoint(prune_root, weights_name="lora.safetensors")
assert found is not None and found[0] == 200, found
# If the newest sidecar is lost (a kill mid-write), keep_last=2 leaves a REAL fallback rather than
# sending the run back to step 0. This is why the default is 2 and not 1.
os.remove(ckpt_dir(prune_root, 200) / TRAINER_STATE_NAME)
found = find_latest_checkpoint(prune_root, weights_name="lora.safetensors")
assert found is not None and found[0] == 150, found
# ...and once no sidecar is left, a pruned checkpoint is SKIPPED rather than resumed weights-only:
# restarting Adam's moments and the LR schedule mid-run is a different experiment under the same
# name. Pruning must not weaken that rule.
os.remove(ckpt_dir(prune_root, 150) / TRAINER_STATE_NAME)
found = find_latest_checkpoint(prune_root, weights_name="lora.safetensors")
assert found is None, f"a weights-only checkpoint was offered for resume: {found}"
print("[ok] resume lands on the newest kept sidecar, falls back once, then refuses weights-only")

# Housekeeping must never take a training run down.
assert prune_optimizer_state(prune_root / "nope", keep_last=2, log=lambda m: None) == 0
assert prune_optimizer_state(prune_root, keep_last=0, log=lambda m: None) == 0
print("[ok] pruning a missing dir, or with keep_last < 1, is a no-op rather than an error")

print("\ncheckpoint/resume round-trips, and refuses every partial checkpoint")
