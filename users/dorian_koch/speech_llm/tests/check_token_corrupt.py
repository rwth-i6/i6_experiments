"""Guard: input-token corruption never touches the TARGETS, and is never a silent no-op.

Backlog A45. Two failure modes, and a guard that tested only one of them would be worse than
nothing, because each looks exactly like success from the other side:

  (1) CORRUPTING THE TARGET. `moshi_loss` takes its targets from the tensor it is handed -- "Targets
      are the *input* codes", its own docstring -- so the forward and the loss see the SAME tensor.
      An in-place corruption would train the model to PREDICT NOISE, behind a smooth loss curve and
      a perfectly loadable checkpoint. Nothing downstream would flag it.

  (2) CORRUPTING NOTHING. A corruption that quietly does nothing passes every "targets are intact"
      assertion perfectly, and turns the arm into a null that reads like a real negative result.
      This is the same shape as the `holdout_every` defect, where the split was tested for years and
      the CALLER never was, so every arm logged "eval every N steps" and evaluated nothing.

So both directions are asserted, and the guard is checked for non-vacuity: a deliberately broken
in-place implementation must FAIL check (1), otherwise the check cannot be trusted to catch it.

Also checks the wiring in the launcher itself, at the AST level, because that is where the defect
would actually live: the model must be called with the corrupted copy and the loss with the pristine
tensor. A function-level test of `corrupt_audio_tokens` cannot see a caller that passes the wrong
one.

  CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/.../tests/check_token_corrupt.py
"""

import ast
import sys
from pathlib import Path

import torch

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))

from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_ROW,
    corrupt_audio_tokens,
)

failures = []


def check(cond, msg):
    print(f"{'[ok]' if cond else '[FAIL]'} {msg}")
    if not cond:
        failures.append(msg)


CARD = 2048
DEP_Q = 8
K = 1 + 2 * DEP_Q  # text + moshi's 8 + the user's 8, the real Moshi layout
B, T = 4, 2000


def make_codes(seed=0):
    g = torch.Generator().manual_seed(seed)
    c = torch.randint(0, CARD, (B, K, T), generator=g, dtype=torch.long)
    c[:, TEXT_ROW] = torch.randint(0, 32000, (B, T), generator=g, dtype=torch.long)
    return c


torch.manual_seed(1234)

# ---------------------------------------------------------------------------------------------
# 1. p <= 0 is a strict no-op and returns the SAME OBJECT
# ---------------------------------------------------------------------------------------------
c = make_codes()
out0 = corrupt_audio_tokens(c, 0.0, cardinality=CARD)
check(out0 is c, "p=0.0 returns the same object (no copy, no work)")

# ---------------------------------------------------------------------------------------------
# 2. THE LOAD-BEARING ONE: the input tensor is not modified in place
# ---------------------------------------------------------------------------------------------
c = make_codes()
before = c.clone()
out = corrupt_audio_tokens(c, 0.05, cardinality=CARD)
check(torch.equal(c, before), "input tensor is BIT-IDENTICAL after the call (targets protected)")
check(out is not c, "a new tensor is returned, not the input")

# ---------------------------------------------------------------------------------------------
# 3. NON-VACUITY: it really does corrupt, at about the requested rate
# ---------------------------------------------------------------------------------------------
diff = out != before
n_audio = B * (K - AUDIO_OFFSET) * T
changed = int(diff[:, AUDIO_OFFSET:].sum())
# A uniform redraw coincides with the original 1/CARD of the time, so the observed CHANGE rate is
# p * (1 - 1/CARD). Tolerance is generous but far tighter than "> 0".
expect = 0.05 * (1.0 - 1.0 / CARD) * n_audio
check(changed > 0, f"corruption actually happened ({changed} tokens changed)")
check(
    abs(changed - expect) / expect < 0.10,
    f"change rate within 10% of requested: {changed} vs ~{expect:.0f} expected",
)

# ---------------------------------------------------------------------------------------------
# 4. the TEXT row is never touched
# ---------------------------------------------------------------------------------------------
check(
    torch.equal(out[:, TEXT_ROW], before[:, TEXT_ROW]),
    "text row untouched (the monologue is where the collapse lives; its vocab differs anyway)",
)

# ---------------------------------------------------------------------------------------------
# 5. BOTH streams are corrupted -- the user half is not silently skipped
# ---------------------------------------------------------------------------------------------
moshi_rows = slice(AUDIO_OFFSET, AUDIO_OFFSET + DEP_Q)
user_rows = slice(AUDIO_OFFSET + DEP_Q, AUDIO_OFFSET + 2 * DEP_Q)
n_moshi = int((out[:, moshi_rows] != before[:, moshi_rows]).sum())
n_user = int((out[:, user_rows] != before[:, user_rows]).sum())
check(n_moshi > 0, f"moshi stream corrupted ({n_moshi} tokens)")
check(n_user > 0, f"user stream corrupted ({n_user} tokens)")
check(
    abs(n_moshi - n_user) / max(1, (n_moshi + n_user) / 2) < 0.15,
    f"both streams at a comparable rate ({n_moshi} vs {n_user})",
)

# ---------------------------------------------------------------------------------------------
# 6. every written value is a VALID code
# ---------------------------------------------------------------------------------------------
aud = out[:, AUDIO_OFFSET:]
check(
    bool(((aud >= 0) & (aud < CARD)).all()),
    f"every audio token stays within [0, {CARD - 1}]",
)

# ---------------------------------------------------------------------------------------------
# 7. SENTINELS are preserved. `codes` carries out-of-range values at delayed/invalid positions
#    (moshi_loss clamps its targets precisely because of that), and the LM's embedding is built as
#    `card + 1` entries, so index CARD is the real "no audio yet" token. Replacing one with a valid
#    code would turn silence into a sound at a position the model expects to be empty.
# ---------------------------------------------------------------------------------------------
c = make_codes(seed=7)
c[:, AUDIO_OFFSET:, :50] = CARD  # the initial/empty token
c[:, AUDIO_OFFSET:, 50:60] = -1  # an out-of-range sentinel
before = c.clone()
out = corrupt_audio_tokens(c, 0.9, cardinality=CARD)  # 0.9 so a leak is unmissable
check(
    torch.equal(out[:, AUDIO_OFFSET:, :50], before[:, AUDIO_OFFSET:, :50]),
    "the `card` initial-token sentinel is preserved even at p=0.9",
)
check(
    torch.equal(out[:, AUDIO_OFFSET:, 50:60], before[:, AUDIO_OFFSET:, 50:60]),
    "out-of-range (-1) sentinels are preserved even at p=0.9",
)
check(
    int((out[:, AUDIO_OFFSET:, 60:] != before[:, AUDIO_OFFSET:, 60:]).sum()) > 0,
    "...while ordinary codes in the same tensor WERE corrupted (not a blanket skip)",
)


# ---------------------------------------------------------------------------------------------
# 8. NON-VACUITY OF THE GUARD ITSELF. A broken in-place implementation must fail check 2 -- if it
#    passes, check 2 is not actually testing anything.
# ---------------------------------------------------------------------------------------------
def corrupt_in_place_BROKEN(codes, prob, *, cardinality):
    """The bug this guard exists to catch: writes through to the caller's tensor."""
    audio = codes[:, AUDIO_OFFSET:]
    hit = torch.rand(audio.shape) < prob
    audio[hit] = torch.randint(0, cardinality, audio.shape, dtype=audio.dtype)[hit]
    return codes


c = make_codes(seed=3)
before = c.clone()
_ = corrupt_in_place_BROKEN(c, 0.05, cardinality=CARD)
check(
    not torch.equal(c, before),
    "the broken in-place version IS caught by the same comparison (guard is non-vacuous)",
)

# ---------------------------------------------------------------------------------------------
# 9. dtype and shape are preserved
# ---------------------------------------------------------------------------------------------
c = make_codes()
out = corrupt_audio_tokens(c, 0.05, cardinality=CARD)
check(out.dtype == c.dtype and out.shape == c.shape, "dtype and shape preserved")

# ---------------------------------------------------------------------------------------------
# 10. THE WIRING, at AST level. The function being correct proves nothing about the caller, and the
#     caller is where this would actually break: the model must get the corrupted copy and the loss
#     the pristine tensor.
# ---------------------------------------------------------------------------------------------
LAUNCHER = SETUP / "recipe/speech_llm/full_duplex/moshi_family/moshi_finetune_launcher.py"
tree = ast.parse(LAUNCHER.read_text())
loss_step = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "loss_step":
        loss_step = node
check(loss_step is not None, "found loss_step in the launcher")

if loss_step is not None:
    lm_args = [
        n.args[0].id
        for n in ast.walk(loss_step)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "lm"
        and n.args
        and isinstance(n.args[0], ast.Name)
    ]
    check(
        bool(lm_args) and all(a == "model_in" for a in lm_args),
        f"every lm(...) in loss_step is called with model_in (got {lm_args})",
    )

    loss_targets = [
        n.args[1].id
        for n in ast.walk(loss_step)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "moshi_loss"
        and len(n.args) >= 2
        and isinstance(n.args[1], ast.Name)
    ]
    check(
        loss_targets == ["codes"],
        f"moshi_loss receives the PRISTINE `codes` as targets (got {loss_targets})",
    )

    src = ast.get_source_segment(LAUNCHER.read_text(), loss_step) or ""
    check(
        "corrupt_audio_tokens(" in src,
        "loss_step calls corrupt_audio_tokens",
    )

# The launcher must read the knob UNCONDITIONALLY -- report_unread_config is fatal, so a
# branch-local read aborts every run that takes the other path.
top = LAUNCHER.read_text()
reads = [ln for ln in top.splitlines() if "token_corrupt_audio_prob" in ln and "cfg.get" in ln]
check(len(reads) == 1, f"exactly one cfg.get for the knob (got {len(reads)})")
check(
    bool(reads) and reads[0].startswith("    token_corrupt_audio_prob"),
    "the cfg.get read is at function top level, not nested in a branch",
)

print()
if failures:
    print(f"{len(failures)} FAILURE(S):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks passed")
