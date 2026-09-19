"""Guard: ``text_loss_weight`` really removes the text term, and only the text term.

Added with the knob (2026-09-19) for the A48 arm, which trains the AUDIO stream only. Three things
have to hold, and each is a way the knob could be quietly wrong:

  * **it works** -- at 0.0 the objective equals the audio term alone, exactly;
  * **it is scoped** -- the audio term is bit-identical to a normal run, so an "audio-only" arm is
    genuinely the same audio objective and not a rescaled one;
  * **the DIAGNOSTIC survives** -- the reported ``text_loss`` component stays UNSCALED. This is the
    one that would hurt most if wrong: `text_loss` is the curve every A-arm is read against, and
    multiplying it by 0 would make an untrained text stream render as a perfect one. Today's
    readout already showed that a loss curve falling smoothly while the model learns to say nothing
    is the characteristic failure here.

Non-vacuous throughout: the text term is asserted to be materially non-zero first, so "removing it"
cannot pass by the term having been negligible anyway.

Run: CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_text_loss_weight.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe/speech_llm/full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    TEXT_ROW,
    duplex_ce_loss,
)

B, T, DEP_Q, CARD, TEXT_CARD = 2, 37, 8, 2048, 32000
AUDIO_OTHER, TEXT_PAD = 0.02, 0.3

FAILURES: list[str] = []


def check(cond, msg, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {msg} {detail if not cond else ''}")
    if not cond:
        FAILURES.append(msg)


def fixture(seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(B, 1 + 16, T, dtype=torch.long)
    kind = torch.rand(B, T, generator=g)
    codes[:, TEXT_ROW] = torch.where(
        kind < 0.60,
        torch.full((B, T), TEXT_PADDING_ID),
        torch.where(
            kind < 0.68,
            torch.full((B, T), TEXT_EPAD_ID),
            torch.randint(4, TEXT_CARD, (B, T), generator=g),
        ),
    )
    codes[:, AUDIO_OFFSET:] = torch.randint(0, CARD, (B, 16, T), generator=g)
    out = SimpleNamespace(
        text_logits=torch.randn(B, 1, T, TEXT_CARD, generator=g),
        text_mask=torch.rand(B, 1, T, generator=g) > 0.1,
        logits=torch.randn(B, DEP_Q, T, CARD, generator=g),
        mask=torch.rand(B, DEP_Q, T, generator=g) > 0.1,
    )
    out.logits[~out.mask] = float("nan")
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss_mask[0, :3] = False
    return codes, out, loss_mask


def run(w):
    codes, out, loss_mask = fixture()
    return duplex_ce_loss(
        out,
        codes,
        loss_mask,
        dep_q=DEP_Q,
        other_audio_weight=AUDIO_OTHER,
        text_pad_weight=TEXT_PAD,
        text_loss_weight=w,
        weighted_mean=True,
        return_components=True,
    )


print("[1] the knob removes the text term")
full = run(1.0)
none = run(0.0)
t_full, c_full = (full if isinstance(full, tuple) else (full, {}))[:2]
t_none, c_none = (none if isinstance(none, tuple) else (none, {}))[:2]
text_full = float(c_full.get("text_loss", float("nan")))
audio_full = float(c_full.get("audio_loss", float("nan")))
text_none = float(c_none.get("text_loss", float("nan")))
audio_none = float(c_none.get("audio_loss", float("nan")))

# Non-vacuity FIRST: if the text term were ~0 anyway, every assertion below would pass for free.
check(text_full > 0.5, "the text term is materially non-zero to begin with", f"{text_full:.4f}")
check(
    abs(float(t_full) - (text_full + audio_full)) < 1e-4,
    "at 1.0 the objective is text + audio",
    f"{float(t_full):.5f} vs {text_full + audio_full:.5f}",
)
check(
    abs(float(t_none) - audio_none) < 1e-6,
    "at 0.0 the objective IS the audio term exactly",
    f"{float(t_none):.6f} vs {audio_none:.6f}",
)
check(
    abs(float(t_full) - float(t_none)) > 0.5,
    "the two objectives really differ (non-vacuity)",
    f"{float(t_full):.4f} vs {float(t_none):.4f}",
)

print("[2] it is SCOPED -- the audio objective is untouched")
check(
    abs(audio_full - audio_none) < 1e-6,
    "audio_loss is identical at both weights",
    f"{audio_full:.6f} vs {audio_none:.6f}",
)

print("[3] the reported text_loss stays UNSCALED (it is the diagnostic)")
check(
    abs(text_full - text_none) < 1e-6,
    "text_loss is reported unscaled at 0.0, not zeroed",
    f"{text_full:.6f} vs {text_none:.6f}",
)
check(text_none > 0.5, "...and is still a real number to watch", f"{text_none:.4f}")

print("[4] a half weight scales linearly, so the knob is a weight and not a switch")
half = run(0.5)
t_half, c_half = (half if isinstance(half, tuple) else (half, {}))[:2]
want = 0.5 * text_full + audio_full
check(
    abs(float(t_half) - want) < 1e-4, "at 0.5 the objective is 0.5*text + audio", f"{float(t_half):.5f} vs {want:.5f}"
)

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
