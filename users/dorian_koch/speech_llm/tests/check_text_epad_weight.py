"""Guard: ``text_epad_weight`` weights EPAD on its own, and ONLY when an arm sets it.

Added with the knob (2026-09-23) for the PersonaPlex self-play ablation ``ab_epad1`` (EPAD at 1.0, PAD at
the paper's 0.3). Every PersonaPlex run before it weighted EPAD as padding (the reference's
``text_padding_ids``), so the default must reproduce that exactly. Checked:

  * **default = old behaviour** -- ``text_epad_weight=None`` gives the bit-identical loss and components;
  * **it works** -- at 1.0 the text loss equals a hand-computed weighted mean with PAD 0.3, EPAD 1.0,
    words 1.0, and differs materially from the default (non-vacuous: the fixture has EPAD targets);
  * **it is scoped** -- the audio loss and the pad diagnostics (EPAD still counts as padding) are unchanged;
  * **it refuses nonsense** -- a weight without ``text_epad_id`` raises;
  * **the config** -- the PersonaPlex LoRA renderer emits ``text_epad_weight`` only when set (a p3 config
    stays byte-identical), as a YAML float, refuses it without ``emit_epad``, and the launcher reads it.

Fails if the loss ignores the kwarg (check [2]), if the renderer drops it ([5]) or always renders it ([5]).

Run: CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_text_epad_weight.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe/sisyphus"))
sys.path.insert(0, str(SETUP / "recipe/speech_llm/full_duplex"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

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
        torch.where(kind < 0.75, torch.full((B, T), TEXT_EPAD_ID), torch.randint(4, TEXT_CARD, (B, T), generator=g)),
    )
    codes[:, AUDIO_OFFSET:] = torch.randint(0, CARD, (B, 16, T), generator=g)
    out = SimpleNamespace(
        text_logits=torch.randn(B, 1, T, TEXT_CARD, generator=g),
        text_mask=torch.rand(B, 1, T, generator=g) > 0.1,
        logits=torch.randn(B, DEP_Q, T, CARD, generator=g),
        mask=torch.rand(B, DEP_Q, T, generator=g) > 0.1,
    )
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss_mask[0, :3] = False
    return codes, out, loss_mask


def run(**kw):
    codes, out, loss_mask = fixture()
    return duplex_ce_loss(
        out,
        codes,
        loss_mask,
        dep_q=DEP_Q,
        other_audio_weight=AUDIO_OTHER,
        text_pad_weight=TEXT_PAD,
        text_epad_id=TEXT_EPAD_ID,
        weighted_mean=True,
        return_components=True,
        **kw,
    )


print("[1] default (None) = EPAD weighted as PAD, bit for bit")
t_old, c_old = run()
t_none, c_none = run(text_epad_weight=None)
check(torch.equal(t_old, t_none), "total identical", f"{float(t_old)} vs {float(t_none)}")
check(c_old == c_none, "components identical")

print("[2] at 1.0 the text loss is the hand-computed weighted mean (PAD 0.3, EPAD 1.0, word 1.0)")
t_one, c_one = run(text_epad_weight=1.0)
codes, out, loss_mask = fixture()
tgt = codes[:, TEXT_ROW].reshape(-1)
ce = F.cross_entropy(out.text_logits[:, 0].float().reshape(-1, TEXT_CARD), tgt, reduction="none")
valid = (out.text_mask[:, 0] & loss_mask).reshape(-1).float()
w = torch.where(tgt == TEXT_PADDING_ID, TEXT_PAD, 1.0) * valid
want = float((ce * w).sum() / w.sum())
n_epad = int(((tgt == TEXT_EPAD_ID).float() * valid).sum())
check(n_epad >= 5, "the fixture has EPAD targets (non-vacuous)", f"{n_epad}")
check(abs(c_one["text_loss"] - want) < 1e-5, "text_loss matches the hand computation", f"{c_one['text_loss']} vs {want}")
check(abs(c_one["text_loss"] - c_old["text_loss"]) > 1e-3, "and differs from the default", f"{c_one['text_loss']} vs {c_old['text_loss']}")

print("[3] scoped: audio loss and pad diagnostics unchanged")
check(c_one["audio_loss"] == c_old["audio_loss"], "audio_loss identical")
for k in ("tgt_pad_frac", "pred_pad_frac", "pad_excess"):
    check(c_one[k] == c_old[k], f"{k} identical (EPAD still counts as padding)")

print("[4] a weight without text_epad_id raises")
codes, out, loss_mask = fixture()
try:
    duplex_ce_loss(
        out, codes, loss_mask, dep_q=DEP_Q, other_audio_weight=AUDIO_OTHER, text_pad_weight=TEXT_PAD, text_epad_weight=1.0
    )
    check(False, "raised ValueError")
except ValueError:
    check(True, "raised ValueError")

print("[5] renderer + launcher")
from i6_experiments.users.dorian_koch.speech_llm.finetune import PERSONAPLEX_LIB_LORA_ADAPTER as ppx  # noqa: E402
from speech_llm.full_duplex.sis_recipe.doriank.track_persona import AB_ARMS, AB_HPARAMS, P2_HPARAMS  # noqa: E402


class _P:
    def __init__(self, p):
        self.p = p

    def get(self):
        return self.p

    def get_path(self):
        return self.p


def render(hp):
    job = SimpleNamespace(
        train_data=_P("/corpus"), eval_data=None, out_rundir=_P("/tmp/run"), out_config=_P("/tmp/c.yaml"), seed=0,
        duration_sec=60, audio_jitter_sec=2.0, lora_rank=128, hparams=hp, adapter=ppx,
    )  # fmt: skip
    return ppx.render_config(job, ppx.batch_size, 1000)


plain, base = render(P2_HPARAMS), render(AB_HPARAMS)
check("text_epad_weight" not in plain and "text_epad_weight" not in base, "not rendered when unset (p3, ab_base)")
epad = render({**AB_HPARAMS, **AB_ARMS["ab_epad1"]})
cfg = yaml.safe_load(epad)
check(cfg.get("text_epad_weight") == 1.0 and isinstance(cfg["text_epad_weight"], float), "ab_epad1 renders a float 1.0")
check(
    [ln for ln in epad.splitlines() if not ln.startswith("text_epad_weight")] == base.splitlines(),
    "ab_epad1's config = ab_base's + that one line",
)
try:
    render({**AB_HPARAMS, "emit_epad": False, "text_epad_weight": 1.0})
    check(False, "refused without emit_epad")
except ValueError:
    check(True, "refused without emit_epad")
launcher = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/personaplex/finetune_launcher.py").read_text()
check('cfg.get("text_epad_weight")' in launcher, "the PersonaPlex launcher reads text_epad_weight")
check("text_epad_weight=text_epad_weight" in launcher, "...and passes it to the loss")

print()
if FAILURES:
    print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
    sys.exit(1)
print("all checks passed")
