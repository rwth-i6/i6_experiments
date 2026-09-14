"""The 2026-09-14 repairs must be ON for a new run and ABSENT from an older pinned one.

Both halves matter. If the defaults leak into an older epoch, every finished arm re-hashes and
re-trains. If they fail to apply at POLICY_LATEST, a new arm silently reproduces the four defects
we just spent 30 GPU-h measuring.
"""

import sys, os

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
os.environ.setdefault("CUDA_HOME", "/usr")
from speech_llm.full_duplex.sis_recipe.doriank.runs import (
    POLICY_2026_07_31,
    POLICY_2026_08_05,
    POLICY_LATEST,
    RULE_TEXTSTREAM_DEFAULTS,
    _apply_textstream_defaults,
    _policy_applies,
)
from speech_llm.full_duplex.sis_recipe.doriank.train_config import Loss, TextStream

fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{('  ' + extra) if extra else ''}")
    if not c:
        fails.append(label)


class M:
    default_loss = Loss(audio_other=0.02, text_pad=0.3)


print("policy gating")
ck(not _policy_applies(POLICY_2026_07_31, RULE_TEXTSTREAM_DEFAULTS), "2026-07-31 does NOT get the defaults")
ck(not _policy_applies(POLICY_2026_08_05, RULE_TEXTSTREAM_DEFAULTS), "2026-08-05 does NOT get the defaults")
ck(
    _policy_applies(POLICY_LATEST, RULE_TEXTSTREAM_DEFAULTS),
    "POLICY_LATEST DOES get the defaults",
    f"(LATEST={POLICY_LATEST})",
)

print("\nfilling from nothing")
t, l = _apply_textstream_defaults(None, None, M)
ck(
    t.emit_epad is True and t.clamp_text_to_speech is True and t.onset_floor is True,
    "all three text knobs default True",
    str(t),
)
ck(l.norm == "weight", "loss_norm defaults to the paper denominator", str(l))

print("\nan explicit ablation must SURVIVE (that is why they are separate fields)")
t2, _ = _apply_textstream_defaults(TextStream(onset_floor=False), None, M)
ck(t2.onset_floor is False, "explicit False is not overwritten")
ck(t2.emit_epad is True, "...while the unset knobs still fill in")
_, l2 = _apply_textstream_defaults(None, Loss(audio_other=0.02, text_pad=0.3, norm="count"), M)
ck(l2.norm == "count", "an explicit loss_norm is not overwritten")

print("\nlowering: a POLICY_LATEST run must RENDER all four keys")
from speech_llm.full_duplex.sis_recipe.doriank.train_config import merge_hparams

t3, l3 = _apply_textstream_defaults(None, None, M)
hp = merge_hparams(None, l3, None, None, t3, extra=None)
for k, v in (("emit_epad", True), ("clamp_text_to_speech", True), ("onset_floor", True), ("loss_norm", "weight")):
    ck(hp.get(k) == v, f"hparams['{k}'] == {v!r}", f"got {hp.get(k)!r}")

print("\nhash safety: an OLD-epoch run must emit NONE of them")
hp_old = merge_hparams(None, M.default_loss, None, None, None, extra=None)
for k in ("emit_epad", "clamp_text_to_speech", "onset_floor", "loss_norm"):
    ck(k not in hp_old, f"older epoch emits no '{k}' key")

print("\nFAILURES:", fails or "none")
sys.exit(1 if fails else 0)
