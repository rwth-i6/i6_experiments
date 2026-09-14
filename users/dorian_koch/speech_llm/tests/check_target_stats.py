"""The per-step target-composition stats must be CORRECT, not merely present.

A logged ratio that is wrong is worse than none: it would be used to decide whether a run is
learning the task or learning to be silent.
"""

import sys, os

sys.path.insert(0, "recipe")
sys.path.insert(0, "recipe/sisyphus")
sys.path.insert(0, "recipe/speech_llm/full_duplex")
os.environ.setdefault("CUDA_HOME", "/usr")
import torch
from moshi_family.train_data_common import (
    duplex_ce_loss,
    TEXT_PADDING_ID,
    TEXT_EPAD_ID,
    TEXT_ROW,
    AUDIO_OFFSET,
)

fails = []


def ck(c, label, extra=""):
    print(f"  [{'ok' if c else 'FAIL'}] {label}{'  ' + extra if extra else ''}")
    if not c:
        fails.append(label)


B, T, DEP_Q, CARD, TCARD = 2, 40, 4, 32, 64
torch.manual_seed(0)


class Out:
    pass


def build(pad_frac):
    """codes whose text row is `pad_frac` padding, rest real word ids."""
    codes = torch.randint(1, CARD, (B, 1 + 1 + DEP_Q, T))
    n_pad = int(round(pad_frac * T))
    for b in range(B):
        row = torch.full((T,), 5000 % TCARD, dtype=torch.long)  # a "word" id
        row[:n_pad] = TEXT_PADDING_ID
        codes[b, TEXT_ROW] = row
    o = Out()
    o.text_logits = torch.randn(B, 1, T, TCARD)
    o.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
    o.logits = torch.randn(B, DEP_Q, T, CARD)
    o.mask = torch.ones(B, DEP_Q, T, dtype=torch.bool)
    return o, codes


for frac in (0.0, 0.5, 0.9):
    o, codes = build(frac)
    _, aux = duplex_ce_loss(
        o,
        codes,
        torch.ones(B, T, dtype=torch.bool),
        dep_q=DEP_Q,
        other_audio_weight=0.02,
        text_pad_weight=0.3,
        return_components=True,
    )
    ck(abs(aux["tgt_pad_frac"] - frac) < 1e-6, f"tgt_pad_frac == {frac}", f"got {aux['tgt_pad_frac']:.4f}")
    ck(abs(aux["tgt_word_frac"] - (1 - frac)) < 1e-6, f"tgt_word_frac == {1 - frac}", f"got {aux['tgt_word_frac']:.4f}")

# the loss-share must respond to the WEIGHT, not just the target mix -- that is its whole point
o, codes = build(0.9)
shares = {}
for w in (0.3, 0.1, 0.01):
    _, aux = duplex_ce_loss(
        o,
        codes,
        torch.ones(B, T, dtype=torch.bool),
        dep_q=DEP_Q,
        other_audio_weight=0.02,
        text_pad_weight=w,
        return_components=True,
    )
    shares[w] = aux["text_pad_loss_share"]
print(f"  pad loss share at 90% padding: { ({k: round(v, 3) for k, v in shares.items()}) }")
ck(shares[0.3] > shares[0.1] > shares[0.01], "loss share falls as text_pad_weight falls")
ck(shares[0.3] > 0.5, "at w=0.3 and 90% padding, padding carries most of the text loss", f"{shares[0.3]:.3f}")
# pred_pad_frac must track what the MODEL says, not the target -- force an all-pad prediction
o2, c2 = build(0.5)
o2.text_logits = torch.full((B, T, TCARD), -10.0).unsqueeze(1).squeeze(1)
o2.text_logits = torch.full((B, 1, T, TCARD), -10.0)[:, 0]
o2.text_logits[..., TEXT_PADDING_ID] = 10.0
o2.text_logits = o2.text_logits.unsqueeze(1)
_, aux2 = duplex_ce_loss(
    o2,
    c2,
    torch.ones(B, T, dtype=torch.bool),
    dep_q=DEP_Q,
    other_audio_weight=0.02,
    text_pad_weight=0.3,
    return_components=True,
)
ck(
    abs(aux2["pred_pad_frac"] - 1.0) < 1e-6,
    "a model predicting only PAD reports pred_pad_frac 1.0",
    f"got {aux2['pred_pad_frac']:.3f}",
)
ck(abs(aux2["pad_excess"] - 0.5) < 1e-6, "pad_excess == pred - tgt (1.0 - 0.5)", f"got {aux2['pad_excess']:.3f}")
# ...and a model predicting only WORDS must report ~0, so the metric is not a constant
o3, c3 = build(0.5)
o3.text_logits = torch.full((B, 1, T, TCARD), -10.0)
o3.text_logits[..., 5000 % TCARD] = 10.0
_, aux3 = duplex_ce_loss(
    o3,
    c3,
    torch.ones(B, T, dtype=torch.bool),
    dep_q=DEP_Q,
    other_audio_weight=0.02,
    text_pad_weight=0.3,
    return_components=True,
)
ck(aux3["pred_pad_frac"] == 0.0, "a model predicting no PAD reports 0.0", f"got {aux3['pred_pad_frac']:.3f}")

# and a corpus with NO padding must report a zero share
o0, c0 = build(0.0)
_, aux0 = duplex_ce_loss(
    o0,
    c0,
    torch.ones(B, T, dtype=torch.bool),
    dep_q=DEP_Q,
    other_audio_weight=0.02,
    text_pad_weight=0.3,
    return_components=True,
)
ck(aux0["text_pad_loss_share"] == 0.0, "no padding -> zero pad loss share")
print("\nFAILURES:", fails or "none")
sys.exit(1 if fails else 0)
