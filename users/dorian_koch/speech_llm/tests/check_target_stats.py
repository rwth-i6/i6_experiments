"""The per-step target-composition stats must be CORRECT, not merely present.

Also (2026-09-22) the per-stream CE (``ce_text``, ``ce_text_word``, ``ce_asst_sem/ac``,
``ce_user_sem/ac``): each equals a reference CE over exactly its codebooks and valid positions, the
objective is unchanged, and a swapped split would be caught.

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

# ---- per-stream CE (2026-09-22): each stream's number is ITS OWN mean CE, and the loss is unchanged ----
# PersonaPlex layout: text + 8 assistant + 8 user codebooks, dep_q=16 (the user stream is a target).
# The assistant's codebooks get logits peaked on the target (CE ~ 0), the user's are uniform
# (CE = log CARD), semantic and acoustic differ in how peaked -- so a split that pooled or swapped
# streams would read visibly wrong.
import math  # noqa: E402

import torch.nn.functional as F  # noqa: E402

NQ, DQ = 16, 16
codes16 = torch.randint(0, CARD, (B, 1 + NQ, T))
codes16[:, TEXT_ROW] = torch.where(torch.rand(B, T) < 0.6, TEXT_PADDING_ID, 7)
o16 = Out()
o16.text_logits = torch.randn(B, 1, T, TCARD)
o16.text_mask = torch.ones(B, 1, T, dtype=torch.bool)
o16.logits = torch.zeros(B, DQ, T, CARD)
tgt = codes16[:, AUDIO_OFFSET : AUDIO_OFFSET + DQ]
o16.logits.scatter_(-1, tgt[..., None], 1.0)
o16.logits[:, 0] *= 8.0  # assistant semantic: very peaked
o16.logits[:, 1:8] *= 4.0  # assistant acoustic: peaked
o16.logits[:, 8:] = 0.0  # user: uniform
o16.mask = torch.ones(B, DQ, T, dtype=torch.bool)
o16.mask[:, :, :2] = False  # delayed/invalid positions are excluded
lm16 = torch.ones(B, T, dtype=torch.bool)
lm16[:, :5] = False  # a masked prompt
kw16 = dict(dep_q=DQ, other_audio_weight=0.02, text_pad_weight=0.3, weighted_mean=True, text_epad_id=TEXT_EPAD_ID)
tot_c, aux16 = duplex_ce_loss(o16, codes16, lm16, return_components=True, **kw16)
tot_p = duplex_ce_loss(o16, codes16, lm16, **kw16)
ck(torch.equal(tot_c, tot_p), "the objective is unchanged by return_components")


def ref_ce(k0, k1):
    ok = o16.mask[:, k0:k1] & lm16[:, None, :]
    ce = F.cross_entropy(o16.logits[:, k0:k1].reshape(-1, CARD), tgt[:, k0:k1].reshape(-1), reduction="none")
    return float(ce[ok.reshape(-1)].mean())


tv = lm16.reshape(-1)
tce = F.cross_entropy(o16.text_logits[:, 0].reshape(-1, TCARD), codes16[:, TEXT_ROW].reshape(-1), reduction="none")
word = codes16[:, TEXT_ROW].reshape(-1) != TEXT_PADDING_ID
want16 = {
    "ce_text": float(tce[tv].mean()),
    "ce_text_word": float(tce[tv & word].mean()),
    "ce_asst_sem": ref_ce(0, 1),
    "ce_asst_ac": ref_ce(1, 8),
    "ce_user_sem": ref_ce(8, 9),
    "ce_user_ac": ref_ce(9, 16),
}
for k, v in want16.items():
    ck(k in aux16 and abs(aux16[k] - v) < 1e-5, f"{k} == reference CE {v:.4f}", f"got {aux16.get(k)}")
ck(abs(aux16["ce_user_sem"] - math.log(CARD)) < 1e-5, "uniform user logits -> ce_user_sem == log(card)")
ck(
    aux16["ce_asst_sem"] < aux16["ce_asst_ac"] < aux16["ce_user_ac"],
    "streams are told apart (asst sem < asst ac < user)",
)
# Non-vacuous: a split that took the user stream from the assistant's codebooks is caught.
ck(abs(aux16["ce_user_ac"] - ref_ce(1, 8)) > 1.0, "a swapped stream split would be caught")
# Base Moshi layout (text + 8 + 8 rows, dep_q=8): no user stream in the loss -> no user keys.
o8 = Out()
o8.text_logits, o8.text_mask = o16.text_logits, o16.text_mask
o8.logits, o8.mask = o16.logits[:, :8], o16.mask[:, :8]
_, aux8 = duplex_ce_loss(o8, codes16, lm16, return_components=True, **dict(kw16, dep_q=8))
ck(
    "ce_asst_sem" in aux8 and "ce_user_sem" not in aux8,
    "dep_q=8: assistant keys only",
    str(sorted(k for k in aux8 if k.startswith("ce_"))),
)
print("\nFAILURES:", fails or "none")
sys.exit(1 if fails else 0)
