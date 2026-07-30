"""Regression guard for the shared training-data primitives.

``moshi_family/train_data_common.py`` consolidated four helpers that had been copy-pasted across
the base-Moshi / PersonaPlex / MoshiRAG training paths. Consolidation is only safe if the surviving
implementation is *numerically identical* to the ones it replaced -- a subtly different loss or
collate would still train, still converge, and be invisible in a loss curve.

So this file keeps the pre-consolidation implementations inlined below as ``_legacy_*`` and asserts
the shared versions agree with them exactly. If you deliberately change the shared behaviour, the
legacy copy here must be updated in the same commit -- which is the point: it forces the change to
be stated rather than absorbed.

Run from the setup root, no GPU needed:
    .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_train_data_common.py
(or the moshi_family venv, which is where these modules actually run)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.getcwd(), "recipe", "speech_llm", "full_duplex"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_PADDING_ID,
    TEXT_ROW,
    collate_codes,
    duplex_ce_loss,
    normalize_alignments,
)


# ---------------------------------------------------------------------------------------------
# Pre-consolidation implementations, verbatim.
# ---------------------------------------------------------------------------------------------
def _legacy_norm_aligns(aligns) -> list[tuple]:
    out = []
    for a in aligns or []:
        if isinstance(a, dict):
            word, s, e, spk = (a.get("text"), a.get("start"), a.get("end"), a.get("speaker", ""))
        else:
            word, span, spk = a[0], a[1], (a[2] if len(a) > 2 else "")
            s, e = span[0], span[1]
        out.append((str(word), (float(s), float(e)), spk))
    return out


def _legacy_collate(samples, pad_text):
    K = samples[0][0].shape[0]
    T = max(c.shape[1] for c, _ in samples)
    B = len(samples)
    codes = torch.zeros(B, K, T, dtype=torch.long)
    codes[:, TEXT_ROW] = pad_text
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, (c, m) in enumerate(samples):
        t = c.shape[1]
        codes[i, :, :t] = c
        mask[i, :t] = m
    return codes, mask


def _legacy_collate_with_ref(samples):
    K = samples[0][0].shape[0]
    dim = samples[0][2].shape[1]
    T = max(c.shape[1] for c, _, _ in samples)
    B = len(samples)
    codes = torch.zeros(B, K, T, dtype=torch.long)
    codes[:, TEXT_ROW] = TEXT_PADDING_ID
    mask = torch.zeros(B, T, dtype=torch.bool)
    ref = torch.zeros(B, T, dim)
    for i, (c, m, r) in enumerate(samples):
        t = c.shape[1]
        codes[i, :, :t] = c
        mask[i, :t] = m
        ref[i, :t] = r
    return codes, mask, ref


def _legacy_moshi_loss(
    out,
    codes,
    loss_mask,
    *,
    dep_q,
    first_codebook_weight=1.0,
    audio_other_weight=0.01,
    text_pad_weight=0.5,
    text_pad_id=TEXT_PADDING_ID,
):
    import torch.nn.functional as F

    B, _K, T = codes.shape
    fm = loss_mask.to(codes.device)
    text_logits = torch.nan_to_num(out.text_logits[:, 0])
    text_card = text_logits.shape[-1]
    text_tgt = codes[:, TEXT_ROW]
    tvalid = out.text_mask[:, 0] & fm
    tt = text_tgt.reshape(-1).clamp(0, text_card - 1)
    tl = F.cross_entropy(text_logits.reshape(-1, text_card), tt, reduction="none")
    tw = torch.where(text_tgt.reshape(-1) == text_pad_id, text_pad_weight, 1.0)
    tl = (tl * tw * tvalid.reshape(-1).float()).sum() / tvalid.sum().clamp(min=1)
    audio_logits = torch.nan_to_num(out.logits)
    card = audio_logits.shape[-1]
    audio_tgt = codes[:, AUDIO_OFFSET : AUDIO_OFFSET + dep_q]
    avalid = out.mask[:, :dep_q] & fm[:, None, :]
    al = F.cross_entropy(
        audio_logits[:, :dep_q].reshape(-1, card),
        audio_tgt.reshape(-1).clamp(0, card - 1),
        reduction="none",
    ).view(B, dep_q, T)
    cb_w = torch.full((dep_q,), audio_other_weight, device=audio_logits.device)
    cb_w[0] = first_codebook_weight
    al = (al * cb_w[None, :, None] * avalid.float()).sum() / avalid.float().sum().clamp(min=1)
    return tl + al


class _FakeLMOutput:
    """Stand-in for the model's LMOutput, with the NaN-filled invalid positions the real one has."""

    def __init__(self, rng, B, K, T, card, text_card, dep_q):
        self.logits = torch.from_numpy(rng.normal(size=(B, K, T, card))).float()
        self.text_logits = torch.from_numpy(rng.normal(size=(B, 1, T, text_card))).float()
        self.mask = torch.from_numpy(rng.random((B, K, T)) > 0.2)
        self.text_mask = torch.from_numpy(rng.random((B, 1, T)) > 0.2)
        # The real forward leaves NaN wherever a position is invalid/delayed.
        self.logits[~self.mask] = float("nan")
        self.text_logits[~self.text_mask] = float("nan")
        self.dep_q = dep_q


# ---------------------------------------------------------------------------------------------
# 1. normalize_alignments: both on-disk spellings + the empty case
# ---------------------------------------------------------------------------------------------
align_cases = [
    [],
    None,
    [{"text": "hello", "start": 0.0, "end": 0.4, "speaker": "assistant"}],
    [{"text": "no", "start": 1.0, "end": 1.2}],  # speaker absent
    [("legacy", (2.0, 2.5), "user"), ("tuple", (2.5, 2.9))],  # positional, 2- and 3-tuple
    [{"text": 42, "start": "3.0", "end": "3.5", "speaker": ""}],  # coercion
]
for case in align_cases:
    assert normalize_alignments(case) == _legacy_norm_aligns(case), case
print(f"[ok] normalize_alignments matches legacy on {len(align_cases)} cases")

# ---------------------------------------------------------------------------------------------
# 2. collate_codes: 2-tensor (Moshi/PersonaPlex) and 3-tensor (MoshiRAG) forms
# ---------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
K, dim = 17, 8
for trial in range(20):
    lengths = [int(rng.integers(1, 40)) for _ in range(int(rng.integers(1, 6)))]
    samples2 = [
        (
            torch.from_numpy(rng.integers(0, 2000, size=(K, t))).long(),
            torch.from_numpy(rng.random(t) > 0.3),
        )
        for t in lengths
    ]
    pad = int(rng.integers(0, 5))
    got_c, got_m = collate_codes(samples2, pad_text=pad)
    exp_c, exp_m = _legacy_collate(samples2, pad)
    assert torch.equal(got_c, exp_c) and torch.equal(got_m, exp_m), f"2-tensor collate, trial {trial}"

    samples3 = [(c, m, torch.from_numpy(rng.normal(size=(c.shape[1], dim))).float()) for c, m in samples2]
    got = collate_codes(samples3)
    exp = _legacy_collate_with_ref(samples3)
    assert len(got) == 3
    for g, e in zip(got, exp):
        assert torch.equal(g, e), f"3-tensor collate, trial {trial}"
print("[ok] collate_codes matches legacy over 20 random batches (2- and 3-tensor forms)")

# default pad_text must be the text padding id, matching the RAG collate that hardcoded it
c0, _ = collate_codes(
    [
        (torch.zeros(K, 3, dtype=torch.long), torch.ones(3, dtype=torch.bool)),
        (torch.zeros(K, 5, dtype=torch.long), torch.ones(5, dtype=torch.bool)),
    ]
)
assert (c0[0, TEXT_ROW, 3:] == TEXT_PADDING_ID).all(), "text row must pad with TEXT_PADDING_ID"
assert (c0[0, AUDIO_OFFSET:, 3:] == 0).all(), "audio rows must pad with 0"
print("[ok] collate_codes pads text row with TEXT_PADDING_ID and audio rows with 0")

# ---------------------------------------------------------------------------------------------
# 3. duplex_ce_loss vs the three legacy losses (they differed only in default weights)
# ---------------------------------------------------------------------------------------------
B, T, dep_q, card, text_card = 3, 24, 8, 2048, 32000
for weights in [
    dict(other_audio_weight=0.01, text_pad_weight=0.5),  # base Moshi (fork parity)
    dict(other_audio_weight=0.02, text_pad_weight=0.3),  # PersonaPlex / MoshiRAG (paper)
    dict(other_audio_weight=0.5, text_pad_weight=1.0),  # extremes
    dict(other_audio_weight=0.0, text_pad_weight=0.0),
]:
    r = np.random.default_rng(7)
    out = _FakeLMOutput(r, B, K, T, card, text_card, dep_q)
    codes = torch.from_numpy(r.integers(0, card, size=(B, K, T))).long()
    codes[:, TEXT_ROW] = torch.from_numpy(r.integers(0, text_card, size=(B, T))).long()
    codes[:, TEXT_ROW][r.random((B, T)) < 0.6] = TEXT_PADDING_ID  # monologue is mostly padding
    loss_mask = torch.from_numpy(r.random((B, T)) > 0.1)

    got = duplex_ce_loss(out, codes, loss_mask, dep_q=dep_q, **weights)
    exp = _legacy_moshi_loss(
        out,
        codes,
        loss_mask,
        dep_q=dep_q,
        audio_other_weight=weights["other_audio_weight"],
        text_pad_weight=weights["text_pad_weight"],
    )
    assert torch.isfinite(got), f"loss is not finite at {weights}"
    assert torch.equal(got, exp), f"loss mismatch at {weights}: {got.item()} vs {exp.item()}"
print("[ok] duplex_ce_loss bit-identical to the legacy loss at 4 weight settings")

# the loss must actually respond to its weights (guards against a dropped kwarg)
r = np.random.default_rng(11)
out = _FakeLMOutput(r, B, K, T, card, text_card, dep_q)
codes = torch.from_numpy(r.integers(0, card, size=(B, K, T))).long()
codes[:, TEXT_ROW] = TEXT_PADDING_ID
loss_mask = torch.ones(B, T, dtype=torch.bool)
lo = duplex_ce_loss(out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.01, text_pad_weight=0.1)
hi = duplex_ce_loss(out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.01, text_pad_weight=0.9)
assert hi > lo, f"text_pad_weight has no effect ({lo.item()} vs {hi.item()})"
lo = duplex_ce_loss(out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.01, text_pad_weight=0.5)
hi = duplex_ce_loss(out, codes, loss_mask, dep_q=dep_q, other_audio_weight=0.5, text_pad_weight=0.5)
assert hi > lo, f"other_audio_weight has no effect ({lo.item()} vs {hi.item()})"
print("[ok] both loss weights demonstrably reach the computation")

# ---------------------------------------------------------------------------------------------
# 4. The three architecture modules still export what their launchers import.
# ---------------------------------------------------------------------------------------------
import importlib  # noqa: E402

expected_exports = {
    "moshi_family.moshi_train_data": [
        "AUDIO_OFFSET",
        "TEXT_PADDING_ID",
        "TEXT_ROW",
        "MoshiDataConfig",
        "MoshiTokenizer",
        "build_data_loader",
        "build_mixed_data_loader",
        "moshi_loss",
    ],
    "moshi_family.personaplex.train_data": [
        "PersonaPlexDataConfig",
        "PersonaPlexTokenizer",
        "build_data_loader",
        "personaplex_loss",
    ],
    "moshi_family.moshirag_train_data": [
        "MoshiRagDataConfig",
        "MoshiRagTokenizer",
        "build_data_loader",
        "moshirag_loss",
        "RAG_TOKEN_ID",
    ],
}
for module_name, names in expected_exports.items():
    module = importlib.import_module(module_name)
    missing = [n for n in names if not hasattr(module, n)]
    assert not missing, f"{module_name} no longer exports {missing} (its launcher imports these)"
    print(f"[ok] {module_name} exports all {len(names)} names its launcher imports")

# the constants must be the same objects everywhere, not re-declared per module
import moshi_family.moshi_train_data as m  # noqa: E402
import moshi_family.moshirag_train_data as g  # noqa: E402
import moshi_family.personaplex.train_data as p  # noqa: E402

assert m.TEXT_PADDING_ID == p.TEXT_PADDING_ID == g.TEXT_PADDING_ID == TEXT_PADDING_ID == 3
assert m.TEXT_ROW == p.TEXT_ROW == g.TEXT_ROW == TEXT_ROW == 0
assert m.AUDIO_OFFSET == p.AUDIO_OFFSET == g.AUDIO_OFFSET == AUDIO_OFFSET == 1
print("[ok] codes-layout constants agree across all three architectures")

print("\nALL CHECKS PASSED")
