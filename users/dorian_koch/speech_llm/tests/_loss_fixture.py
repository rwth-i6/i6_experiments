"""The duplex-loss fixture shared by ``check_loss_parity``, ``check_text_loss_weight`` and ``check_text_epad_weight``.

One copy so a change to the A-series weights or the masking plant reaches all three guards (one copy had already lost
the NaN plant).
"""

from types import SimpleNamespace

import torch
from moshi_family.train_data_common import AUDIO_OFFSET, TEXT_EPAD_ID, TEXT_PADDING_ID, TEXT_ROW

B, T, DEP_Q, CARD, TEXT_CARD = 2, 37, 8, 2048, 32000
AUDIO_OTHER, TEXT_PAD = 0.02, 0.3  # the PersonaPlex weights every A-series arm trains with


def loss_fixture(seed: int = 0, *, epad_until: float = 0.68):
    """``(codes, out, loss_mask)``: a text row of ~60% padding, EPAD up to ``epad_until``, the rest real tokens."""
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(B, 1 + 16, T, dtype=torch.long)
    kind = torch.rand(B, T, generator=g)
    codes[:, TEXT_ROW] = torch.where(
        kind < 0.60,
        torch.full((B, T), TEXT_PADDING_ID),
        torch.where(
            kind < epad_until,
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
    # the reference fills invalid positions with anything; ours nan_to_nums -- plant NaNs to prove no masked
    # position leaks into the loss
    out.logits[~out.mask] = float("nan")
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss_mask[0, :3] = False  # a prompt prefix, PersonaPlex-style
    return codes, out, loss_mask
