"""Guard: our duplex loss, at ``weighted_mean=True``, equals the reference ``compute_loss_with_mask``
exactly -- and at the default it does NOT, by a measured, stream-dependent factor.

The deviation this exists for (paper audit, 2026-09-09): ``duplex_ce_loss`` has always divided each
half by the NUMBER of valid positions, while the Moshi paper's eq. 7 and the reference fork
(``moshi_finetune/finetune/loss.py``: ``sum(loss * weights) / sum(weights)``) divide by the SUM OF
THE WEIGHTS. Every finished run's docstring claimed "moshi-finetune parity" for the weights and
nobody looked at the denominator. The consequence is not a global scale (Adam removes that) but a
different constant per half: the audio term comes out ~7.5x smaller than the reference's and the
text term ~0.68x, so the text stream weighs ~5x more against the audio than the paper's "same
importance to the text token and the combined audio tokens".

Drives the REAL reference function on the same random logits/targets/masks, including the EPAD id
in the padding set as the reference's ``train.py`` passes it, and asserts:
  * exact equality at ``weighted_mean=True`` for both halves and the total;
  * inequality at the default, with the ratio in the range the arithmetic above predicts -- so the
    guard cannot pass by the two paths sharing a bug;
  * the launcher reads the knob unconditionally and every ``moshi_loss`` call site forwards it.

Run from the setup root:
    CUDA_HOME=/usr .venv/bin/python recipe/i6_experiments/users/dorian_koch/speech_llm/tests/check_loss_parity.py
"""

import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

SETUP = next(p for p in Path(__file__).absolute().parents if (p / "recipe").is_dir())
sys.path.insert(0, str(SETUP / "recipe"))
sys.path.insert(0, str(SETUP / "recipe" / "speech_llm" / "full_duplex"))
sys.path.insert(0, str(SETUP / "recipe" / "sisyphus"))
os.environ.setdefault("CUDA_HOME", "/usr")

import torch  # noqa: E402

from moshi_family.train_data_common import (  # noqa: E402
    AUDIO_OFFSET,
    TEXT_EPAD_ID,
    TEXT_PADDING_ID,
    TEXT_ROW,
    duplex_ce_loss,
)
from moshi_finetune.finetune.loss import compute_loss_with_mask  # noqa: E402

B, T, DEP_Q, CARD, TEXT_CARD = 2, 37, 8, 2048, 32000
AUDIO_OTHER, TEXT_PAD = 0.02, 0.3  # the PersonaPlex weights every A-series arm trains with


def _fixture(seed: int):
    g = torch.Generator().manual_seed(seed)
    codes = torch.zeros(B, 1 + 16, T, dtype=torch.long)
    # text row: ~65% padding, a few EPADs, the rest real tokens -- the paper's conversational mix
    kind = torch.rand(B, T, generator=g)
    codes[:, TEXT_ROW] = torch.where(
        kind < 0.60,
        torch.full((B, T), TEXT_PADDING_ID),
        torch.where(kind < 0.68, torch.full((B, T), TEXT_EPAD_ID), torch.randint(4, TEXT_CARD, (B, T), generator=g)),
    )
    codes[:, AUDIO_OFFSET:] = torch.randint(0, CARD, (B, 16, T), generator=g)
    out = SimpleNamespace(
        text_logits=torch.randn(B, 1, T, TEXT_CARD, generator=g),
        text_mask=torch.rand(B, 1, T, generator=g) > 0.1,
        logits=torch.randn(B, DEP_Q, T, CARD, generator=g),
        mask=torch.rand(B, DEP_Q, T, generator=g) > 0.1,
    )
    # the reference fills invalid positions with anything; ours nan_to_nums -- plant NaNs to prove
    # neither path lets a masked position leak
    out.logits[~out.mask] = float("nan")
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    loss_mask[0, :3] = False  # a prompt prefix, PersonaPlex-style
    return codes, out, loss_mask


def _reference(codes, out, loss_mask):
    """The reference fork's two calls, as ``train.py`` makes them, with our frame mask folded into
    its target masks (the fork has no separate per-frame mask)."""
    text_mask = out.text_mask & loss_mask[:, None, :]
    audio_mask = out.mask & loss_mask[:, None, :]
    text = compute_loss_with_mask(
        torch.nan_to_num(out.text_logits),
        codes[:, :AUDIO_OFFSET],
        text_mask,
        mode="text",
        text_padding_weight=TEXT_PAD,
        text_padding_ids={TEXT_PADDING_ID, TEXT_EPAD_ID},
    )
    # first_codebook_weight_multiplier=100 is the fork's 1.0 : 0.01; the A-series 0.02 is 1/0.02
    audio = compute_loss_with_mask(
        torch.nan_to_num(out.logits),
        codes[:, AUDIO_OFFSET : AUDIO_OFFSET + DEP_Q],
        audio_mask,
        mode="audio",
        first_codebook_weight_multiplier=1.0 / AUDIO_OTHER,
    )
    return text, audio


def _ours(codes, out, loss_mask, weighted_mean):
    return duplex_ce_loss(
        out,
        codes,
        loss_mask,
        dep_q=DEP_Q,
        other_audio_weight=AUDIO_OTHER,
        text_pad_weight=TEXT_PAD,
        text_epad_id=TEXT_EPAD_ID,
        weighted_mean=weighted_mean,
        return_components=True,
    )


def check_weighted_mean_matches_reference():
    for seed in (0, 1, 2):
        codes, out, loss_mask = _fixture(seed)
        ref_text, ref_audio = _reference(codes, out, loss_mask)
        total, parts = _ours(codes, out, loss_mask, weighted_mean=True)
        assert torch.isfinite(total), total
        torch.testing.assert_close(torch.tensor(parts["text_loss"]), ref_text, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.tensor(parts["audio_loss"]), ref_audio, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(total, ref_text + ref_audio, rtol=1e-5, atol=1e-6)
    print("PASS  weighted_mean=True == the reference compute_loss_with_mask, both halves, 3 seeds")


def check_default_differs_by_the_predicted_factor():
    """The scaling under the old normalisation, per half: audio ~ (1 + 0.02*7)/8 = 0.1425 of the
    reference (exactly sum(w)/count over valid positions), text ~ (0.35 + 0.65*0.3) of it. The
    ratio between the two halves is what changes the training signal; assert it."""
    codes, out, loss_mask = _fixture(3)
    ref_text, ref_audio = _reference(codes, out, loss_mask)
    _, ours = _ours(codes, out, loss_mask, weighted_mean=False)
    audio_ratio = ours["audio_loss"] / float(ref_audio)
    text_ratio = ours["text_loss"] / float(ref_text)
    assert 0.12 < audio_ratio < 0.17, audio_ratio  # the semantic weight over 8 codebooks
    assert 0.45 < text_ratio < 0.85, text_ratio  # padding share x its weight
    imbalance = text_ratio / audio_ratio
    assert imbalance > 3.0, imbalance
    print(
        f"PASS  the default is NOT the reference: audio x{audio_ratio:.3f}, text x{text_ratio:.3f} -> "
        f"text weighs {imbalance:.1f}x more against audio than the paper intends"
    )


def check_masked_positions_do_not_leak():
    """Under both normalisations a NaN logit at a masked position must not reach the loss."""
    codes, out, loss_mask = _fixture(4)
    for wm in (False, True):
        total, _ = _ours(codes, out, loss_mask, weighted_mean=wm)
        assert torch.isfinite(total), (wm, total)
    print("PASS  masked positions never leak, under either denominator")


def check_wiring():
    launcher = (SETUP / "recipe/speech_llm/full_duplex/moshi_family/moshi_finetune_launcher.py").read_text()
    assert re.search(r'^\s*loss_norm = str\(cfg\.get\("loss_norm", "count"\)\)', launcher, re.M), (
        "the launcher must read loss_norm unconditionally (report_unread_config is strict)"
    )
    calls = launcher.count("moshi_loss(")
    forwards = launcher.count("weighted_mean=weighted_mean")
    assert calls >= 3 and forwards == calls, f"{calls} moshi_loss call sites, {forwards} forward the knob"
    template = (SETUP / "recipe/i6_experiments/users/dorian_koch/speech_llm/finetune.py").read_text()
    assert 'loss_norm: {hp.get("loss_norm", "count")}' in template, "the template must render loss_norm"
    from speech_llm.full_duplex.sis_recipe.doriank.train_config import MOSHI_WEIGHTS, PERSONAPLEX_WEIGHTS, Loss

    assert "loss_norm" not in MOSHI_WEIGHTS.to_hparams() and "loss_norm" not in PERSONAPLEX_WEIGHTS.to_hparams(), (
        "the presets must not emit loss_norm, or every finished run re-hashes"
    )
    assert Loss(norm="weight").to_hparams() == {"loss_norm": "weight"}
    print("PASS  launcher reads + forwards loss_norm at every call site; presets stay hash-stable")


if __name__ == "__main__":
    check_weighted_mean_matches_reference()
    check_default_differs_by_the_predicted_factor()
    check_masked_positions_do_not_leak()
    check_wiring()
    print("ALL PASS")
