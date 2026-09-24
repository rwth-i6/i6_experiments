"""Ported from speech-llm c49559ce src/speech_llm/prefix_lm/model/train_steps/sae_blankfree_supervised.py

ANALYSIS ONLY (uses transcripts).  The SUPERVISED blank-free step of the p0 recognizer
(``p0.py``): ``-log p(gold phone string)`` under the bed's own topology.  No main-line arm is
initialised from a checkpoint it produces.

THE LOSS is ``model.blankfree_seed.transcript_logprob`` -- the bed's own exact blank-free likelihood
of a token string, "sum paths whose adjacent-run collapse equals each target, without a blank".  The
bed's minimum duration ``d_min = 2`` lives on the 50 Hz unit clock and the recognizer's
``stride = 3`` already absorbs it, so on the RECOGNIZER axis the bed's min-duration topology is the
plain run-collapse topology this function sums.

TEMPERATURE.  ``transcript_logprob`` takes a normalised per-frame log-posterior and no temperature,
so the step runs it on the recognizer's OWN untempered log-probs; the arm's
``temperature_schedule`` is inert for this loss.

INFEASIBLE UTTERANCES.  A gold string with more (run-collapsed) tokens than the utterance has
recognizer output frames has no path.  Such a row is DROPPED (zero weight, zero gradient, kept out
of the normaliser) and COUNTED into ``blankfree_supervised_infeasible_utts``.

The reported ``blankfree_supervised`` loss is the mean NLL per output frame over the feasible rows;
RETURNN runs this step on the ``dev`` dataset too, so ``dev_loss_blankfree_supervised`` is the
held-out loss the checkpoint selection reads.
"""

from __future__ import annotations

import torch
import returnn.frontend as rf

from returnn.tensor import Tensor as ReturnnTensor

from ..model.blankfree_seed import transcript_logprob
from ..model.train_step import seq_lens

__all__ = ["TARGETS_KEY", "LOSS_NAME", "INFEASIBLE_NAME", "train_step"]

#: the extern_data key of the run-collapsed gold phone ids (``p0.BlankfreeSeedSupportJob``'s
#: ``collapsed_targets.hdf``: 0-based ids into ``phones.PHONES``, adjacent duplicates merged)
TARGETS_KEY = "targets"
#: the RETURNN loss name; the held-out column is ``dev_loss_<LOSS_NAME>``
LOSS_NAME = "blankfree_supervised"
#: the error column carrying the number of dropped (path-less) utterances per batch
INFEASIBLE_NAME = "blankfree_supervised_infeasible_utts"


def train_step(*, model, extern_data, **_kwargs) -> None:
    """``-log p_theta(gold | x)`` on the recognizer of ``SaeBlankfreeModelV1``.

    Only ``model.recognizer`` is touched: every other submodule is built by the model exactly as
    the arm's control builds it and receives no gradient from this step.
    """
    ctx = rf.get_run_ctx()
    feats_ = extern_data[model.features_key]
    targets_ = extern_data[TARGETS_KEY]
    feats = feats_.raw_tensor.float()
    lengths = seq_lens(feats_)
    targets = targets_.raw_tensor.long()
    target_lens = seq_lens(targets_)

    log_q = model.recognizer(feats, lengths)
    out_lens = model.recognizer.output_lengths(lengths).long()
    # the bed's own stride-3 clock check; no host sync
    assert log_q.shape[1] == (feats.shape[1] + 2) // 3, (log_q.shape, feats.shape)

    # A row with no path at all (more collapsed tokens than output frames) is dropped, not scored.
    feasible = (target_lens >= 1) & (out_lens >= target_lens)
    # Kept inside the DP's own preconditions; an infeasible row's VALUE is discarded by the weight.
    dp_target_lens = torch.minimum(target_lens, out_lens).clamp(min=1)
    dp_out_lens = out_lens.clamp(min=1)
    per_utt = -transcript_logprob(log_q, targets, dp_target_lens, dp_out_lens)

    weight = feasible.to(per_utt.dtype)
    loss = per_utt * weight
    # The normaliser counts the FEASIBLE rows' output frames only.
    inv_norm = ReturnnTensor(
        "out_lens", dims=[feats_.dims[0]], dtype="int32",
        raw_tensor=(out_lens * feasible.long()).to(device="cpu", dtype=torch.int32),
    )
    ctx.mark_as_loss(loss, LOSS_NAME, custom_inv_norm_factor=inv_norm, use_normalized_loss=True)
    ctx.mark_as_loss((~feasible).to(torch.float32), INFEASIBLE_NAME, as_error=True)
