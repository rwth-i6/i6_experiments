"""
Corpus-level output-statistics matching: an *unsupervised* training signal for the cross-modal
projection (the text CTC head applied to audio encoder states).

Motivation
----------
With a frozen shared encoder, a single linear projection ``[num_phon+1, model_dim]`` is enough to
read phonemes off the audio states (~41% PER when trained *with* labels). The unsupervised setups
try to obtain that projection by training the text CTC head on the *text* branch and transferring it
to the audio branch. Measured on a trained model, that transfer does not happen: the text-trained
projection has a mean row-wise cosine of 0.03 to the label-trained one and only 2 of 42 rows have the
correct phoneme as their nearest neighbour.

The reason is an identifiability problem. The adversarial loss matches ``p(audio states)`` to
``p(text states)`` as *marginals*, and marginal matching is invariant under any relabeling that
preserves the marginal -- it constrains the shape of the state cloud, not the correspondence. Both
CTC losses are same-modality, so they cannot break that symmetry either. No objective defined purely
over *representations* can identify the projection.

What can identify it is a constraint that depends on *which label* is assigned. Without pairs, the
available label-dependent signal is the structure of phoneme sequences. This module provides the
cheapest such signal: match the corpus-level statistics of what the text head emits on **audio** to
the corpus-level statistics of the **unpaired phoneme text**.

Three terms (each individually switchable):

- ``unigram``: the phoneme marginal (blank excluded and renormalized) must match the text unigram.
  This is a KL, so a phoneme the model never emits costs infinitely much -- directly targeting the
  observed pathology where 13 of 41 phonemes were unreachable and ``AY`` took 8.3% of frames.
- ``length``: the expected non-blank frame rate must match ``mean text tokens per sequence /
  mean audio frames per sequence``. Both averages are corpus-level and estimable from *unpaired*
  data (both branches read the same corpus), and this attacks the insertion problem directly -- the
  failing model emitted 1.86x too many tokens (89% insertions).
- ``bigram`` (opt-in): the phonotactics. Frame-level posteriors are compared as a soft "token"
  bigram, weighting each adjacent pair by the probability that the label *changes*
  (``1 - sum_k p_t[k] p_{t+1}[k]``), so within-phoneme frame repetitions do not dominate.

How much each term identifies (measured on the real statistics)
--------------------------------------------------------------
Measured by ``calc_output_stats_identifiability.py`` on the dev-other phoneme reference (2712 utts,
167k tokens; unigram perplexity 28.6 of 41; bigram carries I(prev;next) = 0.848 bits of phonotactic
information). A relabeling the loss cannot see is a solution training can converge to, so the
question is how much each term leaves ambiguous.

Against *random* relabelings both terms are strong (median KL 1.74 nats unigram, 8.13 bigram; neither
is ever blind). What matters in practice is *local* ambiguity -- gradient descent reaches a
neighbouring labeling far more easily than a random one -- and there the two differ sharply. Over all
820 single phoneme transpositions:

===================================  ==========  =========
threshold (KL below which blind)     unigram     bigram
===================================  ==========  =========
0.001                                101 of 820  1 of 820
0.01                                 303         1
0.05                                 569        23
0.1                                  669        85
===================================  ==========  =========

So the unigram term is blind to swapping any two phonemes of near-equal frequency (``N<->T`` costs
4e-6, ``SH<->Y`` 0, ``W<->HH`` 1e-6, ``B<->P`` 6e-6, ...). Of those 101 swaps, the bigram charges a
median of 0.233 nats, and **exactly one remains cheap under both** -- ``' <-> <SIL>``, which is
vacuous, since neither symbol occurs at all in the wo-silence reference. Excluding it: **zero
non-vacuous residual ambiguity at the single-swap level.**

Conversely the bigram term alone does not penalize a near-uniform output (a degenerate emission has
little sequential structure to be wrong about), which the unigram term catches immediately. The two
are therefore complementary, and ``bigram_scale`` should always be used *together with*
``unigram_scale``, never instead of it -- but on this evidence the ``uni-bi`` combination is the one
to run, not unigram alone.

Caveats: this measures the information available in the statistic, i.e. an upper bound on what the
loss can exploit -- the *predicted* bigram here is a soft frame-level estimate (``_soft_bigram``),
not an exact token bigram. And single transpositions are only the local picture; a labeling that is
wrong on many phonemes at once is not covered. A full LM (``phoneme_lm``) or the wav2vec-U
discriminator remains the stronger version of the same idea.

Targets vs. predictions
-----------------------
Under ``alternate_batching`` a step sees only one modality, so the text targets and the audio
predictions never co-occur in a batch. The targets are therefore *accumulated across steps*: since
they come from reference labels they do not change during training, so plain cumulative counts (not
an EMA) give exactly the corpus statistic. The prediction side is computed from the current batch
only, which keeps it fully differentiable and is a low-variance estimate at typical batch sizes
(~15k frames over ~41 classes).

The accumulator is process-global module state (like ``ctx.step``-driven state elsewhere in this
package). It is not checkpointed, so it restarts empty on resume; ``warmup_steps`` suppresses the
loss until enough text batches have been seen, and the counts re-converge within a few hundred steps.
"""

__all__ = ["DEFAULT_OPTS", "update_and_compute_losses", "reset_state"]

from typing import Any, Dict, Optional

import torch
from torch import Tensor

import returnn.frontend as rf


DEFAULT_OPTS: Dict[str, Any] = {
    "unigram_scale": 1.0,  # KL(text unigram || predicted phoneme marginal)
    "length_scale": 1.0,  # (predicted non-blank rate - corpus token/frame ratio)^2
    "bigram_scale": 0.0,  # KL over the soft phoneme bigram; 0 -> off
    "warmup_steps": 20,  # text batches to accumulate before the loss is applied
}


class _Accumulator:
    """Cumulative reference statistics, filled from the text branch's reference labels."""

    def __init__(self, num_phon: int, device, dtype):
        self.num_phon = num_phon
        self.unigram_counts = torch.zeros(num_phon, device=device, dtype=dtype)
        self.bigram_counts = torch.zeros(num_phon, num_phon, device=device, dtype=dtype)
        self.text_tokens = torch.zeros((), device=device, dtype=dtype)
        self.text_seqs = torch.zeros((), device=device, dtype=dtype)
        self.audio_frames = torch.zeros((), device=device, dtype=dtype)
        self.audio_seqs = torch.zeros((), device=device, dtype=dtype)
        self.num_text_updates = 0

    @property
    def unigram(self) -> Tensor:
        return self.unigram_counts / self.unigram_counts.sum().clamp_min(1.0)

    @property
    def bigram(self) -> Tensor:
        return self.bigram_counts / self.bigram_counts.sum().clamp_min(1.0)

    @property
    def target_nonblank_rate(self) -> Optional[Tensor]:
        """Expected fraction of audio frames that should carry a (non-blank) label.

        ``(text tokens / text seqs) / (audio frames / audio seqs)``: both ratios are per-sequence
        corpus averages over the *same* corpus, so this is well defined even though the audio and
        text sequences in any given batch are unrelated.
        """
        if self.text_seqs == 0 or self.audio_seqs == 0 or self.audio_frames == 0:
            return None
        return (self.text_tokens / self.text_seqs) / (self.audio_frames / self.audio_seqs)


_STATE: Dict[str, _Accumulator] = {}


def reset_state() -> None:
    """Drop the accumulated reference statistics (used by tests)."""
    _STATE.clear()


def _get_state(num_phon: int, device, dtype) -> _Accumulator:
    acc = _STATE.get("acc")
    if acc is None or acc.num_phon != num_phon:
        acc = _Accumulator(num_phon, device, dtype)
        _STATE["acc"] = acc
    return acc


def _valid_mask(lens: Tensor, max_len: int) -> Tensor:
    return torch.arange(max_len, device=lens.device)[None, :] < lens[:, None]


def _kl(target: Tensor, pred: Tensor) -> Tensor:
    """KL(target || pred) in nats, over the entries where the target has mass."""
    mask = target > 0
    return (target[mask] * (target[mask].log() - pred[mask].clamp_min(1e-10).log())).sum()


def update_and_compute_losses(
    *,
    model,
    modality: str,
    encoder_output: Tensor,
    encoder_lens: Tensor,
    target_indices: Tensor,
    target_indices_lens: Tensor,
    opts: Dict[str, Any],
    loss_suffix: str = "",
) -> None:
    """
    Update the reference statistics (text branch) or add the matching losses (audio branch).

    :param modality: "text" -> accumulate reference statistics from ``target_indices``;
        "audio" -> run the **text** CTC head over the audio encoder states and add the losses.
    :param encoder_output: ``[B, T, F]`` shared-encoder states of this modality.
    :param encoder_lens: ``[B]`` valid lengths of ``encoder_output``.
    :param target_indices: ``[B, S]`` reference labels (phoneme ids for the text branch).
    :param target_indices_lens: ``[B]`` valid lengths of ``target_indices``.
    :param opts: see :data:`DEFAULT_OPTS`.
    """
    assert modality in ("audio", "text"), modality
    opts = {**DEFAULT_OPTS, **opts}
    ctx = rf.get_run_ctx()

    num_phon = model.text_out_dim - 3  # the model reserves the top 3 ids for [mask, bos, eos]
    assert model.text_blank_idx == num_phon, (
        f"expected the CTC blank ({model.text_blank_idx}) to be the last output of the text head"
        f" (num_phon={num_phon})"
    )
    assert len(model.out_text_aux_logits) > 0, (
        "output-statistics matching needs the text CTC head (model_args.text_aux_loss_layers must be set)"
    )
    acc = _get_state(num_phon, encoder_output.device, torch.float32)

    if modality == "text":
        # reference statistics only -- no gradient, no loss.
        with torch.no_grad():
            labels = target_indices.long()
            lens = target_indices_lens.to(device=labels.device)
            mask = _valid_mask(lens, labels.shape[1])
            flat = labels[mask]
            flat = flat[flat < num_phon]  # defensive: ignore any special ids
            acc.unigram_counts += torch.bincount(flat, minlength=num_phon).to(
                device=acc.unigram_counts.device, dtype=acc.unigram_counts.dtype
            )
            acc.text_tokens += lens.sum().to(device=acc.text_tokens.device, dtype=acc.text_tokens.dtype)
            acc.text_seqs += torch.tensor(float(lens.numel()), device=acc.text_seqs.device)
            if opts["bigram_scale"] > 0.0:
                # only pairs inside the same sequence
                pair_mask = mask[:, 1:] & mask[:, :-1]
                prev, nxt = labels[:, :-1][pair_mask], labels[:, 1:][pair_mask]
                keep = (prev < num_phon) & (nxt < num_phon)
                prev, nxt = prev[keep], nxt[keep]
                acc.bigram_counts += (
                    torch.bincount(prev * num_phon + nxt, minlength=num_phon * num_phon)
                    .reshape(num_phon, num_phon)
                    .to(device=acc.bigram_counts.device, dtype=acc.bigram_counts.dtype)
                )
            acc.num_text_updates += 1
        return

    # --- audio branch: the cross-modal projection we actually want to shape --------------------
    with torch.no_grad():
        acc.audio_frames += encoder_lens.sum().to(device=acc.audio_frames.device, dtype=acc.audio_frames.dtype)
        acc.audio_seqs += torch.tensor(float(encoder_lens.numel()), device=acc.audio_seqs.device)

    if acc.num_text_updates < opts["warmup_steps"]:
        return  # reference statistics not yet meaningful

    # the *text* head on *audio* states -- exactly the path recognition uses, and the only path
    # that never receives a training signal from the (same-modality) CTC losses.
    logits = model.out_text_aux_logits[-1](encoder_output)  # [B, T, num_phon+1]
    probs = logits.float().softmax(dim=-1)
    mask = _valid_mask(encoder_lens.to(probs.device), probs.shape[1])
    probs = probs[mask]  # [N, num_phon+1]
    if probs.shape[0] == 0:
        return
    phon, blank = probs[:, :num_phon], probs[:, num_phon]

    if opts["unigram_scale"] > 0.0:
        pred_unigram = phon.sum(dim=0)
        pred_unigram = pred_unigram / pred_unigram.sum().clamp_min(1e-10)
        loss = _kl(acc.unigram, pred_unigram)
        ctx.mark_as_loss(loss, name=f"out_stats_unigram{loss_suffix}", dims=[], scale=opts["unigram_scale"])

    if opts["length_scale"] > 0.0:
        target_rate = acc.target_nonblank_rate
        if target_rate is not None:
            pred_rate = (1.0 - blank).mean()
            ctx.mark_as_loss(
                (pred_rate - target_rate.detach()) ** 2,
                name=f"out_stats_length{loss_suffix}",
                dims=[],
                scale=opts["length_scale"],
            )
            if ctx.stage == "train_step":
                ctx.mark_as_loss(pred_rate, name=f"out_stats_nonblank_rate{loss_suffix}", dims=[], as_error=True)
                ctx.mark_as_loss(
                    target_rate.detach(), name=f"out_stats_target_rate{loss_suffix}", dims=[], as_error=True
                )

    if opts["bigram_scale"] > 0.0:
        pred_bigram = _soft_bigram(logits, encoder_lens, num_phon)
        if pred_bigram is not None:
            ctx.mark_as_loss(
                _kl(acc.bigram, pred_bigram), name=f"out_stats_bigram{loss_suffix}", dims=[], scale=opts["bigram_scale"]
            )


def _soft_bigram(logits: Tensor, encoder_lens: Tensor, num_phon: int) -> Optional[Tensor]:
    """
    Differentiable estimate of the *token* bigram implied by frame-level posteriors.

    Adjacent frames usually carry the same phoneme (a phoneme spans several frames), so a plain
    frame bigram would be dominated by self-transitions while the reference token bigram has almost
    none. Each pair is therefore weighted by the probability that the label changes,
    ``w_t = 1 - sum_k p_t[k] p_{t+1}[k]``, which is a differentiable stand-in for CTC's
    repeat-collapsing.
    """
    probs = logits.float().softmax(dim=-1)[..., :num_phon]  # [B, T, P], blank dropped
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-10)
    mask = _valid_mask(encoder_lens.to(probs.device), probs.shape[1])
    pair_mask = mask[:, 1:] & mask[:, :-1]
    if not bool(pair_mask.any()):
        return None
    prev = probs[:, :-1][pair_mask]  # [M, P]
    nxt = probs[:, 1:][pair_mask]  # [M, P]
    change = (1.0 - (prev * nxt).sum(dim=-1, keepdim=True)).clamp_min(0.0)  # [M, 1]
    bigram = (prev * change).transpose(0, 1) @ nxt  # [P, P]
    return bigram / bigram.sum().clamp_min(1e-10)
