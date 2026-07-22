"""
Train step for the wav2vec-U (wav2vec Unsupervised) GAN — the *first* training stage of the
fairseq ``examples/wav2vec/unsupervised`` pipeline.

This reproduces what fairseq does per update for ``criterion=model`` + the composite optimizer:

  * ``model.set_num_updates(step)`` anneals the (gumbel) temperature and selects, via
    ``discrim_step(step) = step % 2 == 1``, whether this is a **discriminator** or a **generator**
    update.
  * fairseq's composite optimizer steps *only* the active param group each update. We emulate that
    with a single RETURNN optimizer by freezing the inactive group's parameters (the generator on
    discriminator steps, the discriminator on generator steps). See ``models/optim.py`` for the
    per-group lr / weight-decay wiring.
  * ``ModelCriterion`` sums the (already internally scaled) losses returned by the model and the
    fairseq trainer normalizes the gradient by ``sample_size`` (the batch size). We reproduce that
    by dividing each loss by ``sample_size`` before marking it.

Data contract (``CombinedDataset``, unpaired speech + text):
    The unpaired audio (``data`` = 512-d speech features) and text (``phon_indices`` = phoneme ids)
    live in one ``CombinedDataset``: **each sequence carries exactly one modality** (the other key is
    length 0). Batched *without* ``alternate_batching`` (plain batching over the interleaved
    ordering), a single batch therefore contains **both** modalities on **different rows** — the
    feature-rows (``data`` length > 0) and the text-rows (``phon_indices`` length > 0), unpaired and
    random. The GAN needs both in the *same* forward (the discriminator compares generator-output
    distributions against real-text one-hots, and the WGAN gradient penalty interpolates between
    them), so we split the batch by modality here and feed both to ``model.forward``.

    NB: do NOT use ``alternate_batching`` with this step — it yields single-modality batches, which
    cannot feed the GAN's cross-modality losses.
"""

from typing import Optional

import torch

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict

from ..definitions.wav2vec_u import Model
from ..optim import set_active_param_group


def _padding_mask_from_lens(lens: torch.Tensor, max_len: int, device) -> torch.Tensor:
    """Return a ``[B, max_len]`` bool mask that is True at padding (>= length) positions."""
    ar = torch.arange(max_len, device=device).unsqueeze(0)
    return ar >= lens.to(device).unsqueeze(1)


def train_step(
    *,
    model: Model,
    extern_data: TensorDict,
    features_key: str = "data",
    text_key: str = "phon_indices",
    **_kwargs,
):
    ctx = rf.get_run_ctx()
    step = int(ctx.step)

    # temperature annealing + discriminator/generator selection (fairseq set_num_updates)
    model.set_num_updates(step)
    d_step = model.discrim_step(step)

    # --- the speech-feature rows and the text rows of this (mixed) batch ---
    features_: ReturnnTensor = extern_data[features_key]
    features_all: torch.Tensor = features_.raw_tensor  # [B, T_d, F]
    features_lens_all: torch.Tensor = features_.dims[1].dyn_size_ext.raw_tensor  # [B]
    device = features_all.device

    text_: ReturnnTensor = extern_data[text_key]
    text_all: torch.Tensor = text_.raw_tensor.long()  # [B, T_p]
    text_lens_all: torch.Tensor = text_.dims[1].dyn_size_ext.raw_tensor  # [B]

    feat_rows = features_lens_all.to(device) > 0
    text_rows = text_lens_all.to(device) > 0

    # a valid GAN update needs at least one speech row AND one text row in the batch
    if not (feat_rows.any() and text_rows.any()):
        return

    features = features_all[feat_rows]  # [Bf, T_d, F]
    features_lens = features_lens_all.to(device)[feat_rows]
    padding_mask = _padding_mask_from_lens(features_lens, features.size(1), device)

    random_label = text_all[text_rows]  # [Bt, T_p]
    text_lens = text_lens_all.to(device)[text_rows]
    text_pad_mask = _padding_mask_from_lens(text_lens, random_label.size(1), device)
    random_label = random_label.masked_fill(text_pad_mask, model.pad)

    # fairseq's composite optimizer computes grads for BOTH groups (the active step's loss flows
    # through both), clips the GLOBAL grad norm, and steps only the active group. We do NOT freeze
    # here, so both groups get grads and RETURNN's global clip matches fairseq's clip scope;
    # GanAlternatingAdamW then steps only the group selected below (dropping the other's grads after
    # the clip). See models/optim.py.
    set_active_param_group("discriminator" if d_step else "generator")

    result = model(features, padding_mask, random_label=random_label)

    losses = result["losses"]
    Bf = features.size(0)  # #speech (fake) rows -- fairseq's sample_size
    Bt = random_label.size(0)  # #text (real) rows
    # Normalize each loss by the batch count it was reduced over. fairseq always has Bf == Bt == N;
    # our mixed batches have Bf != Bt. dense_* are summed over Bf, token_d over Bt, grad_pen over
    # min(Bf, Bt); code_pen/smoothness were multiplied by sample_size=Bf in the model, so /Bf yields
    # their mean.
    loss_divisors = {
        "dense_g": Bf,
        "dense_d": Bf,
        "code_pen": Bf,
        "smoothness": Bf,
        "mmi": Bf,
        "token_d": Bt,
        "grad_pen": max(min(Bf, Bt), 1),
    }
    for name, loss in losses.items():
        if loss is None:
            continue
        ctx.mark_as_loss(loss / loss_divisors[name], name=name, dims=[])

    # logging-only metrics (fairseq log_keys: temp, code_ppl, ...)
    if ctx.stage == "train_step":
        ctx.mark_as_loss(torch.tensor(float(result["temp"]), device=device), name="temp", dims=[], as_error=True)
        ctx.mark_as_loss(
            torch.tensor(float(d_step), device=device), name="d_steps", dims=[], as_error=True
        )
        if result["code_ppl"] is not None:
            ctx.mark_as_loss(result["code_ppl"], name="code_ppl", dims=[], as_error=True)
        if result["prob_ppl"] is not None:
            ctx.mark_as_loss(result["prob_ppl"], name="prob_ppl", dims=[], as_error=True)
