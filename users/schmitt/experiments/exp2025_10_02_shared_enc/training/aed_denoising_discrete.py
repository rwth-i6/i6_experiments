from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple, Union, Optional, Dict

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence
import numpy as np

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict

from .util import get_random_mask, mask_sequence


class DenoisingAedModel(Protocol):
    mask_idx: int
    bos_idx: int
    eos_idx: int
    blank_idx: int

    embedding: nn.Embedding
    discriminator: Optional[nn.Module]

    @abstractmethod
    def forward(self, indices: Tensor, seq_lens: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return:
        """

    @abstractmethod
    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """


def train_step(
    *,
    model: DenoisingAedModel,
    extern_data: TensorDict,
    ce_loss_scale: float,
    masked_ce_loss_scale: float,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 0,
    masking_opts: Optional[Dict] = None,
    loss_name: Optional[str] = None,
    aux_loss_scales: Optional[Sequence[float]] = None,
    adv_loss_scale: float = 0.0,
    true_adv_target: Optional[int] = None,
    **_kwargs,
):
    loss_suffix = '_' + loss_name if loss_name else ''
    ctx = rf.get_run_ctx()
    num_eos_symbols = 1
    assert set(extern_data.data.keys()) == {"data", "seq_tag"} or set(extern_data.data.keys()) == {"data", "target", "seq_tag"}
    assert extern_data["data"].sparse_dim is not None
    label_indices_: ReturnnTensor = extern_data["data"]
    label_indices: Tensor = label_indices_.raw_tensor
    label_indices_lens: Tensor = label_indices_.dims[1].dyn_size_ext.raw_tensor

    # when using Combine dataset, we can have zero-length sequences
    if torch.all(label_indices_lens == 0).item():
        # print("All sequences have zero length, skipping this batch.")
        return
    label_indices = label_indices[label_indices_lens > 0]
    label_indices_lens = label_indices_lens[label_indices_lens > 0]

    if "target" in extern_data.data:
        assert extern_data["target"].sparse_dim is not None
        target_indices_: ReturnnTensor = extern_data["target"]
        target_indices: Tensor = target_indices_.raw_tensor
        target_indices_lens: Tensor = target_indices_.dims[1].dyn_size_ext.raw_tensor
    else:
        target_indices_ = label_indices_
        target_indices = label_indices
        target_indices_lens = label_indices_lens

    # mask is True, if original value should be kept
    if masking_opts["mask_prob"] == 0.0:
        label_indices_masked = label_indices
        label_indices_masked_lens = label_indices_lens.to(label_indices.device)
        B = label_indices_masked_lens.size(0)  # noqa
        T = label_indices_masked_lens.max().item()  # noqa
        phon_mask = torch.ones(B, T, device=label_indices.device).bool()
    else:
        if masking_opts.get("mask", None) is not None:
            phon_mask = masking_opts["mask"]
        else:
            phon_mask = get_random_mask(label_indices_lens.to(label_indices.device), **masking_opts)
        label_indices_masked, label_indices_masked_lens = mask_sequence(
            label_indices, label_indices_lens.to(label_indices.device), phon_mask, mask_value=model.mask_idx)

    adv_target = None
    adv_loss_name = None
    if adv_loss_scale > 0:
        if ctx.step % 4 in (0, 1):
            # train generator
            model.discriminator.freeze()
            model.unfreeze_encoder()
            adv_target = 1 - true_adv_target
            adv_loss_name = "gen"
        else:
            # train discriminator
            model.discriminator.unfreeze()
            model.freeze_encoder()
            adv_target = true_adv_target
            adv_loss_name = "disc"

    encoder_output, aux_logits, encoder_lens, _ = model.forward(label_indices_masked, label_indices_masked_lens)
    if adv_loss_name != "disc":
        # only compute the reconstruction loss, if we are not training the discriminator
        # since the encoder is frozen in that case

        input_labels = F.pad(target_indices, (1, 0), "constant", value=model.bos_idx)
        input_labels_len = target_indices_lens + 1
        logits = model.decode_seq(
            input_labels, input_labels_len.to(device=input_labels.device), encoder_output, encoder_lens
        )
        logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False)

        single_seqs = unpad_sequence(target_indices, target_indices_lens, batch_first=True)
        eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=label_indices.device, dtype=torch.int32)
        targets_w_eos_packed = pack_sequence(
            [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
            enforce_sorted=False,
        ).data
        single_masks = unpad_sequence(phon_mask, label_indices_lens, batch_first=True)
        phon_mask_packed = pack_sequence(
            # append True for each EOS symbol
            [torch.concat((mask, torch.ones(num_eos_symbols, dtype=torch.bool, device=label_indices.device)), dim=-1) for
             mask in single_masks],
            enforce_sorted=False,
        ).data

        ce_loss = F.cross_entropy(
            logits_packed.data,
            targets_w_eos_packed.long(),
            label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
            reduction="none",
        )

        num_masked_rf = label_indices_.dims[1].get_size_tensor().copy()
        num_masked_rf.raw_tensor = (~phon_mask).sum(dim=1).to(num_masked_rf.raw_tensor.dtype)

        error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)

        # loss is only calculated on masked positions
        if (~phon_mask_packed).sum().item() > 0:
            if masked_ce_loss_scale > 0:
                ctx.mark_as_loss(
                    # mask=False corresponds to masked positions in the original sequence
                    ce_loss[~phon_mask_packed],
                    f"masked_ce{loss_suffix}",
                    scale=masked_ce_loss_scale,
                    custom_inv_norm_factor=num_masked_rf,
                    use_normalized_loss=True,
                    # as_error=True
                )
            if ctx.stage == "train_step":
                ctx.mark_as_loss(error[~phon_mask_packed], f"masked_fer{loss_suffix}", as_error=True)

        if ctx.stage == "train_step":
            ctx.mark_as_loss(
                ce_loss,
                f"ce{loss_suffix}",
                scale=ce_loss_scale,
                custom_inv_norm_factor=label_indices_.dims[1].get_size_tensor() + num_eos_symbols,
                use_normalized_loss=True,
                # as_error=True
            )
            ctx.mark_as_loss(error, f"fer{loss_suffix}", as_error=True)

    if adv_loss_scale > 0.0:
        assert true_adv_target is not None
        single_enc_seqs = unpad_sequence(encoder_output, encoder_lens, batch_first=True)
        enc_out_packed = pack_sequence(
            single_enc_seqs,
            enforce_sorted=False,
        ).data
        disc_out = model.discriminator(enc_out_packed)
        adv_loss = F.binary_cross_entropy_with_logits(
            disc_out,
            torch.full_like(disc_out, fill_value=adv_target),
            reduction="none",
        )
        ctx.mark_as_loss(
            adv_loss,
            name=f"adv{loss_suffix}_{adv_loss_name}",
            scale=adv_loss_scale,
            custom_inv_norm_factor=label_indices_.dims[0].get_size_tensor(),
            use_normalized_loss=True,
        )

    if aux_loss_scales is None or len(aux_loss_scales) == 0 or all(scale == 0 for scale in aux_loss_scales):
        # it is allowed to ignore the aux loss outputs of the model
        return

    assert len(aux_loss_scales) == len(aux_logits)
    target_lens_ = target_indices_lens.to(device=target_indices.device)
    for i, (aux_logits_layer_i, scale) in enumerate(zip(aux_logits, aux_loss_scales)):
        aux_log_probs = F.log_softmax(aux_logits_layer_i, dim=-1)
        aux_loss = F.ctc_loss(
            log_probs=aux_log_probs.transpose(0, 1).to(torch.float32),
            targets=target_indices,
            input_lengths=encoder_lens,
            target_lengths=target_lens_,
            blank=model.blank_idx,
            reduction="none",
            zero_infinity=True,
        )
        ctx.mark_as_loss(
            aux_loss,
            name=f"ctc-{i}{loss_suffix}",
            custom_inv_norm_factor=target_indices_.dims[1].get_size_tensor(),
            scale=scale,
            use_normalized_loss=True,
        )
