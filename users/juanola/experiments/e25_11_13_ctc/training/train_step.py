__all__ = ["train_step"]

from typing import Sequence

import returnn.frontend as rf
import torch
import torch.nn.functional as F
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence
from returnn.frontend import RunCtx

from ..constants import DATA_PARAM_NAME, CLASSES_PARAM_NAME
from ...e25_10_17_sllm_d2.networks.interfaces.aed_ctc_model_protocol import AedCtcModelProtocol


def train_step(
        *,
        # RETURNN PARAMS
        model: AedCtcModelProtocol,
        extern_data: TensorDict,

        # TRAIN_STEP PARAMS
        aux_loss_scales: Sequence[float],
        aed_loss_scale: float,
        label_smoothing: float,
        label_smoothing_start_epoch: int,
        num_eos_symbols: int = 1, #only defined here

        **_kwargs,
):
    """
    RETURNN ENTRYPOINT!!
    """
    ctx: RunCtx = rf.get_run_ctx()

    assert aed_loss_scale > 0 or (
            len(aux_loss_scales) > 0 and any(scale > 0 for scale in aux_loss_scales)
    ), "must use at least AED or CTC aux loss"
    assert num_eos_symbols >= 1

    data_: ReturnnTensor = extern_data[DATA_PARAM_NAME]
    data: Tensor = data_.raw_tensor
    data_lens: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)

    targets_: ReturnnTensor = extern_data[CLASSES_PARAM_NAME]
    targets: Tensor = targets_.raw_tensor
    target_lens: Tensor = targets_.dims[1].dyn_size_ext.raw_tensor

    # ENCODER (FORWARD) STEP
    encoder_output, aux_logits, logits_lens, _ = model.forward(data, data_lens)

    if aed_loss_scale > 0:
        input_labels = F.pad(targets, (1, 0), "constant", value=model.bos_idx) # [B, MaxTextLen]
        input_labels_len = target_lens + 1 # [B]

        # DECODER (FORWARD) STEP
        logits: Tensor = model.decode_seq(input_labels, input_labels_len, encoder_output, logits_lens)

        # ??? some transformations
        logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False)
        single_seqs = unpad_sequence(targets, target_lens, batch_first=True)
        eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=targets.device, dtype=torch.int32)
        targets_w_eos_packed = pack_sequence(
            [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
            enforce_sorted=False,
        ).data

        # "ce" LOSS
        aed_loss = F.cross_entropy(
            logits_packed.data,
            targets_w_eos_packed.long(),
            label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
            reduction="none",
        )
        ctx.mark_as_loss(
            aed_loss,
            "ce",
            scale=aed_loss_scale,
            custom_inv_norm_factor=targets_.dims[1].get_size_tensor() + num_eos_symbols,
            use_normalized_loss=True,
        )

        # "fer" ERROR LOSS
        error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)
        ctx.mark_as_loss(error, "fer", as_error=True)

    # ???
    if len(aux_loss_scales) == 0 or all(scale == 0 for scale in aux_loss_scales):
        # it is allowed to ignore the aux loss outputs of the model
        return

    # "ctc" LOSSes
    assert len(aux_loss_scales) == len(aux_logits)
    target_lens_ = target_lens.to(device=targets.device)
    for i, (aux_logits_layer_i, scale) in enumerate(zip(aux_logits, aux_loss_scales)):
        aux_log_probs = F.log_softmax(aux_logits_layer_i, dim=-1)
        aux_loss = F.ctc_loss(
            log_probs=aux_log_probs.transpose(0, 1).to(torch.float32),
            targets=targets,
            input_lengths=logits_lens,
            target_lengths=target_lens_,
            blank=model.blank_idx,
            reduction="none",
            zero_infinity=True,
        )
        ctx.mark_as_loss(
            aux_loss,
            name=f"ctc-{i}",
            custom_inv_norm_factor=targets_.dims[1].get_size_tensor(),
            scale=scale,
            use_normalized_loss=True,
        )
