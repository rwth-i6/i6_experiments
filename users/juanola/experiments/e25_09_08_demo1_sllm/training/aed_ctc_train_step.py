__all__ = ["AedCtcModel", "train_step"]

from abc import abstractmethod
from typing import List, Protocol, Sequence, Tuple

import returnn.frontend as rf
import torch
import torch.nn.functional as F
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence


class AedCtcModel(Protocol):
    blank_idx: int
    bos_idx: int
    eos_idx: int

    @abstractmethod
    def forward(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return: tuple of:
            - encoder output
            - list of CTC aux logits
            - length tensor of the output
            - mask tensor of the output
        """

    @abstractmethod
    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_mask: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterwards.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """


def train_step(
    *,
    model: AedCtcModel,
    extern_data: TensorDict,
    aux_loss_scales: Sequence[float],
    aed_loss_scale: float,
    label_smoothing: float,
    label_smoothing_start_epoch: int,
    num_eos_symbols: int = 1,
    **_kwargs,
):
    ctx = rf.get_run_ctx()

    assert aed_loss_scale > 0 or (
        len(aux_loss_scales) > 0 and any(scale > 0 for scale in aux_loss_scales)
    ), "must use at least AED or CTC aux loss"
    assert num_eos_symbols >= 1

    data_: ReturnnTensor = extern_data["data"]
    data: Tensor = data_.raw_tensor
    data_lens: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    targets_: ReturnnTensor = extern_data["classes"]
    targets: Tensor = targets_.raw_tensor
    target_lens: Tensor = targets_.dims[1].dyn_size_ext.raw_tensor

    encoder_output, aux_logits, logits_lens, _ = model.forward(data, data_lens)

    if aed_loss_scale > 0:
        input_labels = F.pad(targets, (1, 0), "constant", value=model.bos_idx)
        input_labels_len = target_lens + 1
        logits = model.decode_seq(
            input_labels, input_labels_len.to(device=input_labels.device), encoder_output, logits_lens
        )
        logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False)

        single_seqs = unpad_sequence(targets, target_lens, batch_first=True)
        eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=targets.device, dtype=torch.int32)
        targets_w_eos_packed = pack_sequence(
            [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
            enforce_sorted=False,
        ).data

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

        error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)
        ctx.mark_as_loss(error, "fer", as_error=True)

    if len(aux_loss_scales) == 0 or all(scale == 0 for scale in aux_loss_scales):
        # it is allowed to ignore the aux loss outputs of the model
        return

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
