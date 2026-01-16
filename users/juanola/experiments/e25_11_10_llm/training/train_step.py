__all__ = ["train_step"]

from typing import Sequence

import returnn.frontend as rf
import torch
import torch.nn.functional as F
from returnn.frontend import RunCtx
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pack_sequence, unpad_sequence

from ..networks.interfaces.lm_decoder_model_protocol import LmDecoderModelProtocol


def train_step( # TODO: LLM
        *,
        # RETURNN PARAMS
        model: LmDecoderModelProtocol,
        extern_data: TensorDict, #TODO: check extern_data vs data

        # TRAIN_STEP PARAMS
        aux_loss_scales: Sequence[float], # not used
        aed_loss_scale: float, # not used
        label_smoothing: float,
        label_smoothing_start_epoch: int,
        num_eos_symbols: int = 1, #only defined here

        **_kwargs,
):
    """
    RETURNN ENTRYPOINT!!

    Coupled with LmDataset from returnn.datasets.lm
    """
    assert num_eos_symbols >= 1

    ctx: RunCtx = rf.get_run_ctx()

    targets_ = extern_data["data"] # Target / label / ground truth # TODO: extract const
    targets: Tensor = targets_.raw_tensor
    target_lens: Tensor = targets_.dims[1].dyn_size_ext.raw_tensor

    # DECODER (FORWARD) STEP
    input_labels = F.pad(targets, (1, 0), "constant", value=model.bos_idx) # [B, MaxTextLen]
    input_labels_len = target_lens + 1 # [B]
    logits: Tensor = model.decode_seq_lm(input_labels, input_labels_len) # [B, SeqLen, vocab_size] | ex. SeqLen [TK1, TK2, EOS]

    # LOGITS PREP
    logits_packed = pack_padded_sequence(logits, input_labels_len, batch_first=True, enforce_sorted=False) # Remove PAD

    # TARGETS PREP
    single_seqs = unpad_sequence(targets, target_lens, batch_first=True) # Remove padding
    eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=targets.device, dtype=torch.int32) # New EOS...
    targets_w_eos_packed = pack_sequence(
        [torch.concat((seq, eos_tensor), dim=-1) for seq in single_seqs],
        enforce_sorted=False,
    ).data

    # "ce" LOSS
    cross_entropy_loss = F.cross_entropy(
        logits_packed.data,
        targets_w_eos_packed.long(),
        label_smoothing=label_smoothing if ctx.epoch >= label_smoothing_start_epoch else 0.0,
        reduction="none",
    )
    ctx.mark_as_loss(
        cross_entropy_loss,
        "ce",
        custom_inv_norm_factor=targets_.dims[1].get_size_tensor() + num_eos_symbols,
        use_normalized_loss=True,
    )

    # "fer" ERROR LOSS (not used for training)
    error = torch.argmax(logits_packed.data, dim=-1).not_equal(targets_w_eos_packed)
    ctx.mark_as_loss(error, "fer", as_error=True)
