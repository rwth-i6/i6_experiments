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
from ..networks.interfaces.lm_decoder_model_protocol import LmDecoderModelProtocol


def train_step(
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

    # TODO: from robin:
    """
    to me, it seems like "delayed" is just the labels with BOS prepended -> i.e. the actual labels are "delayed" on 
    step to the right. in our SLLM and AED setups, we do this manually inside the train step - see the padding function 
    which adds BOS
    
    +
    
    CHeck out albert setup https://github.com/rwth-i6/i6_experiments/blob/main/users/zeyer/experiments/exp2024_04_23_baselines/lm.py
    """

    # TODO: check that data is received correctly
    targets_ = extern_data["data"] # Target / label / ground truth
    targets: Tensor = targets_.raw_tensor
    target_lens: Tensor = targets_.dims[1].dyn_size_ext.raw_tensor
    #target_lens = extern_data["data:size1"]

    data_ = extern_data["delayed"] # Already generated sequence (shifted by one position)
    data: Tensor = data_.raw_tensor
    data_lens: Tensor = data_.dims[1].dyn_size_ext.raw_tensor.to(device=data.device)
    #data_lens = extern_data["delayed:size1"]


    # DECODER (FORWARD) STEP
    # No encoder output | only delayed seq (to predict next token)
    logits: Tensor = model.decode_seq(data, data_lens, None, None)
    # TODO: understand exactly how delayed data looks [0, seq]? and what the model expects

    # TODO: finish below on how data is obtained and used
    # ??? some transformations
    logits_packed = pack_padded_sequence(logits, data_lens, batch_first=True, enforce_sorted=False)
    single_seqs = unpad_sequence(targets, target_lens, batch_first=True)
    eos_tensor = torch.tensor(num_eos_symbols * [model.eos_idx], device=targets.device, dtype=torch.int32)
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
