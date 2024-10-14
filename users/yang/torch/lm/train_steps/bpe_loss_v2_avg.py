"""
This train step trains a phoneme based LSTM model
Audio data is completely ignored here
"""

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.yang.torch.utils import get_seq_mask


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    assert extern_data["data"].raw_tensor is not None
    targets = extern_data["data"].raw_tensor.long()
    delayed = extern_data["delayed"].raw_tensor.long()
    targets_len_rf = extern_data["data"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None


    batch_size, max_seq_len = targets.shape
    seq_mask = get_seq_mask(targets_len, max_seq_len, targets.device)

    output_logits = model(delayed, seq_mask) # (B, S, F)

    loss = torch.nn.functional.cross_entropy(output_logits.transpose(1, 2), targets, reduction='none')

    loss = (loss*seq_mask).sum()
    ppl = torch.exp(loss/targets_len.sum())
    # rf.get_run_ctx().mark_as_loss(
    #     name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    # )
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    rf.get_run_ctx().mark_as_loss(
        name="fake_ppl", loss=ppl, as_error=True,
    )
