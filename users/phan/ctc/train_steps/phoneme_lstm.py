"""
This train step trains a phoneme based LSTM model
Audio data is completely ignored here
"""

from returnn.tensor import batch_dim
from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch
from i6_experiments.users.phan.utils import get_seq_mask


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    assert extern_data["data"].raw_tensor is not None
    targets = extern_data["data"].raw_tensor.long()

    targets_len_rf = extern_data["data"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None


    model.train()

    batch_size, max_seq_len = targets.shape
    # pad 0 at the beginning
    eos_targets = torch.cat(
        [torch.zeros((batch_size, 1), device=targets.device), targets],
        dim=1,
    ).long()
    # pad 0 at the end
    targets_eos = torch.cat(
        [targets, torch.zeros((batch_size, 1), device=targets.device)],
        dim=1,
    ).long()
    eos_targets_one_hot = torch.nn.functional.one_hot(eos_targets, num_classes=model.cfg.vocab_dim).float()
    targets_eos_len = targets_len + 1

    log_lm_probs = model(eos_targets_one_hot) # (B, S, F)
    loss = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_eos, reduction='none')
    seq_mask = get_seq_mask(targets_eos_len, max_seq_len + 1, targets.device)
    loss = (loss*seq_mask).sum()
    targets_len_rf.raw_tensor += 1 # because padding
    ppl = torch.exp(loss/targets_len_rf.raw_tensor.sum())
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
    rf.get_run_ctx().mark_as_loss(
        name="ppl", loss=ppl, as_error=True,
    )
