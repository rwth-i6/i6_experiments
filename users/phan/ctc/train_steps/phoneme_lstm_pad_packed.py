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
    audio_features = extern_data["data"].raw_tensor
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()

    targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None
    

    model.train()

    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets, targets_len, batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    batch_size, max_seq_len = targets.shape
    # pad 0 at the beginning and end
    targets_padded = torch.cat(
        [torch.zeros((batch_size, 1), device=targets.device), targets, torch.zeros((batch_size, 1), device=targets.device)],
        dim=1,
    ).long()
    targets_padded_len = targets_len + 2
    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets_padded, targets_padded_len, batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_one_hot = torch.nn.functional.one_hot(targets_padded, num_classes=model.cfg.vocab_dim).float()
    

    log_lm_probs = model(targets_one_hot) # (B, S, F)
    
    loss = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_masked, ignore_index=-100, reduction='sum')

    targets_len_rf.raw_tensor += 2 # because padding
    rf.get_run_ctx().mark_as_loss(
        name="log_ppl", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )
