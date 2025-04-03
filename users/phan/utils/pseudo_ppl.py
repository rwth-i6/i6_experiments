"""
Some functions to compute pseudo PPL
There are be several ways to do this. The main concern is memory.
"""

import torch
import returnn.frontend as rf
from i6_experiments.users.phan.utils.masking import get_seq_mask

def compute_log_pseudo_ppl_loop_s(
    model:torch.nn.Module,
    targets: torch.Tensor,
    targets_len: torch.Tensor,
    mask_idx: int,
    n_classes: int,
):
    """
    This loops over S, mask out targets at the s-th position,
    compute the CE and accumulates the loss.

    :param model: torch model to forward targets
    :param targets: target sequences to compute log pseudo PPL (B, S)
    :param targets_len: target sequence lengths (B,)
    :param mask_idx: Index of mask token
    :param n_classes: Number of classes for one-hot
    """
    batch_size, max_seq_len = targets.shape
    device = targets.device
    seq_mask = get_seq_mask(targets_len, max_seq_len, device)
    acc_loss = 0
    for s in range(max_seq_len):
        targets_s = targets.clone()
        targets_s[:, s] = mask_idx
        targets_s_onehot = torch.nn.functional.one_hot(targets_s, num_classes=n_classes).float()
        log_lm_probs = model(targets_s_onehot, targets_len)
        ce = torch.nn.functional.cross_entropy(log_lm_probs[:, s:s+1, :].transpose(1, 2), targets[:, s:s+1], reduction='none')
        acc_loss += (ce.squeeze(-1) * seq_mask[:, s]).sum()
    loss = acc_loss/seq_mask.sum()
    return loss

def compute_log_pseudo_ppl_loop_b(
    model:torch.nn.Module,
    targets: torch.Tensor,
    targets_len: torch.Tensor,
    mask_idx: int,
    n_classes: int,
):
    """
    This loops over batch, for a sequence
    [a_1, ..., a_S] creates the masking
    [1st pos is masked, 2nd pos is masked, ...,  Sth pos is masked]
    then forward these at once.

    :param model: torch model to forward targets
    :param targets: target sequences to compute log pseudo PPL (B, S)
    :param targets_len: target sequence lengths (B,)
    :param mask_idx: Index of mask token
    :param n_classes: Number of classes for one-hot
    """
    batch_size, _ = targets.shape
    acc_loss = 0.
    for b in range(batch_size):
        targets_b = targets[b, :targets_len[b]]
        targets_b = targets_b.unsqueeze(0).expand(targets_len[b], -1)
        mask = torch.eye(targets_len[b]).bool().to(targets.device)
        targets_b_masked = torch.where(mask, mask_idx, targets_b)
        targets_b_masked = torch.nn.functional.one_hot(targets_b_masked, num_classes=n_classes).float()
        log_lm_probs = model(targets_b_masked, torch.full((targets_len[b],), targets_len[b]))
        ce = torch.nn.functional.cross_entropy(log_lm_probs.transpose(1, 2), targets_b, reduction='none')
        loss = (ce * mask.float()).sum()
        acc_loss += loss
    avg_loss = acc_loss / targets_len.sum().float()
    return avg_loss

def compute_log_pseudo_ppl_loop_s_rf_models(
    model: rf.Module,
    targets: rf.Tensor,
    targets_spatial_dim: rf.Tensor,
    mask_idx: int,
    model_kwargs: dict,
):
    """
    Same as compute_pseudo_ppl_loop_s but specifically for RF.

    This loops over S, mask out targets at the s-th position,
    compute the CE and accumulates the loss.

    :param model: RF model to forward targets. Must implement __call__ method.
        Output of __call__ must be a dict with key "output".
    :param targets: target sequences to compute log pseudo PPL (B, S)
    :param targets_len: target sequence lengths (B,)
    :param mask_idx: Index of mask token
    :param model_kwargs: kwargs for the RF model
    :returns: CE loss in shape (B,)
    """
    batch_size, max_seq_len = targets.raw_tensor.shape
    torch_target_lengths = targets_spatial_dim.dyn_size_ext.raw_tensor
    device = targets.device
    seq_mask = get_seq_mask(torch_target_lengths, max_seq_len, device)
    acc_loss = 0 # accumulated loss
    targets_raw = targets.raw_tensor
    for s in range(max_seq_len):
        targets_s = targets_raw.clone()
        targets_s[:, s] = mask_idx
        targets.raw_tensor = targets_s
        ilm_out_raw = model(targets, targets_spatial_dim, **model_kwargs)["output"].raw_tensor
        # print("s", s)
        # print(targets_s)
        # print("ilm_out_raw", ilm_out_raw)
        log_lm_probs = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, T, V)
        ce = torch.nn.functional.cross_entropy(log_lm_probs[:, s:s+1, :].transpose(1, 2), targets_raw[:, s:s+1].long(), reduction='none')
        acc_loss += ce.squeeze(-1) * seq_mask[:, s]
    return acc_loss
