"""
Some functions to compute some metrics for the bi-directional ILM.
Notable metrics: pseudo-PPL (in pseudo_ppl.py), l2r pseudo-PPL
"""

import torch
import returnn.frontend as rf
from i6_experiments.users.phan.utils.masking import get_seq_mask

def compute_log_pseudo_ppl_l2r_loop_s_rf_models(
    model: rf.Module,
    targets: rf.Tensor,
    targets_spatial_dim: rf.Tensor,
    mask_idx: int,
    model_kwargs: dict,
):
    """
    Compute the L2R pseudo PPL. This is like pseudo PPL but with right
    context masked out

    This loops over S, mask out targets from the s-th position,
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
        targets_s[:, s:] = mask_idx
        targets.raw_tensor = targets_s
        ilm_out_raw = model(targets, targets_spatial_dim, **model_kwargs)["output"].raw_tensor
        # print("s", s)
        # print(targets_s)
        # print("ilm_out_raw", ilm_out_raw)
        log_lm_probs = ilm_out_raw.transpose(0, 1).log_softmax(-1) # (B, T, V)
        ce = torch.nn.functional.cross_entropy(log_lm_probs[:, s:s+1, :].transpose(1, 2), targets_raw[:, s:s+1].long(), reduction='none')
        acc_loss += ce.squeeze(-1) * seq_mask[:, s]
    return acc_loss
