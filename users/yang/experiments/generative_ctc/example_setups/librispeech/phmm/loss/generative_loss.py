import torch


from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.wav2vec2_hf_ctc_v2 import mask_tensor

def generative_nce(log_probs: torch.Tensor,
        soft_targets: torch.Tensor,
        sampling_type: str,
        seq_len: torch.Tensor,
        sampling_ratio: float=0.1,
        share_samples: bool=False,
        ratio_corrector: float = 1.0):

    '''
    log_probs: [B,T,V]
    soft_targets: [B,T,V]
    share_sample: if this is true, the sampling is done only once, and shared along the batch/sequence
                  if the sampling ratio is 1, this just take the whole batch/sequence

    we want the model output to be self-normalized, therefore constarining it to be in (0,1) is reasonable:
    log_probs should be the log_sigmoid output of the model
    sampling is already done when preparing the batch, so here we just to the summation
    we do the sampling with replacement
    '''
    assert sampling_type in ['batch', 'sequence']
    B,T,V = log_probs.shape
    len_mask = mask_tensor(log_probs, seq_len)
    n_samples = int(seq_len.sum().item())
    K = max(1, int(n_samples * sampling_ratio))
    device = log_probs.device
    log_sampling_ratio = torch.as_tensor(sampling_ratio,device=device, dtype=log_probs.dtype)
    log_ratio_corrector = torch.as_tensor(ratio_corrector, device=device, dtype=log_probs.dtype)
    if sampling_type == 'batch':
        valid_samples = log_probs[len_mask]# [n_samples, V]
        if share_samples: # all the frames share the same sampling result, this might enable a larger sampling ratio
            idx = torch.randint(0, n_samples, (1,1,K), device=device)
            negative_samples = valid_samples[idx]
        else: # different samples for each frame
            idx = torch.randint(0, n_samples, (B,T,K), device=device)
            negative_samples = valid_samples[idx]




    elif sampling_type == 'sequence':
        # sampling is done on the sequence level, i.e. the negative samples are only from the same sequence
        weights = len_mask.to(dtype=torch.float32)
        if share_samples:
            sample_idx = torch.multinomial(weights, K, replacement=True)  # [B, K]
            batch_idx = torch.arange(B, device=device)[:, None]  # [B, 1]
            negative_samples = log_probs[batch_idx, sample_idx].unsqueeze(1).expand(B, T, K, V)
        else:
            sample_idx = torch.multinomial(weights, T * K, replacement=True).view(B, T, K)  # [B, T, K]
            batch_idx = torch.arange(B, device=device)[:, None, None]  # [B, 1, 1]
            negative_samples = log_probs[batch_idx, sample_idx]  # [B, T, K, V]

    else:
        NotImplementedError
    positive_sample = log_probs - torch.logaddexp(log_probs, log_sampling_ratio + log_ratio_corrector) # [B,T,V]
    positive_sample = positive_sample * soft_targets
    negative_term = log_sampling_ratio + log_ratio_corrector - torch.logaddexp(negative_samples, log_sampling_ratio + log_ratio_corrector)
    negative_term = negative_term.sum(dim=-2)
    negative_term = negative_term * soft_targets
    loss = (negative_term-positive_sample).sum(-1)
    loss = loss[len_mask]

    return loss
