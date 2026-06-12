import torch


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    positions = torch.arange(tensor.shape[1], device=tensor.device)
    return positions[None, :] < seq_len[:, None]


def generative_nce(
    log_probs: torch.Tensor,
    soft_targets: torch.Tensor,
    sampling_type: str,
    seq_len: torch.Tensor,
    sampling_ratio: float = 0.1,
    share_samples: bool = False,
    ratio_corrector: float = 1.0,
):
    assert sampling_type in ["batch", "sequence"]
    if sampling_ratio <= 0.0:
        raise ValueError(f"sampling_ratio must be positive, got {sampling_ratio}")
    if ratio_corrector <= 0.0:
        raise ValueError(f"ratio_corrector must be positive, got {ratio_corrector}")

    batch_mask = seq_len > 0
    if not torch.any(batch_mask):
        return log_probs.new_zeros((0,))

    log_probs = log_probs[batch_mask]
    soft_targets = soft_targets[batch_mask]
    seq_len = seq_len[batch_mask]

    batch_dim, time_dim, _ = log_probs.shape
    len_mask = mask_tensor(log_probs, seq_len)
    n_samples = int(seq_len.sum().item())
    num_negatives = max(1, int(n_samples * sampling_ratio))
    device = log_probs.device
    log_sampling_ratio = torch.log(torch.as_tensor(sampling_ratio, device=device, dtype=log_probs.dtype))
    log_ratio_corrector = torch.log(torch.as_tensor(ratio_corrector, device=device, dtype=log_probs.dtype))

    if sampling_type == "batch":
        valid_samples = log_probs[len_mask]
        if share_samples:
            idx = torch.randint(0, n_samples, (1, 1, num_negatives), device=device)
        else:
            idx = torch.randint(0, n_samples, (batch_dim, time_dim, num_negatives), device=device)
        negative_samples = valid_samples[idx]
    else:
        weights = len_mask.to(dtype=torch.float32)
        if share_samples:
            sample_idx = torch.multinomial(weights, num_negatives, replacement=True)
            batch_idx = torch.arange(batch_dim, device=device)[:, None]
            negative_samples = log_probs[batch_idx, sample_idx].unsqueeze(1).expand(
                batch_dim, time_dim, num_negatives, -1
            )
        else:
            sample_idx = torch.multinomial(weights, time_dim * num_negatives, replacement=True).view(
                batch_dim, time_dim, num_negatives
            )
            batch_idx = torch.arange(batch_dim, device=device)[:, None, None]
            negative_samples = log_probs[batch_idx, sample_idx]

    denominator = log_sampling_ratio + log_ratio_corrector
    positive_sample = (log_probs - torch.logaddexp(log_probs, denominator)) * soft_targets
    negative_term = denominator - torch.logaddexp(negative_samples, denominator)
    negative_term = negative_term.sum(dim=-2) * soft_targets
    loss = -(negative_term + positive_sample).sum(-1)
    return loss[len_mask]
