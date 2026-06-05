import torch
import torch.nn.functional as F


def segment_contrastive_loss(
    features: torch.Tensor,
    seq_len: torch.Tensor,
    num_samples: int = 5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Contrast adjacent frames against samples from the same sequence.

    features: [B, T, C]
    seq_len: [B]
    returns: [N] losses for all valid current-frame positions.
    """
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    seq_len = seq_len.to(device=features.device, dtype=torch.long)
    batch_mask = seq_len > 1
    if not torch.any(batch_mask):
        return features.new_zeros((0,))

    features = features[batch_mask]
    seq_len = seq_len[batch_mask]
    batch_size, time_dim, _feature_dim = features.shape

    features = F.normalize(features, p=2.0, dim=-1)
    current = features[:, :-1]  # [B, T-1, C]
    positive = features[:, 1:]  # [B, T-1, C]
    pair_time_dim = current.shape[1]

    frame_ids = torch.arange(pair_time_dim, device=features.device)[None, :]
    valid_pair_mask = frame_ids < (seq_len[:, None] - 1)

    pos_logits = (current * positive).sum(dim=-1, keepdim=True) / temperature  # [B, T-1, 1]

    if num_samples > 0:
        all_frame_ids = torch.arange(time_dim, device=features.device)[None, :]
        weights = (all_frame_ids < seq_len[:, None]).to(dtype=torch.float32)
        sample_idx = torch.multinomial(weights, pair_time_dim * num_samples, replacement=True).view(
            batch_size, pair_time_dim, num_samples
        )
        batch_idx = torch.arange(batch_size, device=features.device)[:, None, None]
        sampled = features[batch_idx, sample_idx]  # [B, T-1, K, C]
        neg_logits = (current.unsqueeze(2) * sampled).sum(dim=-1) / temperature  # [B, T-1, K]
        denom_logits = torch.cat([pos_logits, neg_logits], dim=-1)
    else:
        denom_logits = pos_logits

    loss = -pos_logits.squeeze(-1) + torch.logsumexp(denom_logits, dim=-1)
    return loss[valid_pair_mask]
