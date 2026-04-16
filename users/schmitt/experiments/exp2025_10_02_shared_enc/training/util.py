from typing import Tuple, Union

import torch
from torch import Tensor


def get_random_mask(seq_lens: Tensor, mask_prob: float, min_span: int, max_span: int) -> Tensor:
    """
    Generate a random mask for sequences of given lengths.

    Args:
        seq_lens:
        mask_prob:
        min_span:
        max_span:

    Returns:

    """
    import os

    B = seq_lens.size(0)  # noqa
    T = seq_lens.max().item()  # noqa

    num_to_mask = (seq_lens.float() * mask_prob).ceil().int()
    seed = int.from_bytes(os.urandom(4), "little")
    torch.manual_seed(seed)
    mask_lens = torch.randint(low=min_span, high=max_span + 1, size=(B, T), device=seq_lens.device)
    mask_lens_cum_sum = torch.cumsum(mask_lens, dim=1)
    mask_lens[mask_lens_cum_sum > num_to_mask.unsqueeze(1)] = 0
    num_masks = (mask_lens > 0).sum(dim=1).max().item()
    mask_lens = mask_lens[:, :num_masks]

    mask = torch.ones(B, T, device=seq_lens.device).bool()
    for b in range(B):
        mask_lens_b = mask_lens[b]
        if mask_lens_b.sum() == 0:
            continue
        max_start = seq_lens[b].item() - mask_lens_b.max().item()
        seed = int.from_bytes(os.urandom(4), "little")
        torch.manual_seed(seed)
        mask_starts_b = torch.randint(low=0, high=max_start + 1, size=(num_masks,))
        mask_starts_b[mask_lens_b == 0] = 0
        for n in range(num_masks):
            start = mask_starts_b[n]
            length = mask_lens_b[n]
            if length > 0:
                mask[b, start: start + length] = False

    return mask


def mask_sequence(x: Tensor, lens: Tensor, mask: Tensor, mask_value: Union[int, Tensor]) -> Tuple[Tensor, Tensor]:
    """
    Given x of shape (B, T, ...) and a boolean mask of shape (B, T), return a new tensor where spans of False
    are removed and replaced by a single mask_value. Also return the new lengths.

    Args:
        x:
        lens:
        mask:
        mask_value:

    Returns:

    """
    B, T = x.size(0), x.size(1)  # noqa

    result = torch.zeros_like(x)

    cumsum = torch.cumsum(mask.int(), dim=1) - 1
    indices = mask[:, 1:].int() - mask[:, :-1].int()
    indices[indices < 0] = 0
    indices = torch.cumsum(indices, dim=1)
    indices = torch.cat([torch.zeros(mask.size(0), 1, device=x.device).int(), indices], dim=1)
    indices += cumsum
    row_idx = torch.arange(mask.size(0), device=x.device).unsqueeze(1) * mask.size(1)
    indices += row_idx
    indices = indices[mask]

    if len(result.shape) == 2:
        result.view(-1)[indices] = x[mask]
    else:
        result.view(B * T, -1)[indices] = x[mask]

    mask_indices = (~mask).long()
    mask_indices = torch.cat(
        [torch.zeros(mask.size(0), 1, device=x.device).int(), mask_indices],
        dim=1
    )
    mask_indices = mask_indices[:, 1:] - mask_indices[:, :-1]
    mask_indices[mask_indices < 0] = 0
    new_lens = lens - (~mask).sum(dim=1) + mask_indices.sum(dim=1)
    mask_indices_cumsum = torch.cumsum(mask_indices, dim=1) + cumsum
    mask_indices_cumsum += row_idx
    mask_indices = mask_indices_cumsum[mask_indices.bool()]

    if len(result.shape) == 2:
        assert isinstance(mask_value, int)
        result.view(-1)[mask_indices] = mask_value
    else:
        result.view(B * T, -1)[mask_indices] = mask_value.view(B * T, -1)[mask_indices]

    max_len = new_lens.max().item()
    result = result[:, :max_len]

    return result, new_lens
