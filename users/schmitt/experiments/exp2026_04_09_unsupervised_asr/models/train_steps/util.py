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
    # # max_span should at most be as large as the smallest num_to_mask
    # # otherwise, it can happen that a sequence receives no mask at all
    # # because mask_lens_cum_sum > num_to_mask.unsqueeze(1) is True for the first sampled length
    # max_span = min(max_span, num_to_mask.min().item())
    # min_span = min(min_span, max_span - 1)
    # assert min_span > 0, "min_span must be strictly positive; otherwise, there might be no masked positions"
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
                mask[b, start : start + length] = False

    return mask


def expand_sequence(x: Tensor, lens: Tensor, min_dup: int, max_dup: int) -> Tuple[Tensor, Tensor]:
    """
    Upsample each token by a random consecutive-repeat count in [min_dup, max_dup], making the
    sequence longer (mimicking the audio>text length ratio). Padding positions are left untouched.

    Args:
        x: (B, T) sparse index sequence
        lens: (B,) valid lengths
        min_dup: minimum per-token duplication count (>= 1)
        max_dup: maximum per-token duplication count (>= min_dup)

    Returns:
        (expanded [B, T'] sequence, new lengths [B])
    """
    import os

    assert min_dup >= 1 and max_dup >= min_dup
    B, T = x.size(0), x.size(1)  # noqa
    device = x.device

    seed = int.from_bytes(os.urandom(4), "little")
    torch.manual_seed(seed)
    dur = torch.randint(low=min_dup, high=max_dup + 1, size=(B, T), device=device)
    valid = torch.arange(T, device=device).unsqueeze(0) < lens.unsqueeze(1)
    dur = dur * valid.to(dur.dtype)

    new_lens = dur.sum(dim=1)
    out = torch.zeros(B, int(new_lens.max().item()), dtype=x.dtype, device=device)
    for b in range(B):
        n = int(lens[b].item())
        if n == 0:
            continue
        rep = torch.repeat_interleave(x[b, :n], dur[b, :n])
        out[b, : rep.numel()] = rep

    return out, new_lens


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
    mask_indices = torch.cat([torch.zeros(mask.size(0), 1, device=x.device).int(), mask_indices], dim=1)
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
