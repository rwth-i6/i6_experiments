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
    
    # Generate random mask lengths and cap them to the maximum sequence length in the batch to avoid negative max_start
    mask_lens = torch.randint(low=min_span, high=max_span + 1, size=(B, T), device=seq_lens.device)
    mask_lens = torch.min(mask_lens, seq_lens.unsqueeze(1))
    
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
        max_start = max(0, max_start)  # ensure it is not negative
        seed = int.from_bytes(os.urandom(4), "little")
        torch.manual_seed(seed)
        mask_starts_b = torch.randint(low=0, high=max_start + 1, size=(num_masks,), device=seq_lens.device)
        mask_starts_b[mask_lens_b == 0] = 0
        for n in range(num_masks):
            start = mask_starts_b[n]
            length = mask_lens_b[n]
            if length > 0:
                mask[b, start : start + length] = False

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

def mask_sequence_expand(x: Tensor, lens: Tensor, mask: Tensor, mask_value: Union[int, Tensor], insert_prob: float = 0.1) -> Tuple[Tensor, Tensor]:
    """
    Given x of shape (B, T, ...) and a boolean mask of shape (B, T), return a new tensor where spans of False
    are masked but not collapsed. Instead, a span of length N is replaced by N + K masks where K is U(1, 5).
    Additionally, random masks of length 1-5 can be inserted before unmasked tokens with `insert_prob`.
    """
    B = x.size(0)
    device = x.device
    
    new_x_list = []
    new_lens_list = []
    
    # We will build up a list of tensors for each sequence, then pad
    for b in range(B):
        seq_len = lens[b].item()
        x_b = x[b, :seq_len]
        mask_b = mask[b, :seq_len]
        
        out_tokens = []
        
        i = 0
        while i < seq_len:
            if not mask_b[i]:
                # Found a masked span
                start = i
                while i < seq_len and not mask_b[i]:
                    i += 1
                span_len = i - start
                
                # N + K masks
                K = torch.randint(1, 6, (1,)).item()
                num_masks = span_len + K
                
                if isinstance(mask_value, int):
                    masks_tensor = torch.full((num_masks,) + x_b.shape[1:], mask_value, dtype=x_b.dtype, device=device)
                else:
                    masks_tensor = mask_value.unsqueeze(0).expand(num_masks, *mask_value.shape).clone()
                out_tokens.append(masks_tensor)
            else:
                # Found an unmasked token
                # Decide if we insert random masks before it
                if torch.rand((1,)).item() < insert_prob:
                    K_insert = torch.randint(1, 6, (1,)).item()
                    if isinstance(mask_value, int):
                        masks_tensor = torch.full((K_insert,) + x_b.shape[1:], mask_value, dtype=x_b.dtype, device=device)
                    else:
                        masks_tensor = mask_value.unsqueeze(0).expand(K_insert, *mask_value.shape).clone()
                    out_tokens.append(masks_tensor)
                
                # Add the unmasked token
                out_tokens.append(x_b[i:i+1])
                i += 1
                
        if len(out_tokens) > 0:
            out_seq = torch.cat(out_tokens, dim=0)
        else:
            out_seq = torch.empty((0,) + x_b.shape[1:], dtype=x_b.dtype, device=device)
            
        new_x_list.append(out_seq)
        new_lens_list.append(out_seq.size(0))
        
    new_lens = torch.tensor(new_lens_list, device=device, dtype=lens.dtype)
    
    if len(new_x_list) > 0:
        result = torch.nn.utils.rnn.pad_sequence(new_x_list, batch_first=True, padding_value=0)
    else:
        result = torch.empty((B, 0) + x.shape[2:], dtype=x.dtype, device=device)
        
    return result, new_lens
