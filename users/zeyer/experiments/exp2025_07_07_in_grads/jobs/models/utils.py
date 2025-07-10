from typing import Optional, Tuple
import torch


# noinspection PyShadowingBuiltins
def apply_input_slice(input: torch.Tensor, slice: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    :param input: [B,T,F]
    :param slice: (start, end) where start and end are tensors of shape [B], or scalars.
    :return: sliced input tensor, [B,T',F], where T' is the length of the slice
    """
    if not slice:
        return input
    start, end = slice
    if start.ndim == end.ndim == 0:  # scalars, can use simple code
        return input[:, start:end]  # [B,T',F]
    if start.numel() == end.numel() == 1:  # single-element tensors, can use simple code
        return input[:, start.item() : end.item()]

    slice_len = (end - start).max()
    slice_indices = torch.arange(slice_len)  # [T']
    indices = start[:, None] + slice_indices[None, :]  # [B, T']
    indices = indices.unsqueeze(2).expand(-1, -1, input.shape[2])  # [B, T', F]
    return torch.gather(input, 1, indices)
