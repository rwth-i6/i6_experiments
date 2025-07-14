from typing import Optional, Tuple, Union
import torch


# noinspection PyShadowingBuiltins
def dyn_slice(
    input: torch.Tensor, slice: Optional[Tuple[Union[int, torch.Tensor], Union[int, torch.Tensor]]]
) -> torch.Tensor:
    """
    :param input: [B,T,...]
    :param slice: (start, end) where start and end are tensors of shape [B], or scalars.
    :return: sliced input tensor, [B,T',F], where T' is the length of the slice
    """
    if not slice:
        return input
    start, end = slice
    if isinstance(start, int) and isinstance(end, int):  # scalars, can use simple code
        return input[:, start:end]  # [B,T',...]
    if not isinstance(start, torch.Tensor):
        start = torch.tensor(start, device=input.device, dtype=input.dtype)
    if not isinstance(end, torch.Tensor):
        end = torch.tensor(end, device=input.device, dtype=input.dtype)
    if start.ndim == end.ndim == 0:  # scalars, can use simple code
        return input[:, start:end]  # [B,T',...]
    if start.numel() == end.numel() == 1:  # single-element tensors, can use simple code
        return input[:, start.flatten()[0] : end.flatten()[0]]  # [B,T',...]
    assert start.shape == end.shape == input.shape[:1]

    slice_len = (end - start).max()
    slice_indices = torch.arange(slice_len)  # [T']
    indices = start[:, None] + slice_indices[None, :]  # [B, T']
    if input.ndim > 2:
        indices = indices.reshape(*(indices.shape + (1,) * (input.ndim - 2)))  # [B, T', 1, 1, ...]
        indices = indices.expand(-1, -1, *input.shape[2:])  # [B, T', ...]
    return torch.gather(input, 1, indices)
