"""
utility functions needed by all/most TTS systems
"""
import torch
from torch.nn import functional as F
from typing import Optional


def sequence_mask(length: torch.Tensor, max_length: Optional[int] = None) -> torch.Tensor:
    """
    Creates a boolean mask from a length tensor

    Adding additional axes or converting to target dtype might be required,
    depending on the usage

    :param length: [B]
    :param max_length: int
    :return: [B, T]
    """
    if max_length is None:
        max_length = length.max()
    x = torch.arange(max_length, dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)  # noqa, this is a Tensor not a boolean


def convert_pad_shape(pad_shape):
    """
    The torch padding up expects a flat list of pad shapes beginning from the last axis
    Thus we first reverse the order and then flatten

    :param pad_shape: list of [pad_front, pad_end] for each axis
    :return: flat list of padding shapes to be used for torch.nn.functional.pad
    """
    reversed_pad_shape = pad_shape[::-1]
    flat_pad_shape = [item for sublist in reversed_pad_shape for item in sublist]
    return flat_pad_shape


def generate_path(duration: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Convert an N to T duration tensor into a monotonic path grid

    :param duration: [B, N]
    :param mask [B, N, T]
    :return one-hot (like mask) path grid of [B, N, T]
    """
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path
