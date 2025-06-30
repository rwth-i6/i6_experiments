import torch
from enum import Enum
from typing import Optional, Union



class Mode(Enum):
    STREAMING = 0
    OFFLINE = 1


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


def add_lookahead(x: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int):
    if lookahead_size <= 0:
        return x, sequence_mask

    batch_sz = x.size(0)
    num_chunks = x.size(1)
    chunk_sz = x.size(2)
    feat_dim = x.size(3)

    future_ac_ctx = torch.zeros(batch_sz, num_chunks, lookahead_size, feat_dim, device=x.device)
    seq_mask = torch.zeros(batch_sz, num_chunks, lookahead_size, dtype=torch.bool, device=sequence_mask.device)

    x = x.view(batch_sz, -1, feat_dim)  # [B, T, F]
    sequence_mask = sequence_mask.view(batch_sz, -1)
    for i in range(num_chunks-1):
        next_chunk = chunk_sz*(i+1)

        future_ac_ctx[:, i, :x.size(1)-next_chunk] = x[:, next_chunk: next_chunk+lookahead_size]
        seq_mask[:, i, :x.size(1)-next_chunk] = sequence_mask[:, next_chunk: next_chunk + lookahead_size]

    x = x.view(batch_sz, num_chunks, chunk_sz, feat_dim)
    sequence_mask = sequence_mask.view(batch_sz, num_chunks, chunk_sz)

    # [B, N, C', F'] -> [B, N, C'+R, F']
    extended_chunk = torch.cat((x, future_ac_ctx), dim=2)
    extended_seq_mask = torch.cat((sequence_mask, seq_mask), dim=2)

    return extended_chunk, extended_seq_mask


def create_chunk_mask(
        seq_len: int, chunk_size: int, lookahead_size: int = 0,
        carry_over_size: Optional[int] = None, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    chunk_size := num. subsampled frames in one chunk
    seq_len = N * (chunk_size + lookahead_size) with N = #chunks
    output of some embed may see every embed in the past and in the current chunk
    """
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    chunk_ext_size = chunk_size + lookahead_size

    if carry_over_size is None:
        carry_over = chunk_ext_size
    else:
        carry_over = int(carry_over_size * chunk_ext_size)

    for i in range(0, seq_len, chunk_ext_size):
        # attend to carry_over
        attn_mask[i:i + chunk_ext_size, max(0, i - carry_over):i] = True
        if lookahead_size > 0:
            # don't attend to their lookahead
            attn_mask[i:i + chunk_ext_size].view(chunk_ext_size, -1, chunk_ext_size)[:, :, -lookahead_size:] = False
        # attend to current chunk and its lookahead
        attn_mask[i:i + chunk_ext_size, i:i + chunk_ext_size] = True

    return attn_mask


def num_samples_to_frames(n_fft, hop_length, center, num_samples: int) -> int:
    if center:
        return (num_samples // hop_length) + 1
    else:
        return ((num_samples - n_fft) // hop_length) + 1
