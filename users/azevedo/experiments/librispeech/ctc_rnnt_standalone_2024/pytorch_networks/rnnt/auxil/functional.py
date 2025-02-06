import torch
from typing import Optional, Union, Tuple
from enum import Enum

from ..decoder.chunk_handler import AudioStreamer, StreamingASRContextManager


class Mode(Enum):
    STREAMING = 0
    OFFLINE = 1


class TrainingStrategy(Enum):
    STREAMING = 0
    UNIFIED = 1
    SWITCHING = 2
    OFFLINE = 3
    INTRA_BATCH = 4



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

    def roll_cat(left, last_val):
        # for each chunk i we want to concatenate it with lookahead frames of chunk i+1
        # i    ||  i+1[:lookahead_size]
        # i+1  ||  i+2[:lookahead_size]
        right = torch.roll(left[:, :, :lookahead_size], shifts=-1, dims=1)
        right[:, -1] = last_val  # last chunk has no lookahead
        return torch.cat((left, right), dim=2)

    # lookahead (assumes lookahead_size < chunk_size)
    x = roll_cat(x, 0)  # (B, N, C+R, F')
    # adjust sequence mask
    sequence_mask = roll_cat(sequence_mask, False)  # (B, N, C+R)

    return x, sequence_mask


def add_lookahead_v2(x: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int):
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


def get_future_ac_ctx(
        x: torch.Tensor, lengths: torch.Tensor, lookahead_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        x (torch.Tensor): (B, N, C) chunked raw audio tensor
        lengths (torch.Tensor): (B) lengths of NON-chunked raw audio tensor
        lookahead_size (int): num of future samples to add to each chunk

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (B, N, C+R) chunked + FACtx raw audio tensor, (B, N) effective lengths of FACtx
    """
    if lookahead_size <= 0:
        return x

    batch_sz = x.size(0)
    num_chunks = x.size(1)
    chunk_sz = x.size(2)

    future_ac_ctx = torch.zeros(batch_sz, num_chunks, lookahead_size, device=x.device)
    added_lengths = torch.zeros(batch_sz, num_chunks, device=lengths.device, dtype=lengths.dtype)

    x = x.view(batch_sz, -1)  # [B, T]
    for i in range(num_chunks-1):
        next_chunk = chunk_sz*(i+1)
        future_ac_ctx[:, i, :x.size(1)-next_chunk] = x[:, next_chunk: next_chunk+lookahead_size]

        # calculate length of future acoustic context for current chunks
        rem_lengths = lengths - next_chunk
        rem_lengths[rem_lengths < 0] = 0
        rem_lengths[rem_lengths > lookahead_size] = lookahead_size
        added_lengths[:, i] = rem_lengths

    return future_ac_ctx, added_lengths


def num_samples_to_frames(n_fft, hop_length, center, num_samples: int) -> int:
    if center:
        return (num_samples // hop_length) + 1
    else:
        return ((num_samples - n_fft) // hop_length) + 1


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


def pad_chunk_frames(self, tensor, mask, chunk_size):
    batch_size = tensor.size(0)

    # pad conformer time-dim to be able to chunk (by reshaping) below
    time_dim_pad = -tensor.size(1) % chunk_size
    # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
    tensor_pad = torch.nn.functional.pad(tensor, (0, 0, 0, time_dim_pad),
                                            "constant", 0)
    mask_pad = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

    # separate chunks to signal the conformer that we are chunking input
    tensor_pad = tensor_pad.view(batch_size, -1, chunk_size,
                                        tensor_pad.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
    mask_pad = mask_pad.view(batch_size, -1, chunk_size)  # (B, N, C)

    return tensor_pad, mask_pad


def rotary_pos_encoding(
        x: torch.Tensor, start_time: int = 0
) -> torch.Tensor:
    """
    Implements Rotary Positional Embedding (RoPE) as presented in
    "Roformer: Enhanced transformer with rotary position embedding."

    :param x: [B, T, d]
    :param start_time: start index of first frame
    """
    d = x.size(-1)
    T = x.size(-2)
    assert d % 2 == 0, "Err: dimension of input must be even."

    pos = torch.arange(T, dtype=torch.float, device=x.device).unsqueeze(-1) + start_time  # [T, 1]
    thetas = torch.pow(10e4,
                       -2 * (torch.cos(torch.arange(d//2, device=x.device) - 1)) / d).unsqueeze(0)  # [1, d/2]

    trig_in = torch.matmul(pos, thetas)  # [T, d/2]
    cos_vals = torch.cos(trig_in)
    cos_vals = torch.repeat_interleave(cos_vals, 2, dim=-1).unsqueeze(0)  # [1, T, d]

    sin_vals = torch.sin(trig_in)
    sin_vals = torch.repeat_interleave(sin_vals, 2, dim=-1).unsqueeze(0)  # [1, T, d]

    # alternating -sin and sin
    m = torch.arange(d, device=x.device)
    m = torch.where(m % 2 == 0, -1, 1)[None, None]
    sin_vals = sin_vals * m

    output_value = x * cos_vals + x * sin_vals  # [B, T, d]

    return output_value


def process_offline_sample(
        raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, run_ctx
) -> torch.Tensor:
    hypothesis, _ = run_ctx.rnnt_decoder.infer(
        input=raw_audio[None],
        length=raw_audio_len[None],
        beam_width=run_ctx.beam_size,
    )

    return hypothesis[0][0][:-1]

def process_streaming_sample(
        raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, config, run_ctx
) -> torch.Tensor:
    chunk_streamer = AudioStreamer(
        raw_audio=raw_audio[None],
        raw_audio_len=raw_audio_len[None],
        left_size=config.chunk_size,
        right_size=config.lookahead_size,
        stride=config.stride,
        pad_value=config.pad_value,
    )

    hypothesis = None
    # context manager for handling states of streaming inference (only need to pass chunk and its effective size)
    with StreamingASRContextManager(run_ctx.rnnt_decoder, carryover_sz=config.carry_over_size, beam_sz=run_ctx.beam_size) as cm:
        for chunk, eff_chunk_sz in chunk_streamer:
            _, hypothesis = cm.process_chunk(ext_chunk=chunk, eff_chunk_sz=eff_chunk_sz) 

    return hypothesis[0][0][:-1]