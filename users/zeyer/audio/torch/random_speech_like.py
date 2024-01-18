"""
Generated random speech like data.

Based on: https://github.com/albertz/playground/blob/master/create-random-speech-like-sound.py
"""


from typing import Optional
import torch


def generate_random_speech_like_audio(
    batch_size: int,
    num_frames: int,
    *,
    samples_per_sec: int = 16_000,
    frequency: float = 150.0,
    num_random_freqs_per_sec: int = 15,
    amplitude: float = 0.3,
    amplitude_frequency: Optional[float] = None,
) -> torch.Tensor:
    """
    generate audio

    :return: shape [batch_size,num_frames]
    """
    frame_idxs = torch.arange(num_frames, dtype=torch.int64)  # [T]

    samples = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=num_random_freqs_per_sec,
    )  # [T,B]

    if amplitude_frequency is None:
        amplitude_frequency = frequency / 75.0
    amplitude_variations = _integrate_rnd_frequencies(
        batch_size,
        frame_idxs,
        base_frequency=amplitude_frequency,
        samples_per_sec=samples_per_sec,
        num_random_freqs_per_sec=amplitude_frequency,
    )  # [T,B]

    samples *= amplitude * (0.666 + 0.333 * amplitude_variations)
    return samples.permute(1, 0)  # [B,T]


def _integrate_rnd_frequencies(
    batch_size: int,
    frame_idxs: torch.Tensor,
    *,
    base_frequency: float,
    samples_per_sec: int,
    num_random_freqs_per_sec: float,
) -> torch.Tensor:
    rnd_freqs = torch.empty(
        size=(int(len(frame_idxs) * num_random_freqs_per_sec / samples_per_sec) + 1, batch_size),
        dtype=torch.float32,
    )  # [T',B]
    torch.nn.init.trunc_normal_(rnd_freqs, a=-1.0, b=1.0)
    rnd_freqs = (rnd_freqs * 0.5 + 1.0) * base_frequency  # [T',B]

    freq_idx_f = (frame_idxs * num_random_freqs_per_sec) / samples_per_sec
    freq_idx = freq_idx_f.to(torch.int64)
    next_freq_idx = torch.clip(freq_idx + 1, 0, len(rnd_freqs) - 1)
    frac = (freq_idx_f % 1)[:, None]  # [T,1]
    freq = rnd_freqs[freq_idx] * (1 - frac) + rnd_freqs[next_freq_idx] * frac  # [T,B]

    ts = torch.cumsum(freq / samples_per_sec, dim=0)  # [T,B]
    return torch.sin(2 * torch.pi * ts)
