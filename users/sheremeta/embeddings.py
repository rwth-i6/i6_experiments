"""Reusable embedding modules shared across the DSM-ASR model variants."""

import math

import torch
from torch import nn


class ContinuousScalarEmbedding(nn.Module):
    """Sinusoidal embedding of a continuous scalar into an additive conditioning vector (B,) -> (B, out_dim)."""

    freqs: torch.Tensor

    def __init__(
        self,
        out_dim: int,
        *,
        min_wavelength: float,
        max_wavelength: float,
        embed_dim: int = 64,
        zero_init: bool = True,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        assert 0 < min_wavelength < max_wavelength, "need 0 < min_wavelength < max_wavelength"
        self.embed_dim = embed_dim

        half = embed_dim // 2
        wavelengths = torch.exp(
            torch.linspace(math.log(float(min_wavelength)), math.log(float(max_wavelength)), half)
        )
        freqs = (2.0 * math.pi) / wavelengths
        # persistent=False so this derived buffer never lands in a checkpoint
        self.register_buffer("freqs", freqs, persistent=False)

        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Linear(embed_dim, out_dim)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        v = value.to(self.freqs.dtype).unsqueeze(-1)
        ang = v * self.freqs.unsqueeze(0)
        pe = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
        return self.proj(self.norm(pe))
