"""Audio perturbations and maps from transformed timestamps to the original timeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch


def _resample(x: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    """Resample a 1-D waveform using the best available backend.

    Torchaudio is preferred, Julius is the secondary backend, and linear
    interpolation is retained as a dependency-free fallback.
    """
    if orig_freq == new_freq:
        return x
    try:
        import torchaudio.functional as AF

        return AF.resample(x, orig_freq, new_freq)
    except Exception:
        pass
    try:
        import julius

        return julius.resample_frac(x, int(orig_freq), int(new_freq))
    except Exception:
        pass
    # use non-band-limited interpolation only as a last resort; it is adequate
    # for a robustness stimulus but not for high-fidelity audio processing
    n_new = int(round(x.shape[-1] * new_freq / orig_freq))
    y = torch.nn.functional.interpolate(
        x.reshape(1, 1, -1).float(), size=max(n_new, 1), mode="linear", align_corners=False
    )
    return y.reshape(-1).to(x.dtype)


@dataclass
class Transform:
    """A waveform perturbation and its map back to the original timeline."""

    name: str
    apply: Callable[[torch.Tensor, int], torch.Tensor]  # maps waveform and sample rate to waveform
    phi_inv: Callable[[np.ndarray], np.ndarray]  # maps transformed seconds to original seconds
    kind: str  # reporting category: "shift", "scale", or "unchanged"


def silence_pad_start(delta_s: float = 0.5) -> Transform:
    """Prepend silence and compensate for the resulting timestamp shift."""

    def apply(x: torch.Tensor, sr: int) -> torch.Tensor:
        pad = torch.zeros(int(round(delta_s * sr)), dtype=x.dtype, device=x.device)
        return torch.cat([pad, x], dim=-1)

    return Transform(f"silence+{int(delta_s * 1000)}ms", apply, lambda t: t - delta_s, "shift")


def speed(rate: float) -> Transform:
    """Change playback speed while preserving the output sample rate.

    A rate above one shortens the waveform, placing an original onset ``t`` at
    ``t / rate``. The inverse map therefore multiplies timestamps by ``rate``.
    """
    def apply(x: torch.Tensor, sr: int) -> torch.Tensor:
        return _resample(x, int(round(sr * rate)), sr)

    return Transform(f"speed x{rate:g}", apply, lambda t: t * rate, "scale")


def gain(factor: float) -> Transform:
    """Scale waveform amplitude and clip the result to the valid sample range."""

    def apply(x: torch.Tensor, sr: int) -> torch.Tensor:
        return (x * factor).clamp(-1.0, 1.0)

    return Transform(f"gain x{factor:g}", apply, lambda t: t, "unchanged")


def additive_noise(snr_db: float = 20.0, seed: int = 0) -> Transform:
    """Add white noise at the requested per-utterance SNR.

    A fixed seed makes the perturbation reproducible.
    """
    def apply(x: torch.Tensor, sr: int) -> torch.Tensor:
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(x.shape, generator=g, dtype=torch.float32)
        sig_p = float((x.float() ** 2).mean()) + 1e-12
        noise_p = float((noise ** 2).mean()) + 1e-12
        scale = (sig_p / (noise_p * (10.0 ** (snr_db / 10.0)))) ** 0.5
        # generate noise on the CPU so the seeded samples do not depend on the
        # waveform's device, then move it to the waveform before addition
        return (x + scale * noise.to(device=x.device, dtype=x.dtype)).clamp(-1.0, 1.0)

    return Transform(f"noise {int(snr_db)}dB", apply, lambda t: t, "unchanged")


def resample_roundtrip(inter_freq: int = 8000) -> Transform:
    """Resample through an intermediate rate and back without shifting timestamps."""

    def apply(x: torch.Tensor, sr: int) -> torch.Tensor:
        return _resample(_resample(x, sr, inter_freq), inter_freq, sr)

    return Transform(f"resample {inter_freq}Hz", apply, lambda t: t, "unchanged")


def default_transforms() -> List[Transform]:
    """Return the standard stability-evaluation perturbation suite."""
    return [
        silence_pad_start(0.5),
        speed(1.05),
        speed(0.95),
        gain(0.7),
        gain(1.3),
        additive_noise(20.0),
        resample_roundtrip(8000),
    ]


ORIG = "orig"  # unmodified audio used as the perturbation reference


def transform_by_name(name: str) -> Optional[Transform]:
    """Resolve a standard transform name, using ``None`` for :data:`ORIG`.

    Both alignment generation and metric aggregation use this lookup so they
    apply the same perturbation and inverse timeline map.
    """
    if name == ORIG:
        return None
    for t in default_transforms():
        if t.name == name:
            return t
    raise ValueError(f"unknown transform {name!r}, known: {[ORIG] + [t.name for t in default_transforms()]}")


def transform_names() -> List[str]:
    """Return the reference name followed by all standard transform names."""

    return [ORIG] + [t.name for t in default_transforms()]


def apply_to_unpadded(
    wav: torch.Tensor, sample_rate: int, *, pad_samples: int, transform: Transform
) -> torch.Tensor:
    """Transform only the real speech of a delay-padded 1-D waveform, then re-pad with fresh zeros."""
    if pad_samples <= 0:
        return transform.apply(wav, sample_rate)
    speech = wav[..., : wav.shape[-1] - pad_samples]
    out = transform.apply(speech, sample_rate)
    return torch.cat([out, out.new_zeros(pad_samples)], dim=-1)
