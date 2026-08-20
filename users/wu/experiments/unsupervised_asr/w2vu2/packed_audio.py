"""Audio-domain compatibility helpers for the packed LibriSpeech decoder."""

from __future__ import annotations

import numpy as np


def reconstruct_pcm16(wave: np.ndarray) -> np.ndarray:
    """Project decoded Ogg samples onto the PCM16 lattice used by the legacy FLAC input.

    The HF/Ogg corpus was encoded from LibriSpeech's PCM16 FLAC.  Tiny Vorbis decode errors can
    change a tied CTC+KenLM beam decision, so the packed path restores the source quantization before
    applying the checkpoint's waveform normalization.  This does not stage or read a FLAC file.
    """
    wave = np.asarray(wave)
    if wave.ndim != 1 or not np.issubdtype(wave.dtype, np.floating):
        raise ValueError(f"expected a floating mono waveform, got {wave.shape} {wave.dtype}")
    quantized = np.clip(np.rint(wave.astype(np.float64) * 32768.0), -32768, 32767)
    return (quantized / 32768.0).astype(np.float32)
