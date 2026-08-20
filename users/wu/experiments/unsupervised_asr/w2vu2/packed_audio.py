"""Audio-domain compatibility helpers for the packed LibriSpeech decoder."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf


def legacy_flac_bytes(wave: np.ndarray, *, sample_rate: int = 16_000) -> bytes:
    """Apply the legacy HF-array -> PCM16-FLAC boundary entirely in memory.

    ``LibriStAudioJob`` decoded Hugging Face audio first and then called ``sf.write`` with
    ``format="FLAC"``. Decoding the original Ogg bytes directly with libsndfile is not equivalent:
    the Hugging Face decoder differs at a few half-LSB samples, and tied CTC+KenLM beams can amplify
    those differences into another transcript. Repeating the original conversion in memory gives
    the worker exactly the old PCM16 samples without creating a per-utterance file tree.
    """
    wave = np.asarray(wave, dtype=np.float32)
    if wave.ndim != 1 or not wave.size:
        raise ValueError(f"expected a nonempty mono waveform, got {wave.shape}")
    if int(sample_rate) != 16_000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")
    encoded = io.BytesIO()
    sf.write(encoded, wave, sample_rate, format="FLAC")
    return encoded.getvalue()


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
