"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/vad_port.py (``rvad_silence``,
``FPS``, ``SUBFRAMES`` only).

rVADfast (utterance-level VAD) -> encoder-rate frame silence masks.  The source module also carried
the SAE 1.0 validation gate against MFA gold silence (``RVADValidationJob``, ``validate_records``,
``gold_silence_25hz``, speech-segment timestamps, a gilkeyio audio reader); none of it feeds the
phase-4a VAD streams, so it is not ported.  In the source, :class:`..vad.BlankfreeVadHdfJob` loaded
this file by path (``vad_port_path``); here it is a normal import.
"""

from __future__ import annotations

import numpy as np

__all__ = ["FPS", "SUBFRAMES", "rvad_silence"]

FPS = 25.0
SUBFRAMES = 4  # rVADfast 10 ms frames per 40 ms (25 Hz) frame


def rvad_silence(wav: np.ndarray, sr: int = 16000, vad_threshold: float = 0.4,
                 vad=None, subframes: int = SUBFRAMES) -> np.ndarray:
    """Waveform -> encoder-rate bool mask (True = silence). Aggregates rVADfast 10 ms speech labels
    (1=speech) to `subframes`-long frames; a frame is silence iff > 50 % of its subframes are
    non-speech. subframes = 100 Hz / encoder_fps (4 -> 25 Hz, 2 -> 50 Hz)."""
    from rVADfast import rVADfast

    if vad is None:
        vad = rVADfast(vad_threshold=vad_threshold)
    labels, _ = vad(np.asarray(wav, dtype=np.float32), sr)
    n = len(labels) // subframes
    if n == 0:
        return np.zeros(0, dtype=bool)
    return labels[: n * subframes].reshape(n, subframes).mean(1) < 0.5
