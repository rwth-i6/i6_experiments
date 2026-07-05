"""
TTS-encoder project, RZ single-GPU variant.

One experiment: the reference-match TTS-encoder trained SINGLE-GPU (nep=100) on the RZ GPUs, to
isolate the multi-GPU / few-updates penalty seen in the FZJ 4-GPU runs (see _fzj.py). Same
architecture / data / synth config as the FZJ ``tts-enc-ref-match`` (textP75, waveform path, noise
0.7 / length 1.0 fixed); the ONLY difference is the GPU regime:
  FZJ ref-match: 4-GPU DDP, nep=25 (x4 -> ~100 effective passes), batched recog.
  this run:      1-GPU,     nep=100,                              non-batched recog.
Nick's 3.53 reference was trained single-GPU on these same RZ GPUs, so this is the clean A/B.

The recipe entry lives here (RZ-specific); the heavy data / model / train-step builder
``_train_tts_encoder`` is imported from the FZJ recipe.
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_rz"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # Reference-match TTS-encoder, SINGLE-GPU / nep=100 (the regime of Nick's 3.53 reference).
    # Identical synth / data to the FZJ tts-enc-ref-match; only num_processes=1 / nep=100 differ.
    _train_tts_encoder(
        "tts-enc-ref-match-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        tts_waveform=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Faithful reference match: same as ref-match-1gpu but the LOG-MEL ASR front-end (asr_logmel=True) --
    # Nick's reference re-extracts at log-mel, so ref-match-1gpu (DbMel default) had a front-end mismatch.
    # tts_waveform_peak_norm=True is required on the log-mel+waveform path (raw log-mel + feature_batch_norm
    # NaNs without peak-normalizing the injected waveform; the DbMel path is fixed-norm and immune).
    _train_tts_encoder(
        "tts-enc-ref-match-logmel-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        tts_waveform=True,
        asr_logmel=True,
        tts_waveform_peak_norm=True,
        glow_tts_noise_scale_range=(0.7, 0.7),
        glow_tts_length_scale_range=(1.0, 1.0),
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
