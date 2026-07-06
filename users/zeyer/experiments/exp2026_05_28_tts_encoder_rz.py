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

import returnn.frontend as rf

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

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
    # Best FZJ mechanism (layer-split pseudo-enc at enc-layer 4 = 3.79 dev-other with muon-nep38 4-GPU) run
    # in the reference's OWN regime: single-GPU / nep=100 / AdamW (NO muon), so it is directly comparable to
    # Nick's 3.53 offline-TTS baseline (same optimizer + regime). Tests whether our layer-split injection
    # beats the reference apples-to-apples. NB muon lifted layer4 a lot on FZJ (nep25 AdamW 4.24 -> muon 3.79),
    # so this AdamW run will likely land above 3.79; the muon version stays the FZJ headline.
    _train_tts_encoder(
        "pseudo-enc-layer4-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Same as pseudo-enc-layer4-1gpu but NO blanks (pseudo_enc_blank_duration_range=(0, 0)): at layer8 the
    # no-blank / single-blank variants beat interleaved blanks by ~0.15 (FZJ muon-nep38: 3.95/3.93 vs 4.11),
    # so applying that at the best depth (layer4) in the reference regime is our most promising unrun config.
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
    # Muon version of pseudo-enc-layer4-1gpu (muon-lr5e3-wdbl): best FZJ mechanism + best optimizer in the
    # single-GPU nep100 regime -- the best-number chase, vs the AdamW #1 which is the fair 3.53 comparison.
    _train_tts_encoder(
        "pseudo-enc-layer4-muon-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=8_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
        base_lr=1.0,
        peak_lr=5e-3,
        extra_config_updates={"optimizer.class": rf.build_dict(Muon)["class"]},
        extra_config_deletes=["optimizer.epsilon"],
    )
