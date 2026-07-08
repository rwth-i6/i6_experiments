"""
TTS-encoder project -- MAIN RZ recipe (single-GPU).
Owns all RZ experiments:
the audio-only baselines (base-ls*),
and the single-GPU / nep=100 TTS-encoder runs (ref-match + pseudo-enc),
the clean A/B vs Nick's single-GPU 3.53 reference.
Imports ``train_ls_base`` + ``DbMelFeatureExtractor`` from the ``exp2026_05_28_tts_encoder`` library,
and ``_train_tts_encoder`` from the ``_fzj`` recipe.
"""

from __future__ import annotations

import returnn.frontend as rf

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import train_ls_base, DbMelFeatureExtractor
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder_fzj import _train_tts_encoder
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.optim_ext.muon import Muon

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_rz"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # WER comments above each call are %, joint AED+CTC first-pass (aed+ctc/recog-1stpass-res.txt),
    # as {"dev-clean", "dev-other", "test-clean", "test-other"}.
    # --- Audio-only single-GPU baselines (moved here from the exp2026_05_28_tts_encoder library). ---
    # base-ls: standard log-mel, bhv 24 -> same hash as the imported base-librispeech baseline (not re-run).
    # {"dev-clean": 1.87, "dev-other": 4.06, "test-clean": 2.06, "test-other": 4.38}
    train_ls_base("base-ls", prefix=prefix)
    # base-ls-dbmel: DbMel front-end (== frozen GlowTTS space); bhv 25 is behavior-neutral here.
    # {"dev-clean": 1.82, "dev-other": 4.19, "test-clean": 2.06, "test-other": 4.49}
    train_ls_base(
        "base-ls-dbmel", prefix=prefix, feature_extraction=rf.build_dict(DbMelFeatureExtractor), behavior_version=25
    )
    # base-ls-newrtrn: base-ls re-run under the current RZ RETURNN (isolates the RETURNN-version effect).
    # {"dev-clean": 1.86, "dev-other": 4.14, "test-clean": 2.03, "test-other": 4.28}
    train_ls_base("base-ls-newrtrn", prefix=prefix, config_updates_extra={"_meta_hash_trigger": "new-returnn-2026-06"})
    # base-ls-muon: muon-lr5e3-wdbl on the single-GPU log-mel baseline (best FZJ optimizer setting, at RZ).
    train_ls_base(
        "base-ls-muon",
        prefix=prefix,
        base_lr=1.0,
        peak_lr=5e-3,
        config_updates_extra={"optimizer.class": rf.build_dict(Muon)["class"]},
        config_deletes_extra=["optimizer.epsilon"],
    )
    # --- TTS-encoder single-GPU runs ---
    # Reference-match TTS-encoder, SINGLE-GPU / nep=100 (the regime of Nick's 3.53 reference).
    # Identical synth / data to the FZJ tts-enc-ref-match; only num_processes=1 / nep=100 differ.
    # {"dev-clean": 1.71, "dev-other": 4.17, "test-clean": 1.88, "test-other": 4.40}  (base-ls-dbmel 4.19 -> ~0 text gain)
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
    # Faithful reference match: same as ref-match-1gpu but the LOG-MEL ASR front-end (asr_logmel=True).
    # Nick's reference re-extracts at log-mel,
    # so ref-match-1gpu (DbMel default) had a front-end mismatch.
    # tts_waveform_peak_norm=True is required on the log-mel+waveform path
    # (raw log-mel + feature_batch_norm NaNs without peak-normalizing the injected waveform;
    # the DbMel path is fixed-norm and immune).
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
    # Best FZJ mechanism (layer-split pseudo-enc at enc-layer 4 = 3.79 dev-other with muon-nep38 4-GPU),
    # run in the reference's OWN regime: single-GPU / nep=100 / AdamW (NO muon),
    # so it is directly comparable to Nick's 3.53 offline-TTS baseline (same optimizer + regime).
    # Tests whether our layer-split injection beats the reference apples-to-apples.
    # NB muon lifted layer4 a lot on FZJ (nep25 AdamW 4.24 -> muon 3.79),
    # so this AdamW run will likely land above 3.79;
    # the muon version stays the FZJ headline.
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
    # Same as pseudo-enc-layer4-1gpu but NO blanks (pseudo_enc_blank_duration_range=(0, 0)):
    # at layer8 the no-blank / single-blank variants beat interleaved blanks by ~0.15
    # (FZJ muon-nep38: 3.95/3.93 vs 4.11),
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
    # Muon version of pseudo-enc-layer4-1gpu (muon-lr5e3-wdbl):
    # best FZJ mechanism + best optimizer in the single-GPU nep100 regime,
    # the best-number chase, vs the AdamW #1 which is the fair 3.53 comparison.
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
    # Same as pseudo-enc-layer4-noblank-1gpu but batch_size_phon 8k -> 25k (ref-match's default):
    # noblank is ~1.0 enc frame/phoneme (< ref-match's ~1.33),
    # so the 8k throttle (a blanket OOM workaround for the blank-heavy variants) is unnecessary here.
    # 25k halves the steps (~7k, like ref-match) and matches ref-match's text batch,
    # for a clean + fast layer-split-vs-front-end comparison.
    _train_tts_encoder(
        "pseudo-enc-layer4-noblank-25kphon-1gpu",
        prefix=prefix,
        text_train_epoch_split=75,
        batch_size_audio_frames=120_000,
        max_phon_len=300,
        batch_size_phon=25_000,
        asr_logmel=True,
        pseudo_speech_enc=True,
        pseudo_enc_start_layer=4,
        pseudo_enc_blank_duration_range=(0, 0),
        pseudo_enc_specaug_max_width=6,
        num_processes=1,
        gpu_mem=96,
        nep=100,
    )
