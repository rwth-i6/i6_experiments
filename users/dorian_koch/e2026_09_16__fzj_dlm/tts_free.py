"""
TTS-free DLM training data (Albert Zeyer's proposal, 2026-09-22).

The DLM data builder turns the LM-text shards into ASR hypotheses via TTS audio:
phonemes -> GlowTTS + vocoder -> waveform -> the winner (feature extraction, encoder, layer-16 CTC).
The winner is a text-injection model: during training it already turns text into input features itself,
with its frozen :class:`PseudoSpeechEncoder` (per-phone mean log-mel from MFA statistics, median durations
x 0.7 with lognormal jitter, lerp). Here the LM-text hypotheses come from exactly that path instead:
phonemes -> ``model.pseudo_enc`` -> the same encoder + CTC. No TTS model at all.

Everything else is left as it is: the LM-text shards, lexicon and phoneme vocab (the TTS dataset and the winner
use the same ``MergeLexiconJob.P8go21pxx40e`` / ``ReturnnVocabFromPhonemeInventory.z2RlZd9Y0jWQ``), the hyps
perturbations (dropout, SpecAugment, mixup of ``GetCtcHypsCfgV6``), the greedy CTC decoding and output format,
and the real-audio (LibriSpeech) hypotheses. The phone_info is the winner's own training one
(between-word silence 0.15, begin/end 0.01, random pronunciation) instead of the GlowTTS-compatible one,
so the pseudo encoder sees text exactly as in its training.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Dict, Optional

# The winner's text-injection phone_info (its returnn.config ``glow_tts_phone_info``), as overrides of the
# TTS dataset's train defaults (tts_model.get_tts_model_dataset_dict: 0.01 / 0.95 / 0.01).
WINNER_TRAIN_PHONE_INFO: Dict[str, Any] = {
    "add_silence_beginning": 0.01,
    "add_silence_between_words": 0.15,
    "add_silence_end": 0.01,
}


def get_tts_free_opts():
    """
    The ``hyps_tts_opts`` for the builder: only phone_info matters (it shapes the phoneme LmDataset);
    the TTS model options are dropped by :func:`get_asr_pseudo_enc_model_def`.
    """
    from denoising_lm_2024.sis_recipe.tts_model import get_tts_opts_default_model

    return get_tts_opts_default_model({"phone_info": dict(WINNER_TRAIN_PHONE_INFO)}, compatible_to_nick=False)


def asr_pseudo_enc_model_def(*, asr_model_def, **kwargs):
    """The winner as is (to be bound with functools.partial); it must carry its pseudo encoder."""
    model = asr_model_def(**kwargs)
    assert getattr(model, "pseudo_enc", None) is not None, "needs pseudo_speech_enc=True in the model config"
    return model


def get_asr_pseudo_enc_model_def(*, asr_model_def, tts_opts=None):
    """
    Stand-in for :func:`denoising_lm_2024.sis_recipe.tts_model.get_asr_with_tts_model_def`:
    no TTS model and no ``preload_from_files``; batch_size_factor 1 as there (the input is phonemes, not audio).
    """
    from i6_experiments.users.zeyer.model_interfaces import ModelDefWithCfg

    del tts_opts  # only its phone_info is used, by the dataset
    config: Dict[str, Any] = {}
    if isinstance(asr_model_def, ModelDefWithCfg):
        config.update(asr_model_def.config)
        asr_model_def = asr_model_def.model_def
    combined = functools.partial(asr_pseudo_enc_model_def, asr_model_def=asr_model_def)
    combined.behavior_version = asr_model_def.behavior_version
    combined.backend = asr_model_def.backend
    combined.batch_size_factor = 1
    return ModelDefWithCfg(model_def=combined, config=config)


def _identity_frontend(source, *, in_spatial_dim):
    return source, in_spatial_dim


def model_recog_pseudo_enc_v6(source, *, in_spatial_dim, model, cfg):
    """
    Stand-in for ``ctc.model_recog_single_v6`` on phoneme input. Function is run within RETURNN.

    Same perturbations (``_apply_v6_perturbation_cfg``, ``_apply_v4_cfg``) and the same greedy CTC decoding
    (``ctc.model_recog_single``, i.e. ``model.encode_and_get_ctc_log_probs``); only the audio front-end
    (pad_audio + log-mel) is replaced by the pseudo encoder's features.
    """
    import returnn.frontend as rf
    from denoising_lm_2024.sis_recipe.ctc import _apply_v4_cfg, _apply_v6_perturbation_cfg, model_recog_single

    assert cfg.sample_method == "greedy" and cfg.sample_opts is None, cfg
    assert getattr(model, "tts_model", None) is None
    if cfg.data_perturbation_opts:
        _apply_v6_perturbation_cfg(model, cfg)  # mixup hook, runs inside encode_from_features
    source, in_spatial_dim, train_flag_funcs = _apply_v4_cfg(source, in_spatial_dim, model, cfg)

    # The duration jitter is gated by the GLOBAL train flag (pseudo_enc: rf.where(train, jitter, 1.0)),
    # which the forward keeps False; without it every hypothesis would use the median durations.
    with rf.get_run_ctx().train_flag_ctx(True):
        feats, feats_spatial_dim = model.pseudo_enc(source, spatial_dim=in_spatial_dim)
    if feats.dtype != "float32":
        feats = rf.cast(feats, "float32")
    feats.feature_dim = model.in_dim

    orig_frontend, orig_pad_audio = model.feature_extraction, model.pad_audio
    model.feature_extraction, model.pad_audio = _identity_frontend, None
    try:
        model_recog_single(feats, in_spatial_dim=feats_spatial_dim, model=model, train_flag_funcs=train_flag_funcs)
    finally:
        model.feature_extraction, model.pad_audio = orig_frontend, orig_pad_audio


@contextlib.contextmanager
def tts_free_hyps():
    """
    Within this context, the data builder's LM-text path (the only caller that passes ``tts_opts``)
    produces pseudo-encoder hypotheses; the real-audio path is untouched.
    """
    import unittest.mock
    from denoising_lm_2024.sis_recipe import ctc as _ctc
    from denoising_lm_2024.sis_recipe import error_correction_model_gen_train_data as _gen
    from denoising_lm_2024.sis_recipe import tts_model as _tts

    orig = _gen.sis_get_ctc_hyps_single_split

    def _patched(*args, **kwargs):
        if kwargs.get("tts_opts") is None:
            return orig(*args, **kwargs)
        with unittest.mock.patch.object(_ctc, "model_recog_single_v6", model_recog_pseudo_enc_v6), (
            unittest.mock.patch.object(_tts, "get_asr_with_tts_model_def", get_asr_pseudo_enc_model_def)
        ):
            return orig(*args, **kwargs)

    with unittest.mock.patch.object(_gen, "sis_get_ctc_hyps_single_split", _patched):
        yield
