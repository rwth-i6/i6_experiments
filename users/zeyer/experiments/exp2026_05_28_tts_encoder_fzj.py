"""
TTS-encoder project, FZJ (Juelich) variant -- larger-scale runs.

- ``base-ls``: standard log-mel CTC+AED baseline. Same Job hash as RZ / the FZJ base setup -> imports.
- ``tts-enc-v1``: the TTS-encoder text-util training. Standard ``exp2024_04_23_baselines.aed.Model`` on the DbMel
  front-end + a frozen GlowTTS attached as ``model.tts`` (reusable wrapper in ``external_models/glow_tts.py``).
  Trained on paired LS audio + text-only LS-LM data via ``alternate_batching``:
    * audio batch:     raw audio -> model.encode -> aux-CTC + AED-CE.
    * text-only batch: phonemes  -> model.tts (frozen) -> log-mel -> model.encode_from_features -> same losses.

  ``aed_glowtts_model_def`` builds the model (frozen GlowTTS as model.tts); ``aed_glowtts_train_step`` is a custom
  RETURNN train_step doing its own extern_data extraction (incl. the 3rd "phonemes" stream the default train_v4
  step does not forward) + the dual-branch losses -- so no separate TrainDef is needed (passed as
  ``config["train_step"]``; train_v4/train_exp only fall back to the default TrainDef when no custom step is set).

  Data: a ``CombinedDataset`` interleaving
    "asr"  = LS OggZip (audio + spm, speed-perturbed); phonemes auto zero-filled (empty) by CombinedDataset.
    "text" = one ``LmDataset`` emitting the raw utf8 bytes of each LS-LM line, wrapped in a ``PostprocessingDataset``
             whose ``map_seq`` derives BOTH the spm target and the GlowTTS phonemes from that text (single corpus
             read; spm + phone_info opts passed via the config). spm[i] & phon[i] are trivially the same line.
  CV (dev/devtrain) stays the baseline audio-only sets, wrapped in a ``PostprocessingDataset`` that adds an empty
  ``phonemes`` stream so they satisfy the extern_data contract (every declared key required in every batch, incl.
  CV). At CV ``have_audio=True`` so the empty phonemes is never used.

NOTE: the data wiring (the text map_seq tokenization, CombinedDataset+alternate_batching, the CV PostprocessingDataset
map_seq, and the text-branch rf graph) is graph-buildable but NOT unit-testable -- validate with a 1-step smoke
run (``time_rqmt<=1``) before a full run. See projects/2026-05-28-tts-encoder.md.

Run from an FZJ setup (py7 is an RZ-only alias; on FZJ use the wrap.sh launcher, which does the module load):
``/e/project1/spell/zeyer1/py-envs/py3.13-torch2.12/wrap.sh python ./sis m recipe/i6_experiments/users/zeyer/experiments/exp2026_05_28_tts_encoder_fzj.py``
"""

from __future__ import annotations

import copy
import dataclasses
import functools
from typing import Optional, Dict, Any

import returnn.frontend as rf
from returnn.tensor import Dim

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.model_interfaces import ModelDef
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import aed as _aed
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import Model
from i6_experiments.users.zeyer.external_models.glow_tts import GlowTtsLogMel, get_glow_tts_phoneme_vocab_size
from i6_experiments.users.zeyer.experiments.exp2026_05_28_tts_encoder import _train_ls_base, DbMelFeatureExtractor

__all__ = ["py"]
__setup_root_prefix__ = "exp2026_05_28_tts_encoder_fzj"

PHONEMES_DATA_KEY = "phonemes"


def py():
    prefix = get_setup_prefix_for_module(__name__)
    # Standard log-mel baseline. Same Job hash as RZ base-ls / the FZJ base setup -> imports, not re-trained.
    _train_ls_base("base-ls", prefix=prefix)
    # TTS-encoder text-util training (single-GPU first, like base).
    _train_tts_encoder("tts-enc-v1", prefix=prefix)

    # TODO: import the finished RZ base-ls-dbmel (ReturnnTrainingJob.8mdaueLDfiGP); do NOT re-train on FZJ.


def _train_tts_encoder(
    name: str,
    *,
    prefix: str,
    text_train_epoch_split: int = 20,
    txt_only_loss_scale: float = 1.0,
    glow_tts_length_scale_range=(0.7, 1.1),
    glow_tts_noise_scale_range=(0.3, 0.9),
):
    from returnn.frontend.decoder.transformer import TransformerDecoder
    from returnn.frontend.encoder.conformer import (
        ConformerEncoder,
        ConformerEncoderLayer,
        ConformerConvSubsample,
        ConformerPositionwiseFeedForward,
    )
    from returnn_common.datasets_old_2022_10.interface import DatasetConfigStatic
    from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_raw_v2,
        get_vocab_by_str,
        get_train_corpus_text,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
        train_exp as aed_train_exp,
        _raw_sample_rate,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
        aed_ctc_timesync_recog_recomb_auto_scale,
    )
    from i6_experiments.users.zeyer.returnn.alternate_batching import alternate_batching
    from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
    from i6_experiments.users.zeyer.external_models.glow_tts import (
        get_glow_tts_phone_info,
        get_glow_tts_phoneme_extern_data,
        get_glow_tts_preload_from_files,
    )

    vocab = "spm10k"
    task = get_librispeech_task_raw_v2(vocab=vocab, train_epoch_split=1, train_epoch_wise_filter=None)
    task = dataclasses.replace(task)
    base_train = task.train_dataset
    in_key = base_train.get_default_input()  # "data"
    tgt_key = base_train.get_default_target()  # "classes"
    base_extern = dict(base_train.get_extern_data())
    base_eval = base_train.get_eval_datasets()

    # paired ASR audio sub-dataset, with speed perturbation (applied here -- do NOT pass __train_audio_preprocess,
    # else aed_train_exp would try to set it on the DatasetConfigStatic below).
    asr_ds = copy.deepcopy(base_train.get_train_dataset())
    assert asr_ds["class"] == "OggZipDataset", asr_ds["class"]
    asr_ds["audio"] = dict(asr_ds["audio"])
    asr_ds["audio"]["pre_process"] = speed_pert_librosa_config

    # text-only sub-dataset: one LmDataset emits the raw utf8 bytes of each LM line; a PostprocessingDataset
    # derives BOTH the spm target and the GlowTTS phonemes from that text in its map_seq (single corpus read,
    # no second tokenizing LmDataset). The spm + phone_info opts travel via the config (tk.Paths resolved there).
    phon_extern = get_glow_tts_phoneme_extern_data()
    corpus_files = [get_librispeech_normalized_lm_data(), get_train_corpus_text()]
    spm_dim = base_extern[tgt_key]["sparse_dim"]  # spm vocab dim (same object as the audio target)
    phon_dim = phon_extern["sparse_dim"]  # GlowTTS phoneme vocab dim
    text_ds = {
        "class": "PostprocessingDataset",
        # real ordering stays on the inner LmDataset; "default" here (see notes_postprocessing_train_dataset).
        "seq_ordering": "default",
        "dataset": {
            "class": "LmDataset",
            "corpus_file": corpus_files,
            "orth_vocab": {"class": "Utf8ByteTargets"},  # data = raw utf8 bytes of the line
            "use_cache_manager": True,
            "seq_end_symbol": None,
            "unknown_symbol": None,
            "partition_epoch": text_train_epoch_split,
            "seq_ordering": "laplace:.1000",
        },
        "map_seq": functools.partial(_glowtts_text_map_seq, target_key=tgt_key, spm_dim=spm_dim, phon_dim=phon_dim),
        "map_outputs": {
            tgt_key: {"dims": [Dim(None, name="spm_seq")], "sparse_dim": spm_dim, "dtype": "int32"},
            PHONEMES_DATA_KEY: {"dims": [Dim(None, name="phon_seq")], "sparse_dim": phon_dim, "dtype": "int32"},
        },
    }

    # interleave audio + text-only; CombinedDataset zero-fills the missing stream per branch (empty length-0).
    combined = {
        "class": "CombinedDataset",
        "datasets": {"asr": asr_ds, "text": text_ds},
        "data_map": {
            ("asr", in_key): in_key,
            ("asr", tgt_key): tgt_key,
            ("text", tgt_key): tgt_key,
            ("text", PHONEMES_DATA_KEY): PHONEMES_DATA_KEY,
        },
        "seq_ordering": "interleave",
    }

    extern_data = {**base_extern, PHONEMES_DATA_KEY: phon_extern}
    eval_datasets = {
        k: _wrap_eval_with_empty_phonemes(v, base_extern=base_extern, phon_extern=phon_extern)
        for k, v in base_eval.items()
    }

    task.train_dataset = DatasetConfigStatic(
        main_name="LS ASR + Text(spm+phon)",
        train_dataset=combined,
        default_input=in_key,
        default_target=tgt_key,
        extern_data=extern_data,
        eval_datasets=eval_datasets,
        use_deep_copy=True,
    )

    model_config: Dict[str, Any] = {
        "behavior_version": 25,
        "__serialization_version": 2,
        "enc_build_dict": rf.build_dict(
            ConformerEncoder,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            num_layers=16,
            out_dim=1024,
            encoder_layer=rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
        ),
        "dec_build_dict": rf.build_dict(
            TransformerDecoder,
            num_layers=6,
            model_dim=1024,
            norm=rf.build_dict(rf.RMSNorm),
            ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
            layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
        ),
        "feature_batch_norm": True,
        "feature_extraction": rf.build_dict(DbMelFeatureExtractor),
    }

    exp = aed_train_exp(
        name,
        configs.config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        task=task,
        model_def=aed_glowtts_model_def,
        model_config=model_config,
        config_updates={
            **configs._get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": {"asr": 100_000 * configs._batch_size_factor, "text": 100_000},
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 2,  # alternate_batching doubles the effective step
            "torch_batching": functools.partial(alternate_batching, asr_key=in_key),  # audio-presence key (we keep "data")
            "train_step": aed_glowtts_train_step,  # custom step; no TrainDef needed
            "learning_rate_control_error_measure": "ce",  # set explicitly since there is no TrainDef
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": 75,  # text batches have no audio length cap
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "preload_from_files": get_glow_tts_preload_from_files(),
            "glow_tts_length_scale_range": glow_tts_length_scale_range,
            "glow_tts_noise_scale_range": glow_tts_noise_scale_range,
            "txt_only_loss_scale": txt_only_loss_scale,
            "separate_txt_only_losses": True,
            # text-only data pipeline: the PostprocessingDataset map_seq reads these to tokenize raw text into
            # spm targets + GlowTTS phonemes (nested tk.Paths are resolved as config values).
            "glow_tts_text_spm_opts": get_vocab_by_str(vocab).get_opts(),
            "glow_tts_phone_info": get_glow_tts_phone_info(train=True),
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset": {"num_workers": 4}},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    return exp


def aed_glowtts_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Standard aed.Model + frozen GlowTTS attached as model.tts (log-mel out_dim == encoder in_dim)."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    model = _aed.aed_model_def(epoch=epoch, in_dim=in_dim, target_dim=target_dim)
    noise_scale_range = config.typed_value("glow_tts_noise_scale_range", (0.3, 0.9))
    length_scale_range = config.typed_value("glow_tts_length_scale_range", (0.7, 1.1))
    model.tts = GlowTtsLogMel(
        phoneme_vocab_dim=Dim(get_glow_tts_phoneme_vocab_size(), name="glowtts_phonemes"),
        out_dim=model.in_dim,  # GlowTTS log-mel feeds straight into the encoder (same DbMel space)
        glow_tts_noise_scale_range=tuple(noise_scale_range),
        glow_tts_length_scale_range=tuple(length_scale_range),
    )
    # GlowTTS is frozen: imported params, never updated.
    for p in model.tts.parameters():
        p.trainable = False
    return model


aed_glowtts_model_def: ModelDef[Model]
aed_glowtts_model_def.behavior_version = 25
aed_glowtts_model_def.backend = "torch"
aed_glowtts_model_def.batch_size_factor = _aed.aed_model_def.batch_size_factor


def aed_glowtts_train_step(*, model: Model, extern_data, **_kwargs_unused):
    """Custom RETURNN train_step: dual-branch AED+CTC (paired audio + text-only via GlowTTS).

    Does its own extern_data extraction -- including the 3rd ``phonemes`` stream that the default train_v4 step
    does not forward -- and the loss computation, so no separate TrainDef is needed. Pass as config["train_step"].

    alternate_batching makes each batch pure audio or pure text-only; we branch on whether the audio stream is
    empty. Audio batch -> model.encode. Text batch -> model.tts(phonemes) (frozen) -> model.encode_from_features.
    Then the standard aux-CTC + AED-CE losses (a "txt_" prefix + txt_only_loss_scale for the text branch).
    """
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict

    config = get_global_config()  # noqa
    data = extern_data[config.typed_value("default_input")]
    data_spatial_dim = data.get_time_dim_tag()
    targets = extern_data[config.typed_value("target")]
    targets_spatial_dim = targets.get_time_dim_tag()
    phonemes = extern_data[PHONEMES_DATA_KEY]
    phonemes_spatial_dim = phonemes.get_time_dim_tag()

    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert isinstance(use_normalized_loss, str) and use_normalized_loss in ("none", "frames", "seqs")
    label_smoothing = config.float("label_smoothing", 0.1)
    aux_ctc_label_smoothing = config.float("aux_ctc_label_smoothing", 0.0)
    text_augment = config.typed_value("text_augment", None)
    separate_txt_only_losses = config.bool("separate_txt_only_losses", True)
    txt_only_loss_scale = config.float("txt_only_loss_scale", 1.0)

    ctc_loss = rf.ctc_loss
    if aux_ctc_label_smoothing:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)

    # alternate_batching => each batch is pure audio or pure text-only. Branch on whether audio is empty.
    audio_sizes_cpu = data_spatial_dim.get_size_tensor()  # [B] on CPU
    have_audio = bool(rf.reduce_max(audio_sizes_cpu, axis=audio_sizes_cpu.dims).raw_tensor.item() > 0)
    if have_audio:
        loss_prefix = ""
        global_loss_scale = 1.0
    else:
        loss_prefix = "txt_" if separate_txt_only_losses else ""
        global_loss_scale = txt_only_loss_scale

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(layer_idx - 1) for layer_idx in aux_loss_layers])
    if have_audio:
        enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    else:
        log_mel, log_mel_spatial_dim = model.tts(phonemes, spatial_dim=phonemes_spatial_dim)
        log_mel = rf.stop_gradient(log_mel)  # TTS is frozen
        enc_raw, enc_spatial_dim = model.encode_from_features(
            log_mel, in_spatial_dim=log_mel_spatial_dim, collected_outputs=collected_outputs
        )
        enc = model.decoder.transform_encoder(enc_raw, axis=enc_spatial_dim)

    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        linear = getattr(model, f"enc_aux_logits_{layer_idx}")
        aux_logits = linear(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        if aux_ctc_label_smoothing:
            aux_ctc_log_probs = rf.label_smoothed_log_prob_gradient(
                aux_ctc_log_probs, smoothing=aux_ctc_label_smoothing, axis=model.wb_target_dim
            )
        aux_loss = ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        if use_normalized_loss in ("none", "frames"):
            aux_loss.mark_as_loss(
                f"{loss_prefix}ctc_{layer_idx}",
                scale=aux_loss_scales[i] * global_loss_scale,
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(),
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            aux_loss.mark_as_loss(
                f"{loss_prefix}ctc_{layer_idx}",
                scale=0,
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(),
            )
            aux_loss.mark_as_loss(
                f"{loss_prefix}seq_ctc_{layer_idx}",
                scale=aux_loss_scales[i] * global_loss_scale,
                use_normalized_loss=True,
            )

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )
    if text_augment:
        input_labels, targets_w_eos, targets_w_eos_spatial_dim = rf.cond(
            rf.get_run_ctx().train_flag,
            lambda: text_augment(
                input_labels=input_labels,
                targets_w_eos=targets_w_eos,
                spatial_dim=targets_w_eos_spatial_dim,
                exclude_labels={model.bos_idx, model.eos_idx},
            ),
            lambda: (input_labels, targets_w_eos, targets_w_eos_spatial_dim),
        )

    collected_outputs = CollectOutputsDict(
        allowed_key_patterns=[str(layer_idx - 1) for layer_idx in dec_aux_loss_layers]
    )
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=collected_outputs,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        out = collected_outputs[str(layer_idx - 1)]
        dec_aux_logits[layer_idx] = linear(norm(out))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )
        if not model.out_eos_separated:
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:
            log_prob = _aed.log_probs_with_eos_separated(
                logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx
            )
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        if use_normalized_loss in ("none", "frames"):
            loss.mark_as_loss(
                f"{loss_prefix}ce{postfix}",
                scale=scale * global_loss_scale,
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            loss.mark_as_loss(f"{loss_prefix}ce{postfix}", scale=0)
            loss_ = rf.pad_packed(loss, dims=batch_dims + [targets_w_eos_spatial_dim], in_dim=pack_dim)
            seq_loss = rf.reduce_sum(loss_, axis=targets_w_eos_spatial_dim)
            seq_loss.mark_as_loss(
                f"{loss_prefix}seq_ce{postfix}", scale=scale * global_loss_scale, use_normalized_loss=True
            )

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"{loss_prefix}fer{postfix}", as_error=True)


_glowtts_text_tok_cache = None


def _glowtts_text_tokenizers():
    """Lazily build + cache (per worker process) the spm vocab + GlowTTS PhoneSeqGenerator from the config."""
    global _glowtts_text_tok_cache
    if _glowtts_text_tok_cache is None:
        from returnn.config import get_global_config
        from returnn.datasets.util.vocabulary import Vocabulary
        from returnn.datasets.lm import PhoneSeqGenerator

        config = get_global_config()
        spm = Vocabulary.create_vocab(**config.typed_value("glow_tts_text_spm_opts"))
        seq_gen = PhoneSeqGenerator(**config.typed_value("glow_tts_phone_info"))
        _glowtts_text_tok_cache = (spm, seq_gen)
    return _glowtts_text_tok_cache


def _glowtts_text_map_seq(seq, *, target_key, spm_dim, phon_dim, rng, **_kwargs):
    """PostprocessingDataset map_seq: raw utf8 bytes -> (spm target, GlowTTS phonemes), both from the same text."""
    import numpy as np
    from returnn.tensor import Tensor, TensorDict, Dim as _Dim

    spm, seq_gen = _glowtts_text_tokenizers()
    orth = bytes(np.asarray(seq["data"].raw_tensor).astype("uint8").tolist()).decode("utf8")
    spm_ids = np.array(spm.get_seq(orth), dtype="int32")
    # per-seq silence/pronunciation-variant randomization, seeded from the epoch-seeded rng.
    seq_gen.random_seed(int(rng.randint(0, 2**31 - 1)))
    phon_ids = seq_gen.seq_to_class_idxs(seq_gen.generate_seq(orth), dtype="int32")

    out = TensorDict()
    out.data[target_key] = Tensor(
        target_key, dims=[_Dim(None, name="spm_seq")], dtype="int32", sparse_dim=spm_dim, raw_tensor=spm_ids
    )
    out.data[PHONEMES_DATA_KEY] = Tensor(
        PHONEMES_DATA_KEY, dims=[_Dim(None, name="phon_seq")], dtype="int32", sparse_dim=phon_dim, raw_tensor=phon_ids
    )
    return out


def _add_empty_phonemes_map_seq(seq, *, phonemes_sparse_dim, **_kwargs):
    """PostprocessingDataset map_seq: pass everything through, add an empty ``phonemes`` stream.

    Used only for the audio-only CV/eval datasets so they carry the (unused-at-CV) phonemes key.
    """
    import numpy as np
    from returnn.tensor import Tensor, TensorDict, Dim as _Dim

    out = TensorDict()
    for k, v in seq.data.items():
        out.data[k] = v
    # Dynamic per-seq dim (length 0 here); PostprocessingDataset requires it dynamic to match map_outputs.
    spatial = _Dim(None, name="phon_seq")
    out.data[PHONEMES_DATA_KEY] = Tensor(
        PHONEMES_DATA_KEY,
        dims=[spatial],
        dtype="int32",
        sparse_dim=phonemes_sparse_dim,
        raw_tensor=np.zeros([0], "int32"),
    )
    return out


def _extern_template_to_map_output(tmpl: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an extern_data template ({dim_tags:[batch,...], [sparse_dim]}) into a PostprocessingDataset
    ``map_outputs`` entry ({dims:[non-batch dims], dtype, [sparse_dim]})."""
    from returnn.tensor import batch_dim

    dims = [d for d in tmpl["dim_tags"] if d != batch_dim]
    out: Dict[str, Any] = {"dims": dims, "dtype": "int32" if "sparse_dim" in tmpl else "float32"}
    if "sparse_dim" in tmpl:
        out["sparse_dim"] = tmpl["sparse_dim"]
    return out


def _wrap_eval_with_empty_phonemes(
    eval_ds: Dict[str, Any], *, base_extern: Dict[str, Any], phon_extern: Dict[str, Any]
) -> Dict[str, Any]:
    """Wrap an audio-only eval dataset so it also emits an empty ``phonemes`` stream (extern_data contract)."""
    map_outputs = {
        k: _extern_template_to_map_output(v) for k, v in {**base_extern, PHONEMES_DATA_KEY: phon_extern}.items()
    }
    return {
        "class": "PostprocessingDataset",
        # Explicit "default": PostprocessingDataset rejects a non-default seq_ordering on itself, and RETURNN
        # would otherwise inject one. The inner eval dataset keeps its own "sorted_reverse".
        "seq_ordering": "default",
        "dataset": eval_ds,
        "map_seq": functools.partial(_add_empty_phonemes_map_seq, phonemes_sparse_dim=phon_extern["sparse_dim"]),
        "map_outputs": map_outputs,
    }
