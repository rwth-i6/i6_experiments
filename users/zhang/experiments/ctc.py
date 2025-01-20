"""
CTC experiments.
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Callable, Union, Tuple, Sequence

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep, _batch_size_factor

if TYPE_CHECKING:
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints
    from i6_experiments.users.zeyer.datasets.task import Task
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

OUT_BLANK_LABEL = "<blank>"
CHECK_DECODER_CONSISTENCY = False
# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

_raw_sample_rate = _batch_size_factor * 100  # bs factor is from 10ms frames to raw samples


def py():
    """Sisyphus entry point"""

    # relPosAttDef: Use the default RelPosSelfAttention instead of the Shawn et al 2018 style, old RETURNN way.
    enc_conformer_layer_default = rf.build_dict(
        rf.encoder.conformer.ConformerEncoderLayer,
        ff_activation=rf.build_dict(rf.relu_square),
        num_heads=8,
    )

    #-------------------setting decoding config-------------------
    decoding_config = {
        "log_add": False,
        "nbest": 1,
        "beam_size": 80,
        "lm_weight": 1.9,  # NOTE: weights are exponentials of the probs
        "use_logsoftmax": True,
        "use_lm": True,
        "use_lexicon": True,
    }

    p1 = "sum" if decoding_config['log_add'] else "max"
    p2 = f"n{decoding_config['nbest']}"
    p3 = f"b{decoding_config['beam_size']}"
    p4 = f"w{str(decoding_config['lm_weight']).replace('.', '')}"
    p5 = "_logsoftmax" if decoding_config['use_logsoftmax'] else ""
    p6 = "_lexicon" if decoding_config['use_lexicon'] else ""
    lm_hyperparamters_str = f"{p1}_{p2}_{p3}_{p4}{p5}{p6}"


    # ---------------------------------------------------------
    # model name: f"v6-relPosAttDef-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100"
    #               f"-maxSeqLenAudio19_5-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2"
    #               f"-[vocab]" + f"-[sample]"
    from .language_models.librispeech_lm import get_4gram_binary_lm
    lms = [get_4gram_binary_lm()]
    # TODO: get ppl of each lm use i6_core.lm.srilm.ComputeNgramLmPerplexityJob

    for vocab, sample, alpha in [
        # ("spm20k", None, None),  # 5.96
        # ("spm20k", "spm", 0.7),  # 6.14
        # # TODO ("spm20k", "bpe", 0.005),
        # ("spm20k", "bpe", 0.01),  # 6.13
        # ("spm20k", "bpe", 0.02),  # 6.21
        # ("bpe10k", None, None),  # 6.49  #TODO: does not work currently, check the BPEjob in librispeech.bpe10k
        # ("bpe10k", "bpe", 0.005),  # 6.48
        # ("bpe10k", "bpe", 0.01),  # 6.40
        # ("spm10k", None, None),  # 6.00
        # # TODO ("spm10k", "spm", 0.8),
        # ("spm10k", "spm", 0.7),  # 6.20
        # ("spm10k", "bpe", 0.001),  # 5.93
        # ("spm10k", "bpe", 0.005),  # 5.89 (!!)
        # ("spm10k", "bpe", 0.01),  # 5.93
        # ("spm_bpe10k", None, None),  # 6.33
        # ("spm_bpe10k", "spm", 1e-4),  # 6.26
        # # TODO ("spm_bpe10k", "bpe", 0.005),
        # ("spm_bpe10k", "bpe", 0.01),  # 6.21
        # ("spm4k", None, None),  # 6.07 (but test-other even better: 5.94?)
        # ("spm4k", "spm", 0.7),  # 6.42
        # # TODO ("spm4k", "bpe", 0.005),
        # ("spm4k", "bpe", 0.01),  # 6.05
        # ("spm1k", None, None),  # 6.07
        # ("spm1k", "spm", 1.0),  # 6.73
        # ("spm1k", "spm", 0.99),  # 6.93
        # ("spm1k", "spm", 0.9),  # 7.04
        # ("spm1k", "spm", 0.7),  # 7.33
        # ("spm1k", "bpe", 0.0),  # 6.07
        # # TODO ("spm1k", "bpe", 0.0005),
        # ("spm1k", "bpe", 0.001),  # 6.15
        # ("spm1k", "bpe", 0.005),  # 6.25
        # ("spm1k", "bpe", 0.01),  # 6.13 (but dev-clean,test-* are better than no sampling)
        # ("spm_bpe1k", None, None),  # 6.03
        # ("spm_bpe1k", "bpe", 0.01),  # 6.05
        # ("spm512", None, None),  # 6.08
        # ("spm512", "bpe", 0.001),  # 6.05
        # ("spm512", "bpe", 0.005),  # 6.01
        # ("spm512", "bpe", 0.01),  # 6.08 (but test-* is better than spm512 without sampling)
        # ("spm128", None, None),  # 6.37
        # # TODO ("spm128", "bpe", 0.001),
        # ("spm128", "bpe", 0.01),  # 6.40
        # # TODO ("spm128", "bpe", 0.005),
         ("bpe128", None, None),
        # ("spm64", None, None),
        # ("bpe64", None, None),
        # ("utf8", None, None),
        # ("char", None, None),
        # ("bpe0", None, None),
    ]:
        for lm in lms:
            # lm_name = lm if isinstance(lm, str) else lm.name
            alias_name = f"ctc-baseline" + "decodingWith_" + lm_hyperparamters_str + "4gramLm"
            decoding_config["lm"] = lm
            train_exp(
                name=alias_name,
                config=config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
                decoder_def=model_recog_lm,
                model_config={"enc_conformer_layer": enc_conformer_layer_default, "feature_batch_norm": True},
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    "max_seq_length_default_target": None,
                    # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                    # out of 281241 seqs in train, we removed only 71 seqs.
                    # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                    "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                },
                vocab=vocab,
                train_vocab_opts=(
                    {
                        "other_opts": (
                            {
                                "spm": {"enable_sampling": True, "alpha": 0.005},
                                "bpe": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.005},
                            }[sample]
                        )
                    }
                    if sample
                    else None
                ),
                decoding_config=decoding_config
            )


_train_experiments: Dict[str, ModelWithCheckpoints] = {}


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    decoder_def: Callable,
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    vocab: str = "bpe10k",
    train_vocab_opts: Optional[Dict[str, Any]] = None,
    dataset_train_opts: Optional[Dict[str, Any]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None, # Returnn config
    config_updates: Optional[Dict[str, Any]] = None, # For iterative changing of config
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    epilog: Sequence[serialization.SerializerObject] = (),
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    env_updates: Optional[Dict[str, str]] = None,
    enabled: bool = True,
    decoding_config: dict = None,
) -> Optional[ModelWithCheckpoints]:
    """
    Train experiment
    """
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zhang.recog import recog_training_exp # TODO： determine if this module need modification
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2

    if not enabled:
        return None

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    # TODO： find out how to apply recognition on other dataset
    task = get_librispeech_task_raw_v2(vocab=vocab, train_vocab_opts=train_vocab_opts, **(dataset_train_opts or {}))
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    # This logic is also in train(), but keep it here because it would break the hash because of _RecogAndScoreFunc...
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = ctc_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = ctc_training
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        epilog=epilog,
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        time_rqmt=time_rqmt,
        env_updates=env_updates,
    )

    recog_post_proc_funcs = []
    if config.get("use_eos_postfix", False):
        recog_post_proc_funcs.append(_remove_eos_label_v2)
    recog_training_exp(
        prefix, task, model_with_checkpoint, recog_def=decoder_def,
        decoding_config=decoding_config,
        recog_post_proc_funcs=recog_post_proc_funcs,
    )

    _train_experiments[name] = model_with_checkpoint
    return model_with_checkpoint


def _remove_eos_label_v2(res: RecogOutput) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.score_results import RecogOutput
    from i6_core.returnn.search import SearchRemoveLabelJob

    return RecogOutput(SearchRemoveLabelJob(res.output, remove_label="</s>", output_gzip=True).out_search_results)


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name


def ctc_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    in_dim, epoch  # noqa
    return Model(**_get_ctc_model_kwargs_from_global_config(target_dim=target_dim))


ctc_model_def: ModelDef[Model]
ctc_model_def.behavior_version = 21
ctc_model_def.backend = "torch"
ctc_model_def.batch_size_factor = _batch_size_factor


def _get_ctc_model_kwargs_from_global_config(*, target_dim: Dim) -> Dict[str, Any]:
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    enc_input_layer = config.typed_value("enc_input_layer", None)
    conv_norm = config.typed_value("conv_norm", None)
    enc_conformer_layer = config.typed_value("enc_conformer_layer", None)
    if enc_conformer_layer:
        assert not conv_norm, "set only enc_conformer_layer or conv_norm, not both"
        assert isinstance(enc_conformer_layer, dict) and "class" in enc_conformer_layer
    else:
        enc_conformer_layer = rf.build_dict(
            rf.encoder.conformer.ConformerEncoderLayer,
            conv_norm=conv_norm or {"class": "rf.BatchNorm", "use_mask": True},
            self_att=rf.build_dict(
                rf.RelPosSelfAttention,
                # Shawn et al 2018 style, old RETURNN way.
                with_bias=False,
                with_linear_pos=False,
                with_pos_bias=False,
                learnable_pos_emb=True,
                separate_pos_emb_per_head=False,
            ),
            ff_activation=rf.build_dict(rf.relu_square),
            num_heads=8,
        )
    enc_other_opts = config.typed_value("enc_other_opts", None)

    return dict(
        in_dim=in_dim,
        enc_build_dict=config.typed_value("enc_build_dict", None),  # alternative more generic/flexible way
        num_enc_layers=num_enc_layers,
        enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
        enc_input_layer=enc_input_layer,
        enc_conformer_layer=enc_conformer_layer,
        enc_other_opts=enc_other_opts,
        target_dim=target_dim,
        blank_idx=target_dim.dimension,
        bos_idx=_get_bos_idx(target_dim),
        eos_idx=_get_eos_idx(target_dim),
        enc_aux_logits=enc_aux_logits or (),
    )


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def ctc_training(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    use_fixed_ctc_grad = config.typed_value("use_fixed_ctc_grad", False)

    ctc_loss = rf.ctc_loss
    if use_fixed_ctc_grad:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        assert use_fixed_ctc_grad == "v2"  # v2 has the fix for scaled/normalized CTC loss
        ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    collected_outputs = {} if aux_loss_layers else None
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    log_probs = model.log_probs_wb_from_logits(logits)
    loss = ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

    if model.decoder:
        # potentially also other types but just assume
        # noinspection PyTypeChecker
        decoder: TransformerDecoder = model.decoder

        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = data.remaining_dims(data_spatial_dim)
        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_w_eos_spatial_dim,
            encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )

        logits_packed, pack_dim = rf.pack_padded(
            logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
        )
        targets_packed, _ = rf.pack_padded(
            targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )

        log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


ctc_training: TrainDef[Model]
ctc_training.learning_rate_control_error_measure = "ctc"


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)  # Batch, Beam

    label_log_prob = model.log_probs_wb_from_logits(logits)  # Batch, Spatial, Vocab
    label_log_prob = rf.where(
        enc_spatial_dim.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    # Just Initialisation
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob, k_dim=Dim(beam_size, name=f"pre-filter-beam"), axis=[model.wb_target_dim]
    )  # seq_log_prob, backrefs_global: Batch, Spatial, PreFilterBeam. backrefs_pre_filter -> Vocab
    label_log_prob_pre_filter_ta = TensorArray.unstack(
        label_log_prob_pre_filter, axis=enc_spatial_dim
    )  # t -> Batch, PreFilterBeam
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim)  # t -> Batch, PreFilterBeam

    max_seq_len = int(enc_spatial_dim.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        # Filter out finished beams
        # Since it is log_prob, + means the prob is actually product,
        # Note for CTC each time step is independent to each other.
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]  # Batch, InBeam, PreFilterBeam
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{t}-beam"), axis=[beam_dim, pre_filter_beam_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam. backrefs -> InBeam. target -> PreFilterBeam.
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)  # Batch, Beam -> Vocab
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # FinalBeam -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: FinalBeam -> Beam
        # backrefs: Beam -> PrevBeam
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # FinalBeam -> PrevBeam

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<blank>"
model_recog.batch_size_dependent = False  # not totally correct, but we treat it as such...


def model_recog_lm(
        *,
        model: Model,
        data: Tensor,
        data_spatial_dim: Dim,
        lm: str,
        lexicon: str,
        hyperparameters: dict
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Uses a LM and beam search.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from torchaudio.models.decoder import ctc_decoder
    import torch
    from returnn.util.basic import cf
    assert lm, "No lm loaded!"
    # Get the logits from the model
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim)

    hyp_params = copy.copy(hyperparameters)
    use_logsoftmax = hyp_params.pop("use_logsoftmax", False)

    if use_logsoftmax:
        label_log_prob = model.log_probs_wb_from_logits(logits)
        label_log_prob = label_log_prob.raw_tensor.cpu()

    # if greedy:
    #     probs, greedy_res = torch.max(label_log_prob, dim=-1)
    #     greedy_res = greedy_res.unsqueeze(1)
    #
    #     scores = torch.sum(probs, dim=-1)
    #     scores = scores.unsqueeze(1)
    #
    #     beam_dim = rtf.TorchBackend.get_new_dim_raw(greedy_res, 1, name="beam_dim")
    #     dims = [batch_dim, beam_dim, enc_spatial_dim]
    #     hyps = rtf.TorchBackend.convert_to_tensor(greedy_res, dims=dims, sparse_dim=model.wb_target_dim, dtype="int64",
    #                                               name="hyps")
    #
    #     dims = [batch_dim, beam_dim]
    #     scores = Tensor("scores", dims=dims, dtype="float32", raw_tensor=scores)
    #
    #     return hyps, scores, enc_spatial_dim, beam_dim

    lm = str(cf(lm))

    use_lexicon = hyp_params.pop("use_lexicon", True)

    configs = {
        "tokens": list(model.wb_target_dim.vocab.labels),
        "blank_token": OUT_BLANK_LABEL,
        "sil_token": OUT_BLANK_LABEL,
        "unk_word": "<unk>",
        "beam_size_token": None,  # 16
        "beam_threshold": 1000000,  # 14
    }
    configs["lexicon"] = lexicon if use_lexicon else None
    configs["lm"] = lm

    configs.update(hyp_params)

    decoder = ctc_decoder(**configs)
    enc_spatial_dim_torch = enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()
    if use_logsoftmax:
        decoder_results = decoder(label_log_prob, enc_spatial_dim_torch)
    else:
        decoder_results = decoder(logits.raw_tensor.cpu(), enc_spatial_dim_torch)

    if use_lexicon:
        print("Use words directly!")
        if CHECK_DECODER_CONSISTENCY:
            for l1 in decoder_results:
                for l2 in l1:
                    lexicon_words = " ".join(l2.words)
                    token_words = " ".join([configs["tokens"][t] for t in l2.tokens])
                    assert not token_words.endswith(
                        "@@"), f"Token words ends with @@: {token_words}, Lexicon words: {lexicon_words}"
                    token_words = token_words.replace("@@ ", "")
                    assert lexicon_words == token_words, f"Words don't match: Lexicon words: {lexicon_words}, Token words: {token_words}"

        words = [[" ".join(l2.words) for l2 in l1] for l1 in decoder_results]
        words = np.array(words)
        words = np.expand_dims(words, axis=2)
        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        scores = torch.tensor(scores)

        beam_dim = Dim(words.shape[1], name="beam_dim")
        enc_spatial_dim = Dim(1, name="spatial_dim")
        words = rf._numpy_backend.NumpyBackend.convert_to_tensor(words, dims=[batch_dim, beam_dim, enc_spatial_dim],
                                                                 dtype="string", name="hyps")
        scores = Tensor("scores", dims=[batch_dim, beam_dim], dtype="float32", raw_tensor=scores)

        return words, scores, enc_spatial_dim, beam_dim
    else:
        def _pad_blanks(tokens, max_len):
            if len(tokens) < max_len:
                # print("We had to pad blanks")
                tokens = torch.cat([tokens, torch.tensor([model.blank_idx] * (max_len - len(tokens)))])
            return tokens

        def _pad_lists(t, max_len, max_len2):
            if t.shape[0] < max_len2:
                print("We had to pad the list")
                t = torch.cat([t, torch.tensor([[model.blank_idx] * max_len] * (max_len2 - t.shape[0]))])
            return t

        def _pad_scores(l, max_len):
            l = torch.tensor(l)
            if len(l) < max_len:
                print("We had to pad scores")
                l = torch.cat([l, torch.tensor([-1000000.0] * (max_len - len(l)))])
            return l

        max_length = int(enc_spatial_dim_torch.max())
        hyps = [torch.stack([_pad_blanks(l2.tokens, max_length) for l2 in l1]) for l1 in decoder_results]
        max_length_2 = max([l.shape[0] for l in hyps])
        hyps = [_pad_lists(t, max_length, max_length_2) for t in hyps]
        hyps = torch.stack(hyps)
        beam_dim = rtf.TorchBackend.get_new_dim_raw(hyps, 1, name="beam_dim")
        dims = [batch_dim, beam_dim, enc_spatial_dim]
        hyps = rtf.TorchBackend.convert_to_tensor(hyps, dims=dims, sparse_dim=model.wb_target_dim, dtype="int64",
                                                  name="hyps")

        scores = [[l2.score for l2 in l1] for l1 in decoder_results]
        max_length_3 = max([len(l) for l in scores])
        scores = torch.stack([_pad_scores(l, max_length_3) for l in scores])
        dims = [batch_dim, beam_dim]
        scores = Tensor("scores", dims=dims, dtype="float32", raw_tensor=scores)

        # print(f"CUSTOM seq_targets: {hyps} \n{hyps.raw_tensor.cpu()},\nscores: {scores} \n{scores.raw_tensor.cpu()}n {scores.raw_tensor.cpu()[0][0]},\nspatial_dim: {enc_spatial_dim.dyn_size_ext.raw_tensor.cpu()},\n beam_size: {beam_dim}")

        return hyps, scores, enc_spatial_dim, beam_dim


# RecogDef API
model_recog_lm: RecogDef[Model]
model_recog_lm.output_with_beam = True
model_recog_lm.output_blank_label = OUT_BLANK_LABEL
model_recog_lm.batch_size_dependent = False  # not totally correct, but we treat it as such...


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_build_dict: Optional[Dict[str, Any]] = None,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        enc_input_layer: Optional[Dict[str, Any]] = None,
        enc_conformer_layer: Optional[Dict[str, Any]] = None,
        enc_other_opts: Optional[Dict[str, Any]] = None,
    ):
        super(Model, self).__init__()

        self.in_dim = in_dim

        import numpy
        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        if enc_build_dict:
            # Warning: We ignore the other args (num_enc_layers, enc_model_dim, enc_other_opts, etc).
            self.encoder = rf.build_from_dict(enc_build_dict, in_dim)
            self.encoder: ConformerEncoder  # might not be true, but assume similar/same interface

        else:
            if not enc_input_layer:
                enc_input_layer = ConformerConvSubsample(
                    in_dim,
                    out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],
                )

            enc_opts = {"input_layer": enc_input_layer, "num_layers": num_enc_layers}

            if enc_conformer_layer:
                enc_opts["encoder_layer"] = enc_conformer_layer

            enc_layer_drop = config.float("enc_layer_drop", 0.0)
            if enc_layer_drop:
                assert "sequential" not in enc_opts
                enc_opts["sequential"] = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)

            if enc_other_opts:
                for k, v in enc_other_opts.items():
                    assert k not in enc_opts, f"enc_other_opts key {k!r} already in enc_opts {enc_opts}"
                    enc_opts[k] = v

            self.encoder = ConformerEncoder(in_dim, enc_model_dim, **enc_opts)

        # Experiments without final layer norm. (We might clean this up when this is not successful.)
        # Just patch the encoder here.
        enc_conformer_final_layer_norm = config.typed_value("enc_conformer_final_layer_norm", None)
        if enc_conformer_final_layer_norm is None:
            pass
        elif enc_conformer_final_layer_norm == "last":  # only in the last, i.e. remove everywhere else
            for layer in self.encoder.layers[:-1]:
                layer: ConformerEncoderLayer
                layer.final_layer_norm = rf.identity
        else:
            raise ValueError(f"invalid enc_conformer_final_layer_norm {enc_conformer_final_layer_norm!r}")

        disable_encoder_self_attention = config.typed_value("disable_encoder_self_attention", None)
        if disable_encoder_self_attention is not None:
            # Disable self-attention in encoder.
            from .model_ext.disable_self_att import apply_disable_self_attention_

            apply_disable_self_attention_(self.encoder, disable_encoder_self_attention)

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        if not wb_target_dim:
            wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))
        self.enc_logits = rf.Linear(self.encoder.out_dim, wb_target_dim)
        self.wb_target_dim = wb_target_dim
        self.out_blank_separated = config.bool("out_blank_separated", False)
        self.blank_logit_shift = config.float("blank_logit_shift", 0.0)

        self.ctc_am_scale = config.float("ctc_am_scale", 1.0)
        self.ctc_prior_scale = config.float("ctc_prior_scale", 0.0)
        self.ctc_prior_type = config.value("ctc_prior_type", "batch")

        static_prior = config.typed_value("static_prior")
        self.static_prior = None
        if static_prior:
            assert isinstance(static_prior, dict)
            assert set(static_prior.keys()) == {"file", "type"}
            v = numpy.loadtxt(static_prior["file"])
            if static_prior["type"] == "log_prob":
                pass  # already log prob
            elif static_prior["type"] == "prob":
                v = numpy.log(v)
            else:
                raise ValueError(f"invalid static_prior type {static_prior['type']!r}")
            self.static_prior = rf.Parameter(
                rf.convert_to_tensor(v, dims=[self.wb_target_dim], dtype=rf.get_default_float_dtype()),
                auxiliary=True,
                non_critical_for_restore=True,
            )

        if target_dim.vocab and not wb_target_dim.vocab:
            from returnn.datasets.util.vocabulary import Vocabulary

            # Just assumption for code now, might extend this later.
            assert wb_target_dim.dimension == target_dim.dimension + 1 and blank_idx == target_dim.dimension
            vocab_labels = list(target_dim.vocab.labels) + [model_recog.output_blank_label]
            wb_target_dim.vocab = Vocabulary.create_vocab_from_labels(
                vocab_labels, user_defined_symbols={model_recog.output_blank_label: blank_idx}
            )

        ctc_label_smoothing = config.float("ctc_label_smoothing", 0.0)
        ctc_label_smoothing_exclude_blank = config.bool("ctc_label_smoothing_exclude_blank", self.out_blank_separated)
        self.ctc_label_smoothing_exclude_blank = ctc_label_smoothing_exclude_blank
        if not self.out_blank_separated:
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.wb_target_dim,
                "exclude_labels": [self.blank_idx] if ctc_label_smoothing_exclude_blank else None,
            }
        else:  # separate blank
            self.ctc_label_smoothing_opts = {
                "smoothing": ctc_label_smoothing,
                "axis": self.target_dim if ctc_label_smoothing_exclude_blank else self.wb_target_dim,
            }
        self.log_prob_normed_grad_opts = config.typed_value("log_prob_normed_grad", None)
        self.log_prob_normed_grad_exclude_blank = config.bool(
            "log_prob_normed_grad_exclude_blank", self.out_blank_separated
        )

        self.feature_batch_norm = None
        if config.bool("feature_batch_norm", False):
            self.feature_batch_norm = rf.BatchNorm(self.in_dim, affine=False, use_mask=True)
        self.feature_norm = config.bool("feature_norm", False)
        self.feature_stats = None
        feature_stats = config.typed_value("feature_stats")
        if feature_stats:
            assert isinstance(feature_stats, dict)
            self.feature_stats = rf.ParameterList(
                {
                    k: rf.Parameter(
                        rf.convert_to_tensor(numpy.loadtxt(v), dims=[self.in_dim], dtype=rf.get_default_float_dtype()),
                        auxiliary=True,
                        non_critical_for_restore=True,
                    )
                    for k, v in feature_stats.items()
                }
            )

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

        self.decoder = None
        aux_attention_decoder = config.typed_value("aux_attention_decoder", None)
        if aux_attention_decoder:
            assert isinstance(aux_attention_decoder, dict)
            aux_attention_decoder = aux_attention_decoder.copy()
            aux_attention_decoder.setdefault("class", "returnn.frontend.decoder.transformer.TransformerDecoder")
            if isinstance(aux_attention_decoder.get("model_dim", None), int):
                aux_attention_decoder["model_dim"] = Dim(aux_attention_decoder["model_dim"], name="dec_model")
            self.decoder = rf.build_from_dict(
                aux_attention_decoder, encoder_dim=self.encoder.out_dim, vocab_dim=target_dim
            )

        vn = config.typed_value("variational_noise", None)
        if vn:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_noise(mod, param_name, std=vn)

        weight_dropout = config.typed_value("weight_dropout", None)
        if weight_dropout:
            # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
            # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
            blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
            blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
            for mod in self.modules():
                if isinstance(mod, blacklist):
                    continue
                for param_name, param in mod.named_parameters(recurse=False):
                    if param_name.endswith("bias"):  # no bias
                        continue
                    if param.auxiliary:
                        continue
                    rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)

    def __call__(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Dim]:
        """
        Encode, get CTC logits.
        Use :func:`log_probs_wb_from_logits` to get log probs
        (might be just log_softmax, but there are some other cases).

        :return: logits, enc, enc_spatial_dim
        """
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
        )
        if self.feature_batch_norm:
            source = self.feature_batch_norm(source)
        if self.feature_norm:
            source = rf.normalize(source, axis=in_spatial_dim)
        if self.feature_stats:
            source = (source - self.feature_stats.mean) / self.feature_stats.std_dev
        if self._mixup:
            source = self._mixup(source, spatial_dim=in_spatial_dim)
        # SpecAugment
        source = rf.audio.specaugment(
            source,
            spatial_dim=in_spatial_dim,
            feature_dim=self.in_dim,
            **self._specaugment_opts,
        )
        # Encoder including convolutional frontend
        enc, enc_spatial_dim = self.encoder(source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs)
        logits = self.enc_logits(enc)
        return logits, enc, enc_spatial_dim

    def log_probs_wb_from_logits(self, logits: Tensor) -> Tensor:
        """
        :param logits: incl blank
        :return: log probs with blank from logits (wb_target_dim)
            If out_blank_separated, we use a separate sigmoid for the blank.
            Also, potentially adds label smoothing on the gradients.
        """
        if not self.out_blank_separated:  # standard case, joint distrib incl blank
            if self.blank_logit_shift:
                logits += rf.sparse_to_dense(
                    self.blank_idx, label_value=self.blank_logit_shift, other_value=0, axis=self.wb_target_dim
                )
            log_probs = rf.log_softmax(logits, axis=self.wb_target_dim)
        else:  # separate blank
            assert self.blank_idx == self.target_dim.dimension  # not implemented otherwise
            dummy_blank_feat_dim = Dim(1, name="blank_feat")
            logits_wo_blank, logits_blank = rf.split(
                logits, axis=self.wb_target_dim, out_dims=[self.target_dim, dummy_blank_feat_dim]
            )
            log_probs_wo_blank = rf.log_softmax(logits_wo_blank, axis=self.target_dim)
            log_probs_wo_blank = self._maybe_apply_on_log_probs(log_probs_wo_blank)
            if self.blank_logit_shift:
                logits_blank += self.blank_logit_shift
            log_probs_blank = rf.log_sigmoid(logits_blank)
            log_probs_emit = rf.squeeze(rf.log_sigmoid(-logits_blank), axis=dummy_blank_feat_dim)
            log_probs, _ = rf.concat(
                (log_probs_wo_blank + log_probs_emit, self.target_dim),
                (log_probs_blank, dummy_blank_feat_dim),
                out_dim=self.wb_target_dim,
            )
            log_probs.feature_dim = self.wb_target_dim
        log_probs = self._maybe_apply_on_log_probs(log_probs)
        if self.ctc_am_scale == 1 and self.ctc_prior_scale == 0:  # fast path
            return log_probs
        log_probs_am = log_probs
        log_probs = log_probs_am * self.ctc_am_scale
        if self.ctc_prior_scale:
            if self.ctc_prior_type == "batch":
                log_prob_prior = rf.reduce_logsumexp(
                    log_probs_am, axis=[dim for dim in log_probs_am.dims if dim != self.wb_target_dim]
                )
                assert log_prob_prior.dims == (self.wb_target_dim,)
            elif self.ctc_prior_type == "static":
                log_prob_prior = self.static_prior
                assert log_prob_prior.dims == (self.wb_target_dim,)
            else:
                raise ValueError(f"invalid ctc_prior_type {self.ctc_prior_type!r}")
            log_probs -= log_prob_prior * self.ctc_prior_scale
        return log_probs

    def _maybe_apply_on_log_probs(self, log_probs: Tensor) -> Tensor:
        """
        :param log_probs: either with blank or without blank
        :return: log probs, maybe some smoothing applied (all on gradients so far, not on log probs itself)
        """
        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim

        log_probs = self._maybe_apply_log_probs_normed_grad(log_probs)

        if self.ctc_label_smoothing_exclude_blank:
            if self.out_blank_separated:
                if log_probs.feature_dim == self.target_dim:
                    log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
            else:
                assert log_probs.feature_dim == self.wb_target_dim
                assert self.ctc_label_smoothing_opts["exclude_labels"] == [self.blank_idx]
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)
        else:
            if log_probs.feature_dim == self.wb_target_dim:
                log_probs = rf.label_smoothed_log_prob_gradient(log_probs, **self.ctc_label_smoothing_opts)

        return log_probs

    def _maybe_apply_log_probs_normed_grad(self, log_probs: Tensor) -> Tensor:
        if not self.log_prob_normed_grad_opts:
            return log_probs

        assert log_probs.feature_dim in (self.wb_target_dim, self.target_dim)
        if not self.out_blank_separated:
            assert log_probs.feature_dim == self.wb_target_dim
        if self.log_prob_normed_grad_exclude_blank:
            assert self.out_blank_separated
            if log_probs.feature_dim == self.wb_target_dim:
                return log_probs
        else:  # not excluded blank
            if log_probs.feature_dim == self.target_dim:
                return log_probs

        from alignments.util import normed_gradient, NormedGradientFuncInvPrior

        opts: Dict[str, Any] = self.log_prob_normed_grad_opts.copy()
        func_opts = opts.pop("func")
        assert isinstance(func_opts, dict)
        func_opts = func_opts.copy()
        assert func_opts.get("class", "inv_prior") == "inv_prior"  # only case for now
        func_opts.pop("class", None)
        func = NormedGradientFuncInvPrior(**func_opts)

        assert log_probs.batch_dim_axis is not None and log_probs.feature_dim_axis is not None
        log_probs_ = log_probs.copy_template()
        log_probs_.raw_tensor = normed_gradient(
            log_probs.raw_tensor,
            batch_axis=log_probs.batch_dim_axis,
            feat_axis=log_probs.feature_dim_axis,
            **opts,
            func=func,
        )
        return log_probs_
