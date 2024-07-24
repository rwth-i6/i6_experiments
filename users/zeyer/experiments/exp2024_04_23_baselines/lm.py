"""
Language models

some ref:
https://github.com/rwth-i6/returnn-experiments/blob/master/2019-lm-transformers/librispeech/bpe_10k/transfo_24_d00.4096_1024.sgd.lr1.8_heads.config
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep


def py():
    """Sisyphus entry point"""
    from sisyphus import gs
    from i6_experiments.common.setups import serialization
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
    from .configs import config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4

    # TODO try train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
    # TODO label smoothing?
    # TODO try "optimizer.class": "RAdam", "optimizer.decoupled_weight_decay": True. but needs newer PyTorch?

    train(
        "lm/trafo",
        config=dict_update_deep(
            config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),  # TODO wrong...
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
                "max_seqs": 32,  # TODO really?
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def, {"_model_def_dict": rf.build_dict(TransformerDecoder, encoder_dim=None, num_layers=12)}
        ),
        train_def=lm_train_def,
    )


def lm_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> rf.Module:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    assert target_dim
    config = get_global_config()  # noqa

    model = rf.build_from_dict(config.typed_value("_model_def_dict"), vocab_dim=target_dim)
    return model


lm_model_def: ModelDef
lm_model_def.behavior_version = 21
lm_model_def.backend = "torch"
lm_model_def.batch_size_factor = 1


def lm_train_def(
    *, model: rf.Module, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    use_normalized_loss = config.bool("use_normalized_loss", True)

    # potentially also other types but just assume
    # noinspection PyTypeChecker
    model: TransformerDecoder
    vocab = model.vocab_dim.vocab
    assert vocab.bos_label_id is not None and vocab.eos_label_id is not None

    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=vocab.bos_label_id
    )
    targets_w_eos, _ = rf.pad(
        targets,
        axes=[targets_spatial_dim],
        padding=[(0, 1)],
        value=vocab.eos_label_id,
        out_dims=[targets_w_eos_spatial_dim],
    )

    batch_dims = data.remaining_dims(data_spatial_dim)
    logits, _ = model(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=None,
        state=model.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(
        logits, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    targets_packed, _ = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.vocab_dim)
    # log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.vocab_dim)
    loss.mark_as_loss("ce", use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.vocab_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


lm_train_def: TrainDef
lm_train_def.learning_rate_control_error_measure = "ce"
