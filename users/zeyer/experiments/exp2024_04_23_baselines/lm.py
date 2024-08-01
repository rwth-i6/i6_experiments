"""
Language models

some ref:
https://github.com/rwth-i6/returnn-experiments/blob/master/2019-lm-transformers/librispeech/bpe_10k/transfo_24_d00.4096_1024.sgd.lr1.8_heads.config
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Any, Sequence, Tuple, Dict

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop

from .configs import config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4


def py():
    """Sisyphus entry point"""
    from i6_experiments.users.zeyer.train_v3 import train
    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset

    # TODO LmDataset not as gzip, to allow for direct mmap?

    # TODO try train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
    # TODO label smoothing?
    # TODO try "optimizer.class": "RAdam", "optimizer.decoupled_weight_decay": True. but needs newer PyTorch?
    # TODO I have max_seq_length 75. For spm10k, that are 98.16% of the data. Maybe try without?

    # TODO check users/zeyer/experiments/exp2023_04_25_rf/README.md
    # TODO check users/zeyer/experiments/exp2023_04_25_rf/Transformer.md

    train(
        "lm/trafo-n12-d512-drop0-b200_10k-wrongLr",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(200, 10_000, 100),  # TODO...
                "learning_rate_piecewise_steps": [561_600, 1_123_200, 1_248_000],  # wrongLr
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder, encoder_dim=None, num_layers=12, model_dim=512, dropout=0.0, att_dropout=0.0
                )
            },
        ),
        train_def=lm_train_def,
    )

    train(
        "lm/trafo-n12-d512-drop0-b200_13k-wrongLr",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(200, 13_000, 100),  # TODO...
                "learning_rate_piecewise_steps": [561_600, 1_123_200, 1_248_000],  # wrongLr
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder, encoder_dim=None, num_layers=12, model_dim=512, dropout=0.0, att_dropout=0.0
                )
            },
        ),
        train_def=lm_train_def,
    )

    train(
        "lm/trafo-n24-d1024-drop0-b32_2k-wrongLr",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(32, 2_000, 100),
                "learning_rate_piecewise_steps": [2_808_000, 5_616_000, 6_240_000],  # wrongLr (but not too wrong)
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder, encoder_dim=None, num_layers=24, model_dim=1024, dropout=0.0, att_dropout=0.0
                )
            },
        ),
        train_def=lm_train_def,
    )

    train(
        "lm/trafo-n24-d512-gelu-drop0-b100_6k-wrongLr",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(100, 6_000, 100),  # TODO...
                "learning_rate_piecewise_steps": [561_600 // 2, 1_123_200 // 2, 1_248_000 // 2],  # wrongLr
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=24,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    train(
        "lm/trafo-n48-d512-gelu-drop0-b32_2k",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {**_get_cfg_lrlin_oclr_by_bs_nep(32, 2_000, 100)},
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=48,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
        ),
        train_def=lm_train_def,
    )

    train(
        "lm/trafo-n96-d512-gelu-drop0-b32_1k-wrongLr",
        config=dict_update_deep(
            config_11gb_lm_v1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep(32, 1_000, 100),  # TODO?
                "learning_rate_piecewise_steps": [561_600 // 5, 1_123_200 // 5, 1_248_000 // 5],  # wrongLr
            },
        ),
        train_dataset=get_librispeech_lm_dataset(vocab="spm10k"),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=96,
                    model_dim=512,
                    ff_activation=rf.build_dict(rf.gelu),
                    dropout=0.0,
                    att_dropout=0.0,
                )
            },
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


# By batch size and num (sub)epochs.
# See also in .configs.
# If the dict is missing some entry,
# unfortunately there is currently no good automatic way to get the number.
# I need to log at the stats of some setup with this batch size.
# I just run some setup with some arbitrary LR scheduling (calling it "wrongLr"),
# or maybe with sqrt-decay, and then look at the stats (steps/ep, or total num steps),
# and give some estimates for the steps here, i.e. 45%, 90%, almost 100%,
# making sure the last number is slightly below the real total number of steps.
_lrlin_oclr_steps_by_bs_nep = {
    (32, 1_000, 100): ...,  # TODO
    (32, 2_000, 100): [2_832_000, 5_665_000, 6_295_000],  # ~62951steps/ep, 100 eps -> 6,295k steps in total
    (32, 10_000, 20): [561_600, 1_123_200, 1_248_000],  # ~62421steps/ep, 20 eps -> 1,248k steps in total
    (100, 6_000, 100): ...,  # TODO
    (200, 10_000, 100): ...,  # TODO
    (200, 13_000, 100): ...,  # TODO
}
# Just specify avg num steps per (sub)epoch for batch size settings: (max_seqs, batch_size).
# TODO instead of the estimates, just exactly calculate this.
_tot_num_steps_by_bs = {
    (200, 10_000): 14040,
    (200, 13_000): 10955,
}


def _get_cfg_lrlin_oclr_by_bs_nep(max_seqs: int, bs_feat: int, n_ep: int, *, peak_lr: float = 1e-3) -> Dict[str, Any]:
    """
    :param max_seqs:
    :param bs_feat: batch size for features (not including _batch_size_factor)
    :param n_ep: num epochs
    """
    from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

    return {
        "__num_epochs": n_ep,
        "batch_size": bs_feat,
        "max_seqs": max_seqs,
        "learning_rate": 1.0,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        # If the dict has no entry for the bs_feat,n_ep combination, see above.
        "learning_rate_piecewise_steps": _lrlin_oclr_steps_by_bs_nep[(max_seqs, bs_feat, n_ep)],
        "learning_rate_piecewise_values": [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
    }


config_11gb_lm_v1 = dict_update_deep(
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    {
        "optimizer.weight_decay": 1e-2,
        "calculate_exp_loss": True,
    },
)
