"""
Attention-based encoder-decoder (AED) experiments
"""

from __future__ import annotations

import copy
import functools
from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List
import tree
import math
import numpy as np
import hashlib
import contextlib

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from returnn.frontend.decoder.transformer import TransformerDecoder

from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from i6_experiments.users.zeyer.returnn.models.rf_layerdrop import SequentialLayerDrop
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.accum_grad_schedules.piecewise_linear import dyn_accum_grad_piecewise_linear

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(prefix_name)

    # 5.60
    # train_exp("base-24gb-v6-lrlin1e_5_450k", config_24gb_v6, config_updates=_get_cfg_lrlin_oclr_by_bs_nep(40_000, 2000))
    #
    # train_exp(  # 5.64
    #     "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_295k",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    # )
    # train_exp(  # 5.51
    #     "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    #     config_updates={"optimizer.weight_decay": 1e-2},
    # )
    train_exp(  # 5.45
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={"optimizer.weight_decay": 1e-2},
    )

    model = train_exp(  # 5.11 (!!)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )
    for name, recog_config in {
        "v1": {
            "beam_search_version": 1,
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
        },
        "v3": {
            "beam_search_version": 3,
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
        },
        "v3-lenReward01": {
            "beam_search_version": 3,
            "beam_size": 12,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
        "v3-beam60": {
            "beam_search_version": 3,
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 1.0,
        },
        "v3-beam60-lenReward01": {
            "beam_search_version": 3,
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
    }.items():
        _recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2/recog_last_" + name,
            model.get_last_fixed_epoch(),
            model_recog_pure_torch,
            recog_config,
        )

    train_exp(  # 5.18 (but "test-other": 6.4)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin2e_5_295k-speedpertV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500, peak_lr=2e-3),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    model = train_exp(  # 5.44 ("test-other": 6.34), worse than speedpertV2 (5.11)
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV3",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        config_updates={
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )
    for name, recog_config in {
        "beam12-batch200": {
            # {"dev-clean": 2.64, "dev-other": 5.44, "test-clean": 2.65, "test-other": 6.33}
            # WTF, why is dev-other here much better? test-clean as well.
            "beam_search_version": 3,
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
            "__batch_size_dependent": True,
        },
        "beam12-batch1": {
            # {"dev-clean": 3.38, "dev-other": 6.23, "test-clean": 2.9, "test-other": 6.26}
            # WTF, why is this so much worse than batch50?
            "beam_search_version": 3,
            "beam_size": 12,
            "length_normalization_exponent": 1.0,
            "__batch_size_dependent": True,
            "max_seqs": 1,
        },
        "beam60-batch50": {
            # {"dev-clean": 2.92, "dev-other": 6.2, "test-clean": 2.84, "test-other": 6.52}
            "beam_search_version": 3,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 1.0,
        },
        "beam60-batch1": {
            # {"dev-clean": 2.89, "dev-other": 6.21, "test-clean": 2.84, "test-other": 6.58}
            "beam_search_version": 3,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 1,
            "length_normalization_exponent": 1.0,
        },
        "beam60-lenReward01-batch50": {
            # {"dev-clean": 2.61, "dev-other": 5.4, "test-clean": 2.82, "test-other": 6.5}
            # test-other: work/i6_core/recognition/scoring/ScliteJob.Acv12UewxtG0/output
            # Percent Substitution      =    4.4%   (2280)
            # Percent Deletions         =    0.6%   ( 292)
            # Percent Insertions        =    1.6%   ( 830)
            # work/i6_core/returnn/forward/ReturnnForwardJobV2.ALH3cRPRSWkr/output/output.py.gz
            # lots of insertions, repeating loop at end: 4294-14317-0014
            "beam_search_version": 4,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
        "beam60-lenReward01-batch50-v5": {  # TODO temp test v5
            "beam_search_version": 5,
            "__batch_size_dependent": True,
            "__recog_def_ext": True,
            "beam_search_collect_individual_seq_scores": True,
            "beam_size": 60,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.1,
        },
        "beam60-lenReward02-batch50": {
            # {"dev-clean": 2.84, "dev-other": 5.39, "test-clean": 2.82, "test-other": 6.49}
            "beam_search_version": 4,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.2,
        },
        "beam60-lenReward005-batch50": {
            # {"dev-clean": 2.61, "dev-other": 5.41, "test-clean": 2.83, "test-other": 6.48}
            "beam_search_version": 4,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.05,
        },
        "beam60-lenReward_005-batch50": {
            # {"dev-clean": 3.76, "dev-other": 5.86, "test-clean": 3.6, "test-other": 6.26}
            "beam_search_version": 4,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": -0.05,
        },
        "beam60-lenNorm0-lenReward0-batch50": {
            # {"dev-clean": 2.64, "dev-other": 5.4, "test-clean": 2.82, "test-other": 6.02}
            # test-other: work/i6_core/recognition/scoring/ScliteJob.hHxGodUNMmaC/output
            # Percent Substitution      =    4.3%   (2254)
            # Percent Deletions         =    0.9%   ( 477)
            # Percent Insertions        =    0.8%   ( 422)
            "beam_search_version": 3,
            "beam_size": 60,
            "__batch_size_dependent": True,
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "length_normalization_exponent": 0.0,
            "length_reward": 0.0,
        },
    }.items():
        _recog(
            "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV3/recog_last_" + name,
            model.get_last_fixed_epoch(),
            model_recog_pure_torch,
            recog_config,
        )

    from .aed_online_data_filter import from_scratch_model_def as aed_online_data_filter_model_def
    from .aed_online_data_filter import from_scratch_training as aed_online_data_filter_train_def

    train_exp(
        "v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-dataFilterV1",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_def=aed_online_data_filter_model_def,
        model_config={"behavior_version": 20},  # new Trafo decoder defaults
        train_def=aed_online_data_filter_train_def,
        config_updates={
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    train_exp(  # 5.84, overfits more
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
    )

    train_exp(  # 5.73, but 6.99 on test-other, how can that be so different?
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_3-lrlin1e_5_100k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"optimizer.weight_decay": 1e-3},
    )
    train_exp(  # 5.65
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_100k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"optimizer.weight_decay": 1e-2},
    )
    train_exp(  # 6.36, too aggressive
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_1-lrlin1e_5_100k",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"optimizer.weight_decay": 1e-1},
    )

    train_exp(  # 5.9 (vs 5.84)
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-mixup",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"mixup": {}},
    )

    train_exp(  # 5.80 (vs 5.84)
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-speedpertV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    train_exp(  # 5.89 (vs 5.84), way reduced overfitting, maybe too aggressive?
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-layerdrop01",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"enc_layer_drop": 0.1, "dec_layer_drop": 0.1},
    )
    train_exp(  # 5.78. big diff between dev and test?
        # {"dev-clean": 2.85, "dev-other": 5.78, "test-clean": 3.24, "test-other": 6.15}
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-layerdrop005",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"enc_layer_drop": 0.05, "dec_layer_drop": 0.05},
    )

    train_exp(  # 7.60 (vs 5.84). unclear?
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-aux12",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"aux_loss_layers": [12]},
    )
    train_exp(  # 5.85 (vs 5.84). but bad on test? "test-clean": 3.42, "test-other": 6.14
        "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k-aux8",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
        config_updates={"aux_loss_layers": [8]},
    )

    train_exp(  # 5.91
        "v6-bhv20-nenc17-11gb-f32-bs10k-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17-dynGradAccumV3",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
            "aux_loss_layers": [17],
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [20_000, 100_000, 200_000],
            "accum_grad_piecewise_values": [4, 4, 2, 1],
        },
    )

    train_exp(  # 5.86
        "v6-bhv20-nenc17-11gb-f32-bs10k-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17-dynGradAccumV3a",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
            "aux_loss_layers": [17],
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [443_000, 887_000, 986_000],
            "accum_grad_piecewise_values": [4, 4, 2, 1],
        },
    )

    train_exp(  # 5.82
        "v6-bhv20-nenc17-11gb-f32-bs10k-accgrad4-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
            "aux_loss_layers": [17],
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
            "accum_grad_multiple_step": 4,
        },
    )

    train_exp(  # 5.84
        "v6-bhv20-nenc17-11gb-f32-bs10k-accgrad4-mgpu4-pavg100-wd1e_2-lrlin1e_5_443k-aux17",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
            "aux_loss_layers": [17],
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
            "accum_grad_multiple_step": 4,
            "optimizer.weight_decay": 1e-2,
        },
    )

    train_exp(  # 5.21 {"dev-clean": 3.33, "dev-other": 5.21, "test-clean": 2.57, "test-other": 6.5}
        "v6-bhv20-nenc17-11gb-f32-bs8k-accgrad4-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-aux6_12-speedpertV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "accum_grad_multiple_step": 4,
            "optimizer.weight_decay": 1e-4,
            "aux_loss_layers": [6, 12],
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    # broken
    # train_exp(
    #     "v6-bhv20-nenc17-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17-dynGradAccumV2",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    #     model_config={
    #         "behavior_version": 20,  # new Trafo decoder defaults
    #         "num_enc_layers": 17,
    #         "aux_loss_layers": [17],
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
    #         "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
    #         "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
    #         "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
    #     },
    # )

    train_exp(  # 5.69
        "v6-bhv20-nenc17-11gb-f32-bs10k-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17-dynGradAccumV4",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        model_config={
            "behavior_version": 20,  # new Trafo decoder defaults
            "num_enc_layers": 17,
            "aux_loss_layers": [17],
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [443_000, 887_000, 986_000],
            "accum_grad_piecewise_values": [4, 1, 4, 12],
        },
    )

    # broken
    # train_exp(
    #     "v6-bhv20-nenc17-11gb-f32-bs10k-mgpu4-pavg100-wd1e_4-lrlin1e_5_443k-aux17-dynGradAccumV4inv",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    #     model_config={
    #         "behavior_version": 20,  # new Trafo decoder defaults
    #         "num_enc_layers": 17,
    #         "aux_loss_layers": [17],
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep(10_000, 500),
    #         "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
    #         "accum_grad_piecewise_steps": [443_000, 887_000, 986_000],
    #         "accum_grad_piecewise_values": [1, 4, 1, 1],
    #     },
    # )

    # TODO...
    #   - more specaug?
    #   - more speed pert?
    # train_exp(
    #     "v6-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_100k",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k,
    #     config_updates={
    #     },
    # )

    # TODO pretrain with specaugment_steps=(0, 15k, 25k)?


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _recog(
    name: str,
    model_with_checkpoint: ModelWithCheckpoint,
    recog_def: RecogDef,
    recog_config: Optional[Dict[str, Any]] = None,
):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    task = _get_ls_task()

    res = recog_model(task, model_with_checkpoint, recog_def=recog_def, config=recog_config)
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    model_def: Optional[Union[ModelDefWithCfg, ModelDef[Model]]] = None,
    train_def: Optional[TrainDef[Model]] = None,
    model_config: Optional[Dict[str, Any]] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task()
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    if not model_def:
        model_def = from_scratch_model_def
    if model_config:
        model_def = ModelDefWithCfg(model_def, model_config)
    if not train_def:
        train_def = from_scratch_training
    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=model_def,
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(prefix, task, model_with_checkpoint, recog_def=model_recog)

    if fine_tune:
        if isinstance(fine_tune, int):
            fine_tune = [(fine_tune, {})]
        for ep, opts in fine_tune:
            assert isinstance(ep, int) and isinstance(opts, dict)
            suffix = f"/finetune/{ep}"
            opts = opts.copy()
            if opts:
                for k, v in sorted(opts.items()):
                    k: str
                    suffix += "-" + k.lstrip("_")
                    v = str(v).replace("-", "_")
                    if len(v) > 16 and not k.startswith("_"):
                        suffix += "_" + hashlib.md5(v.encode("utf8")).hexdigest()[:8]
                    else:
                        suffix += v
            num_epochs_ = opts.pop("num_epochs", 50)
            config_ = config.copy()
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(ep).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop("lr_decay_type", "geomspace")  # geomspace or linspace
                lr_decay_func = getattr(np, lr_decay_type)
                lr = config_["learning_rate"]
                final_lr = opts.pop("final_lr", 1e-7)
                lrs = list(lr_decay_func(lr, final_lr, num=num_epochs_))
            else:
                assert isinstance(lrs, (list, tuple))
                assert len(lrs) == num_epochs_
            config_["learning_rates"] = lrs
            config_["learning_rate"] = float(lrs[-1])
            config_["specaugment_steps"] = (0, 0, 0)
            config_.update({k: v for k, v in opts.items() if not k.startswith("_")})

            finetune_model_with_ckpt = train(
                prefix + suffix,
                task=task,
                config=config_,
                post_config=post_config,
                model_def=model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog)

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True)
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

_batch_size_factor = 160


class MakeModel:
    """for import"""

    def __init__(self, in_dim: int, target_dim: int, *, eos_label: int = 0, num_enc_layers: int = 12):
        self.in_dim = in_dim
        self.target_dim = target_dim
        self.eos_label = eos_label
        self.num_enc_layers = num_enc_layers

    def __call__(self) -> Model:
        from returnn.datasets.util.vocabulary import Vocabulary

        in_dim = Dim(name="in", dimension=self.in_dim, kind=Dim.Types.Feature)
        target_dim = Dim(name="target", dimension=self.target_dim, kind=Dim.Types.Feature)
        target_dim.vocab = Vocabulary.create_vocab_from_labels(
            [str(i) for i in range(target_dim.dimension)], eos_label=self.eos_label
        )

        return self.make_model(in_dim, target_dim, num_enc_layers=self.num_enc_layers)

    @classmethod
    def make_model(
        cls, in_dim: Dim, target_dim: Dim, *, num_enc_layers: int = 12, pos_emb_dropout: float = 0.0, **extra
    ) -> Model:
        """make"""
        return Model(
            in_dim,
            num_enc_layers=num_enc_layers,
            enc_model_dim=Dim(name="enc", dimension=512, kind=Dim.Types.Feature),
            enc_ff_dim=Dim(name="enc-ff", dimension=2048, kind=Dim.Types.Feature),
            enc_att_num_heads=8,
            enc_conformer_layer_opts=dict(
                conv_norm_opts=dict(use_mask=True),
                self_att_opts=dict(
                    # Shawn et al 2018 style, old RETURNN way.
                    with_bias=False,
                    with_linear_pos=False,
                    with_pos_bias=False,
                    learnable_pos_emb=True,
                    separate_pos_emb_per_head=False,
                    pos_emb_dropout=pos_emb_dropout,
                ),
                ff_activation=lambda x: rf.relu(x) ** 2.0,
            ),
            target_dim=target_dim,
            blank_idx=target_dim.dimension,
            bos_idx=_get_bos_idx(target_dim),
            eos_idx=_get_eos_idx(target_dim),
            **extra,
        )


class Model(rf.Module):
    """Model definition"""

    def __init__(
        self,
        in_dim: Dim,
        *,
        num_enc_layers: int = 12,
        num_dec_layers: int = 6,
        target_dim: Dim,
        wb_target_dim: Optional[Dim] = None,
        blank_idx: int,
        eos_idx: int,
        bos_idx: int,
        enc_aux_logits: Sequence[int] = (),  # layers
        enc_model_dim: Dim = Dim(name="enc", dimension=512),
        dec_model_dim: Dim = Dim(name="dec", dimension=512),
        enc_ff_dim: Dim = Dim(name="enc-ff", dimension=2048),
        enc_att_num_heads: int = 4,
        enc_conformer_layer_opts: Optional[Dict[str, Any]] = None,
        enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
        att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
        att_dropout: float = 0.1,
        enc_dropout: float = 0.1,
        enc_att_dropout: float = 0.1,
    ):
        super(Model, self).__init__()

        from returnn.config import get_global_config

        config = get_global_config(return_empty_if_none=True)

        enc_layer_drop = config.float("enc_layer_drop", 0.0)
        if enc_layer_drop:
            enc_sequential = functools.partial(SequentialLayerDrop, layer_drop=enc_layer_drop)
        else:
            enc_sequential = rf.Sequential
        dec_layer_drop = config.float("dec_layer_drop", 0.0)
        if dec_layer_drop:
            dec_sequential = functools.partial(SequentialLayerDrop, layer_drop=dec_layer_drop)
        else:
            dec_sequential = rf.Sequential

        self.in_dim = in_dim
        self.encoder = ConformerEncoder(
            in_dim,
            enc_model_dim,
            ff_dim=enc_ff_dim,
            input_layer=ConformerConvSubsample(
                in_dim,
                out_dims=[Dim(32, name="conv1"), Dim(64, name="conv2"), Dim(64, name="conv3")],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],
            ),
            encoder_layer_opts=enc_conformer_layer_opts,
            num_layers=num_enc_layers,
            num_heads=enc_att_num_heads,
            dropout=enc_dropout,
            att_dropout=enc_att_dropout,
            sequential=enc_sequential,
        )
        self.decoder = TransformerDecoder(
            num_layers=num_dec_layers,
            encoder_dim=enc_model_dim,
            vocab_dim=target_dim,
            model_dim=dec_model_dim,
            sequential=dec_sequential,
        )

        self.target_dim = target_dim
        self.blank_idx = blank_idx
        self.eos_idx = eos_idx
        self.bos_idx = bos_idx  # for non-blank labels; for with-blank labels, we use bos_idx=blank_idx

        self.enc_key_total_dim = enc_key_total_dim
        self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
        self.att_num_heads = att_num_heads
        self.att_dropout = att_dropout
        self.dropout_broadcast = rf.dropout_broadcast_default()

        if enc_aux_logits:
            if not wb_target_dim:
                wb_target_dim = target_dim + 1
        for i in enc_aux_logits:
            setattr(self, f"enc_aux_logits_{i}", rf.Linear(self.encoder.out_dim, wb_target_dim))

        self._specaugment_opts = {
            "steps": config.typed_value("specaugment_steps") or (0, 1000, 2000),
            "max_consecutive_spatial_dims": config.typed_value("specaugment_max_consecutive_spatial_dims") or 20,
            "max_consecutive_feature_dims": config.typed_value("specaugment_max_consecutive_feature_dims")
            or (_log_mel_feature_dim // 5),
            "num_spatial_mask_factor": config.typed_value("specaugment_num_spatial_mask_factor") or 100,
        }

        self._pretrain_opts: Optional[Dict[str, Any]] = config.typed_value("pretrain_opts")

        self._mixup = None
        if config.typed_value("mixup", None) is not None:
            from i6_experiments.users.zeyer.returnn.models.rf_mixup import Mixup, MixupOpts

            self._mixup = Mixup(feature_dim=self.in_dim, opts=MixupOpts(**config.typed_value("mixup")))

    def encode(
        self,
        source: Tensor,
        *,
        in_spatial_dim: Dim,
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[rf.State, Dim]:
        """encode, and extend the encoder output for things we need in the decoder"""
        # log mel filterbank features
        source, in_spatial_dim = rf.audio.log_mel_filterbank_from_raw(
            source,
            in_spatial_dim=in_spatial_dim,
            out_dim=self.in_dim,
            sampling_rate=16_000,
            log_base=math.exp(2.3026),  # almost 10.0 but not exactly...
        )
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
        with _opt_apply_pretrain_to_encoder(self.encoder, collected_outputs, self._pretrain_opts):
            enc, enc_spatial_dim = self.encoder(
                source, in_spatial_dim=in_spatial_dim, collected_outputs=collected_outputs
            )
        return self.decoder.transform_encoder(enc, axis=enc_spatial_dim), enc_spatial_dim


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


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> Model:
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config()  # noqa
    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    num_enc_layers = config.int("num_enc_layers", 12)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)
    return MakeModel.make_model(
        in_dim,
        target_dim,
        num_enc_layers=num_enc_layers,
        enc_aux_logits=enc_aux_logits or (),
        pos_emb_dropout=pos_emb_dropout,
    )


from_scratch_model_def: ModelDef[Model]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_loss = rf.ctc_loss(
                logits=aux_logits,
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

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_labels = rf.shift_right(targets, axis=targets_spatial_dim, pad_value=model.bos_idx)

    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
    )

    logits_packed, pack_dim = rf.pack_padded(logits, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False)
    targets_packed, _ = rf.pack_padded(
        targets, dims=batch_dims + [targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
    )

    log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
    log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
    loss = rf.cross_entropy(
        target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
    )
    loss.mark_as_loss("ce", scale=aed_loss_scale, use_normalized_loss=use_normalized_loss)

    best = rf.reduce_argmax(logits_packed, axis=model.target_dim)
    frame_error = best != targets_packed
    frame_error.mark_as_loss(name="fer", as_error=True)


from_scratch_training: TrainDef[Model]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


@contextlib.contextmanager
def _opt_apply_pretrain_to_encoder(
    encoder: ConformerEncoder, collected_outputs: Optional[Dict[str, Tensor]], pretrain_opts: Optional[Dict[str, Any]]
):
    """Function is run within RETURNN."""
    if not pretrain_opts:
        yield
        return
    step = rf.get_run_ctx().step
    steps: Union[Sequence[Tuple[int, Dict[str, Any]]], Dict[int, Dict[str, Any]]] = pretrain_opts["steps"]
    if isinstance(steps, (list, tuple)):
        steps_ = {}
        step_bound = 0
        for step_bound_rel, opts in steps:
            step_bound += step_bound_rel
            steps_[step_bound] = opts
        steps = steps_
    assert isinstance(steps, dict)
    for step_bound, opts in sorted(steps.items()):
        if step < step_bound:
            assert isinstance(opts, dict)
            opts_ = opts.copy()
            # somewhat hacky but that is still the easiest way I can think of, without touching a lot of other code
            pretrain_num_layers = opts_.pop("num_layers")
            assert not opts_, f"unhandled opts: {opts_} in opts {opts} for step bound {step_bound}"
            orig_layers = encoder.layers[:]
            del encoder.layers[pretrain_num_layers:]
            yield
            encoder.layers[:] = orig_layers
            if collected_outputs is not None:
                assert len(collected_outputs) == pretrain_num_layers
                for i in range(pretrain_num_layers, len(orig_layers)):
                    collected_outputs[str(i)] = collected_outputs[str(pretrain_num_layers - 1)]
            return
    yield
    return


def model_recog(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
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
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    beam_size = 12
    length_normalization_exponent = 1.0
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    # Eager-mode implementation of beam search.
    # Initial state.
    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim)
    ended = rf.constant(False, dims=batch_dims_)
    out_seq_len = rf.constant(0, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        logits, decoder_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            encoder=enc,
            state=decoder_state,
        )
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
        # Filter out finished beams
        label_log_prob = rf.where(
            ended,
            rf.sparse_to_dense(model.eos_idx, axis=model.target_dim, label_value=0.0, other_value=-1.0e30),
            label_log_prob,
        )
        seq_log_prob = seq_log_prob + label_log_prob  # Batch, InBeam, Vocab
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, model.target_dim]
        )  # seq_log_prob, backrefs, target: Batch, Beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        decoder_state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), decoder_state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == model.eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= rf.where(
                ended,
                (i / (i - 1)) ** length_normalization_exponent,
                1.0,
            )

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

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
    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


def _gather_backrefs(s, *, backrefs: Tensor):
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)  # really the default case
        return s  # e.g. scalar or so, independent from beam
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims  # currently not supported, also not expected
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


# RecogDef API
model_recog: RecogDef[Model]
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False


def model_recog_pure_torch(
    *,
    model: Model,
    data: Tensor,
    data_spatial_dim: Dim,
    max_seq_len: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        recog results info: key -> {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    import time
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search import BeamSearchOpts, beam_search
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v3 import beam_search_v3
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v4 import beam_search_v4
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import beam_search_v5
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.length_reward import LengthRewardScorer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.shallow_fusion import ShallowFusedLabelScorers
    from returnn.config import get_global_config

    config = get_global_config()

    start_time = time.perf_counter_ns()

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
    assert len(batch_dims) == 1, batch_dims  # not implemented otherwise, simple to add...
    batch_dim = batch_dims[0]
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    if max_seq_len is None:
        max_seq_len = enc_spatial_dim.get_size_tensor()
    else:
        max_seq_len = rf.convert_to_tensor(max_seq_len, dtype="int32")
    print("** max seq len:", max_seq_len.raw_tensor)

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    beam_search_version = config.int("beam_search_version", 1)
    beam_search_func = {1: beam_search, 3: beam_search_v3, 4: beam_search_v4, 5: beam_search_v5}[beam_search_version]
    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    if beam_search_opts.get("beam_size") is None:
        beam_search_opts["beam_size"] = config.int("beam_size", 12)
    if beam_search_opts.get("length_normalization_exponent") is None:
        beam_search_opts["length_normalization_exponent"] = config.float("length_normalization_exponent", 1.0)
    if beam_search_opts.get("length_reward") is None:
        beam_search_opts["length_reward"] = config.float("length_reward", 0.0)
    extra = {}
    out_individual_seq_scores = None
    if beam_search_version >= 5 and config.bool("beam_search_collect_individual_seq_scores", False):
        out_individual_seq_scores = {}
        extra["out_individual_seq_scores"] = out_individual_seq_scores
    coverage_scale = beam_search_opts.pop("attention_coverage_scale", 0.0)
    label_scorer = ShallowFusedLabelScorers()
    if coverage_scale:
        s1, s2 = get_label_scorer_and_coverage_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc)
        # Note: insertion order matters here, we want that decoder is scored first.
        label_scorer.label_scorers["decoder"] = (s1, 1.0)
        label_scorer.label_scorers["attention_coverage"] = (s2, coverage_scale)
    else:
        label_scorer.label_scorers["decoder"] = (
            get_label_scorer_pure_torch(model=model, batch_dim=batch_dim, enc=enc),
            1.0,
        )
    if beam_search_version >= 5:
        len_reward = beam_search_opts.pop("length_reward", 0.0)
        label_scorer.label_scorers["length_reward"] = (LengthRewardScorer(), len_reward)

    # Beam search happening here:
    (
        seq_targets,  # [Batch,FinalBeam,OutSeqLen]
        seq_log_prob,  # [Batch,FinalBeam]
        out_seq_len,  # [Batch,FinalBeam]
    ) = beam_search_func(
        label_scorer,
        batch_size=batch_dim.get_dim_value(),
        max_seq_len=max_seq_len.copy_compatible_to_dims_raw([batch_dim]),
        device=data.raw_tensor.device,
        opts=BeamSearchOpts(
            **beam_search_opts,
            bos_label=model.bos_idx,
            eos_label=model.eos_idx,
            num_labels=model.target_dim.dimension,
        ),
        **extra,
    )

    beam_dim = Dim(seq_log_prob.shape[1], name="beam")
    out_spatial_dim = Dim(rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim], name="out_spatial"))
    seq_targets_t = rf.convert_to_tensor(
        seq_targets, dims=[batch_dim, beam_dim, out_spatial_dim], sparse_dim=model.target_dim
    )
    seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

    search_end_time = time.perf_counter_ns()
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"enc {enc_end_time - start_time} ns",
                f"dec {search_end_time- enc_end_time} ns",
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"enc len {enc_spatial_dim.get_dim_value()}",
                f"out len {out_spatial_dim.get_dim_value()}",
            )
        ),
    )

    extra_recog_results = {}
    if out_individual_seq_scores:
        for k, v in out_individual_seq_scores.items():
            extra_recog_results[f"score:{k}"] = rf.convert_to_tensor(v, dims=[batch_dim, beam_dim])

    return seq_targets_t, seq_log_prob_t, extra_recog_results, out_spatial_dim, beam_dim


def get_label_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: rf.State,
):
    import torch
    import functools
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
            return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

            logits, decoder_state = model.decoder(
                rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim),
                spatial_dim=single_step_dim,
                encoder=enc,
                state=tree.map_structure(_map_raw_to_tensor, prev_state),
            )
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

    return LabelScorer()


def get_label_scorer_and_coverage_scorer_pure_torch(
    *,
    model: Model,
    batch_dim: Dim,
    enc: rf.State,
):
    import torch
    import functools
    from returnn.frontend.decoder.transformer import TransformerDecoderLayer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )

    accum_att_weights = rf.zeros(())  # [Batch,Beam,kv_axis]
    accum_att_weights_dec_frame: Tensor  # [Batch,Beam,kv_axis]
    beam_dim: Dim
    enc_spatial_dim: Dim

    def hooked_cross_att(self: rf.CrossAttention, q: Tensor, k: Tensor, v: Tensor, *, kv_axis: Dim) -> Tensor:
        """apply attention"""
        nonlocal enc_spatial_dim
        enc_spatial_dim = kv_axis
        nonlocal accum_att_weights_dec_frame
        # Standard dot attention, inline rf.dot_attention.
        q *= self.key_dim_per_head.dimension**-0.5
        energy = rf.matmul(q, k, reduce=self.key_dim_per_head)
        att_weights = rf.softmax(energy, axis=kv_axis)
        accum_att_weights_dec_frame = rf.maximum(accum_att_weights, rf.reduce_max(att_weights, axis=self.num_heads))
        # Masking not needed because softmax should already have masked,
        # so we have 0.0 att weights for padded frames.
        att = rf.matmul(att_weights, v, reduce=kv_axis, use_mask=False)
        if v.feature_dim in att.dims:
            att.feature_dim = v.feature_dim
        output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
        if self.proj:
            output = self.proj(output)
        return output

    for layer in model.decoder.layers:
        layer: TransformerDecoderLayer
        layer.cross_att.attention = functools.partial(hooked_cross_att, self=layer.cross_att)

    class LabelScorer(LabelScorerIntf):
        """label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            beam_dim = Dim(1, name="initial-beam")
            batch_dims_ = [batch_dim, beam_dim]
            decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
            return tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state)

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            nonlocal beam_dim
            beam_dim = Dim(prev_label.shape[1], name="beam")

            def _map_raw_to_tensor(v):
                if isinstance(v, StateObjTensorExt):
                    tensor: Tensor = v.extra
                    tensor = tensor.copy_template_new_dim_tags(
                        (batch_dim, beam_dim) + tensor.dims[2:], keep_special_axes=True
                    )
                    tensor.raw_tensor = v.tensor
                    return tensor
                elif isinstance(v, StateObjIgnored):
                    return v.content
                else:
                    raise TypeError(f"_map_raw_to_tensor: unexpected {v} ({type(v).__name__})")

            nonlocal accum_att_weights, accum_att_weights_dec_frame
            accum_att_weights_dec_frame = rf.zeros(())
            logits, decoder_state = model.decoder(
                rf.convert_to_tensor(prev_label, dims=[batch_dim, beam_dim], sparse_dim=model.target_dim),
                spatial_dim=single_step_dim,
                encoder=enc,
                state=tree.map_structure(_map_raw_to_tensor, prev_state),
            )
            accum_att_weights += accum_att_weights_dec_frame
            label_log_prob = rf.log_softmax(logits, axis=model.target_dim)
            assert set(label_log_prob.dims) == {batch_dim, beam_dim, model.target_dim}

            return (
                self._map_tensor_to_raw(label_log_prob, beam_dim=beam_dim).tensor,
                tree.map_structure(functools.partial(self._map_tensor_to_raw, beam_dim=beam_dim), decoder_state),
            )

        @staticmethod
        def _map_tensor_to_raw(v, *, beam_dim: Dim):
            if isinstance(v, Tensor):
                if beam_dim not in v.dims:
                    return StateObjIgnored(v)
                batch_dims_ = [batch_dim, beam_dim]
                v = v.copy_transpose(batch_dims_ + [dim for dim in v.dims if dim not in batch_dims_])
                raw = v.raw_tensor
                return StateObjTensorExt(raw, v.copy_template())
            elif isinstance(v, Dim):
                return StateObjIgnored(v)
            else:
                raise TypeError(f"_map_tensor_to_raw: unexpected {v} ({type(v).__name__})")

    class CoverageScorer(LabelScorerIntf):
        """coverage

        Google NMT: https://arxiv.org/pdf/1609.08144.pdf
        Alternative: https://arxiv.org/abs/1612.02695
        Another alternative: https://arxiv.org/pdf/2105.00982.pdf
        """

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """Initial state."""
            return {"prev_score": torch.zeros([batch_size, 1], device=device)}

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """update state"""
            prev_label  # noqa  # unused
            # We assume the label scorer has run before us (make sure by right ordering).
            assert set(accum_att_weights.dims) == {batch_dim, beam_dim, enc_spatial_dim}
            # log1p, to avoid having lots of negative numbers. So this starts more around 0.0.
            coverage_score_raw = rf.reduce_sum(
                rf.log1p(rf.minimum(accum_att_weights, 1.0)), axis=enc_spatial_dim
            ).copy_compatible_to_dims_raw((batch_dim, beam_dim))
            state = {"prev_score": coverage_score_raw}
            return (coverage_score_raw - prev_state["prev_score"])[:, :, None], state

    return LabelScorer(), CoverageScorer()


# RecogDef API
model_recog_pure_torch: RecogDef[Model]
model_recog_pure_torch.output_with_beam = True
model_recog_pure_torch.output_blank_label = None
model_recog_pure_torch.batch_size_dependent = False
