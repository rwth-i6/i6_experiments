"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.yang.torch.luca_ctc.configs import *
from i6_experiments.users.yang.torch.luca_ctc.configs import (
    _batch_size_factor,
    _cfg_lrlin1e_5_295k,
    _get_cfg_lrlin_oclr_by_bs_nep,
    const_linear_learning_rates,
linear_const_linear_learning_rates,
)
# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.configs import *
# from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.configs import (
#     _batch_size_factor,
#     _cfg_lrlin1e_5_295k,
#     _get_cfg_lrlin_oclr_by_bs_nep,
# )
# from .trafo_lm import trafo_lm_kazuki_import

from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc_v2 import from_scratch_model_def, from_scratch_training
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

# from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_conformer_ctc import from_scratch_model_def, from_scratch_training
# from i6_experiments.users.gaudino.models.asr.rf.conformer_ctc.model_recog_ctc_greedy import model_recog

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )

# From Mohammad, 2023-06-29
# dev-clean  2.27
# dev-other  5.39
# test-clean  2.41
# test-other  5.51
# _returnn_tf_config_filename = (
#   "/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/search/ReturnnSearchJobV2.1oORPHJTAcW0/output/returnn.config")
# E.g. via /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work
_returnn_tf_ckpt_filename = "i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb/output/model/average.index"
# /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/AverageTFCheckpointsJob.BxqgICRSGkgb
# Main train (2035 subepochs): /work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.SAh74CLCNJQi
# 15k batch size, accum grad 2 (1350 steps per epoch?)
# (With batch size 40k (here default), I have usually 495 steps/epoch. Same accum grad 2.)
# peak_lr = 0.9e-3 (1e-3 should also be fine), with Adam, optimizer_epsilon = 1e-08
# phase1: peak_lr / 10 -> peak_lr (45%)
# phase2: peak_lr -> peak_lr / 10 (45%)
# phase3: peak_lr / 10 -> 1e-6 (10%)
# all linear decay and step-based
# specaugment like my orig (same here, specaugorig), speed perturb same here.
# weight decay: L2 1e-4 in some layers (not all): FF, depthwise conv, self-att, output, LSTM, readout
# Final from learning_rates file:
# 2035: EpochData(learningRate=<misleading>, error={
# 'dev_error_ctc': 0.0520755184693418,
# 'dev_error_output/output_prob': 0.035661241551042944,
# 'dev_score_ctc': 0.2796084385705723,
# 'dev_score_output/output_prob': 0.1718613621694714,
# 'devtrain_error_ctc': 0.005757552549708462,
# 'devtrain_error_output/output_prob': 0.005408351877314902,
# 'devtrain_score_ctc': 0.022935187616968285,
# 'devtrain_score_output/output_prob': 0.05237826015574962,
# 'train_error_ctc': 0.05592114304093772,
# 'train_error_output/output_prob': 0.041970552995693494,
# 'train_score_ctc': 0.21249712733341475,
# 'train_score_output/output_prob': 0.20816428663741796,
# }),
# Retrain RETURNN training job (600 subepochs): /u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/returnn/training/ReturnnTrainingJob.ZhtaEElHqWlr
# Epoch-wise LR:
# Fixed for 20 subepochs: 1e-4
# Linear(?) decay for remaining (?): 1e-4 to 1e-6
# Final from learning_rates file:
# 600: EpochData(learningRate=1e-06, error={
# 'dev_error_ctc': 0.04999311020358747,
# 'dev_error_output/output_prob': 0.03406881170076022,
# 'dev_score_ctc': 0.2881619431223589,
# 'dev_score_output/output_prob': 0.16851828029171323,
# 'devtrain_error_ctc': 0.003611245977923651,
# 'devtrain_error_output/output_prob': 0.004028583366881808,
# 'devtrain_score_ctc': 0.014762402644778178,
# 'devtrain_score_output/output_prob': 0.0458638666428664,
# 'train_error_ctc': 0.051649620746772214,
# 'train_error_output/output_prob': 0.03977601830532325,
# 'train_score_ctc': 0.19722012590584306,
# 'train_score_output/output_prob': 0.19768974342596793,
# }),


# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80

# simple linear lr function for fine-tuning




def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    #_sis_setup_global_prefix(prefix_name)

    # Moh:      dev-clean  2.27, dev-other  5.39, test-clean  2.41,  test-other  5.51
    # RF recog: {"dev-clean": 2.25, "dev-other": 5.34, "test-clean": 2.42, "test-other": 5.56}
    # _recog_imported()

    # train_exp("from-scratch-train", config_11gb, gpu_mem=11)

    # model = train_exp(  # 5.41
    #     "base-24gb-v6-lrlin1e_5_600k",
    #     config_24gb_v6,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #     },
    # )

    # train_exp(  # dev-other 9.01
    #     "base-24gb-lrlin1e_5_600k_ctc_only",
    #     config_24gb_v6,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers":[],
    #     },
    # )
    #
    # train_exp(  # dev-other 6.93
    #     "base-24gb-lrlin1e_5_600k_ctc_only_aux4_8",
    #     config_24gb_v6,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #     },
    # )
    #
    #
    # # without mel normalization
    # train_exp(  # dev-other
    #     "base-24gb-lrlin1e_5_600k_ctc_only_no_mel_norm",
    #     config_24gb_v6,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "aux_loss_layers":[],
    #         "mel_normalization_ted2": False,
    #     },
    # )

    # train_exp(  # dev-other 7.17
    #     "base-24gb-lrlin1e_5_600k_ctc_only_aux4_8_no_mel_norm_debug",
    #     config_24gb_v6,
    #     config_updates={
    #         "learning_rate": 1.0,
    #         "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "mel_normalization_ted2": False,
    #     },
    # )

    ########################################
    # lr_1 = 1e-6
    # lr_2 = 1e-5
    # lr_3 = 1e-6
    # ep1 = 28
    # ep2 = 2
    # ep3 = 30
    # lrs = linear_const_linear_learning_rates(lr_1,lr_2,lr_3, ep1,ep2, ep3)
    # top_k = [200,1000]
    # batch_sizes = {
    #     200: 20_000,
    #     1000: 16_000,
    # }
    # grad_accum_dict = {
    #     200: 4,
    #     1000: 5,
    # }
    # kd_scale = 0.2
    # for k in top_k:
    #     train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
    #         f"debug_kd_scale-{kd_scale}top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #         config_24gb_finetune,
    #         config_updates={
    #             "learning_rate": float(lrs[-1]),
    #             "learning_rates": lrs,
    #             #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #             # total steps after 2000 epochs: 982.312
    #             # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #             # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #             "__num_epochs": ep1+ep2+ep3,
    #             "mask_eos_output": True,
    #             "accum_grad_multiple_step": grad_accum_dict[k],
    #             "batch_size": batch_sizes[k] * 160,
    #             "add_eos_to_blank": True,
    #             "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.fC2OVTjvivBw/output/models/epoch.040.pt"}},
    #             "mel_normalization_ted2": False,
    #             "kd_top_k": k,
    #             "kd_scale": kd_scale,
    #         },
    #         post_config_updates={
    #             "cleanup_old_models": {"keep": [5]}
    #         }
    #     )

    lr_1 = 1e-6
    lr_2 = 1e-6
    lr_3 = 1e-6
    ep1 = 2
    ep2 = 18
    ep3 = 20
    lrs = linear_const_linear_learning_rates(lr_1,lr_2,lr_3, ep1,ep2, ep3)
    kd_scale = 0.2
    recog_epoch = np.linspace(0, ep1+ep2+ep3, (ep1+ep2+ep3)//5+1, dtype=int)
    recog_epoch = list(map(int,list(recog_epoch)[1:]))
    extra_epochs = [2, 8, 12, 24]
    train_lm_scales = [0.6]
    target_kd_scale = 0.2
    ############### only eos mask
    for i in extra_epochs:
        recog_epoch.append(i)
    for train_lm_scale in train_lm_scales:
        train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
            #f"debug_eosmask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
            f"debug_only_target_kd_scale-{target_kd_scale}_lm_scale-{train_lm_scale}",
            config_24gb_finetune,
            config_updates={
                "learning_rate": float(lrs[-1]),
                "learning_rates": lrs,
                #"dynamic_learning_rate": dyn_lr_piecewise_linear,
                # total steps after 2000 epochs: 982.312
                # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
                # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
                "__num_epochs": ep1+ep2+ep3,
                "mask_eos_output": True,
                "accum_grad_multiple_step": 2,
                "batch_size": 35_000 * 160,
                "add_eos_to_blank": True,
                "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.fC2OVTjvivBw/output/models/epoch.040.pt"}},
                "mel_normalization_ted2": False,
                "kd_scale": kd_scale,
                "eos_mask": True,
                "train_lm_scale": train_lm_scale,
                "no_beam_prefix": True,
                "target_kd_scale": target_kd_scale,

            },
            post_config_updates={
                "cleanup_old_models": {"keep": recog_epoch}
            }
        )
    # multi-gpu setup
    train_lm_scales = [1.0]
    const_lr = [1e-6] * 20
    for train_lm_scale in train_lm_scales:
        train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
            #f"debug_eosmask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
            f"debug_gpu4-only_target_kd_scale-{target_kd_scale}_lm_scale-{train_lm_scale}",
            config_11gb_multigpu_4_finetune,
            config_updates={
                "learning_rate": 1e-6,
                "learning_rates": const_lr,
                #"dynamic_learning_rate": dyn_lr_piecewise_linear,
                # total steps after 2000 epochs: 982.312
                # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
                # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
                "__num_epochs": 20,
                "mask_eos_output": True,
                "accum_grad_multiple_step": 1,
                "batch_size": 12_000 * 160,
                "add_eos_to_blank": True,
                "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.fC2OVTjvivBw/output/models/epoch.040.pt"}},
                "mel_normalization_ted2": False,
                "kd_scale": kd_scale,
                "eos_mask": True,
                "train_lm_scale": train_lm_scale,
                "no_beam_prefix": True,
                "target_kd_scale": target_kd_scale,

            },
            post_config_updates={
                "cleanup_old_models": {"keep": recog_epoch}
            }
        )


    #### multi gpu training test
    # train_lm_scales = [0.6,]
    # recog_epoch = [2,5,8,10,15,20]
    # const_lr = [1e-6] *20
    # for k in top_k:
    #     for train_lm_scale in train_lm_scales:
    #         train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
    #             f"debug_multigpu-4-eosmask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #             config_11gb_multigpu_4_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": const_lr,
    #                 #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": 20,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": 1,
    #                 "batch_size": 7000 * 160,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.fC2OVTjvivBw/output/models/epoch.040.pt"}},
    #                 "mel_normalization_ted2": False,
    #                 "kd_top_k": k,
    #                 "kd_scale": kd_scale,
    #                 "eos_mask": True,
    #                 "train_lm_scale": train_lm_scale,
    #                 "target_in_top_mask": False,
    #
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch}
    #             }
    #         )
    #
    # lr_1 = 1e-5
    # lr_2 = 1e-6
    # ep1 = 20
    # ep2 = 40
    # lrs = const_linear_learning_rates(lr_1,lr_2,ep1,ep2)
    # train_exp(  # dev-other 7.17
    #     f"finetune-24gb-lrlin{lr_1}_ep{ep1}_{lr_2}_ep{ep2}-mask-eos",
    #     config_24gb_finetune,
    #     config_updates={
    #         "learning_rate": float(lrs[-1]),
    #         "learning_rates": lrs,
    #         #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "__num_epochs": ep1+ep2,
    #         "mask_eos_output": True,
    #         "add_eos_to_blank": False,
    #         "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240605/epoch.1981.pt"}},
    #         "mel_normalization_ted2": False,
    #     },
    # )
    #
    # lr_1 = 1e-6
    # lr_2 = 1e-7
    # ep1 = 10
    # ep2 = 30
    # lrs = const_linear_learning_rates(lr_1,lr_2,ep1,ep2)
    # train_exp(  # dev-other 7.17
    #     f"finetune-24gb-lrlin{lr_1}_ep{ep1}_{lr_2}_ep{ep2}-add-eos",
    #     config_24gb_finetune,
    #     config_updates={
    #         "learning_rate": float(lrs[-1]),
    #         "learning_rates": lrs,
    #         #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #         # total steps after 2000 epochs: 982.312
    #         # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #         # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #         "__num_epochs": ep1+ep2,
    #         "mask_eos_output": True,
    #         "add_eos_to_blank": True,
    #         "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240605/epoch.1981.pt"}},
    #         "mel_normalization_ted2": False,
    #     },
    # )

_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name
# def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
#     if not prefix_name:
#         from .sis_setup import get_prefix_for_config
#
#         prefix_name = get_prefix_for_config(__file__)
#     global _sis_prefix
#     _sis_prefix = prefix_name


def _recog(
    name: str,
    model_with_checkpoint: ModelWithCheckpoint,
    recog_def: RecogDef = None,
    recog_config: Optional[Dict[str, Any]] = None,
    *,
    search_rqmt: Optional[Dict[str, Any]] = None,
    dev_sets: Optional[Collection[str]] = None,
):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    if recog_def is None:
        recog_def = model_recog

    task = _get_ls_task()

    res = recog_model(
        task,
        model_with_checkpoint,
        recog_def=recog_def,
        config=recog_config,
        search_rqmt=search_rqmt,
        dev_sets=dev_sets,
    )
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    fine_tune: Optional[Union[int, List[Tuple[int, Dict[str, Any]]]]] = None,
    time_rqmt: Optional[int] = None,
    model_avg: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
        train,
    )
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

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        model_def=from_scratch_model_def,
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(
        prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    )

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
            config_["import_model_train_epoch1"] = model_with_checkpoint.get_epoch(
                ep
            ).checkpoint
            config_.pop("dynamic_learning_rate")
            lrs = opts.pop("learning_rates", None)
            if lrs is None:
                lr_decay_type = opts.pop(
                    "lr_decay_type", "geomspace"
                )  # geomspace or linspace
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
                model_def=from_scratch_model_def,
                train_def=from_scratch_training,
                num_epochs=num_epochs_,
                gpu_mem=gpu_mem,
            )
            # _recog(name + suffix + "/recog/last", finetune_model_with_ckpt.get_last_fixed_epoch())
            recog_training_exp(
                prefix + suffix, task, finetune_model_with_ckpt, recog_def=model_recog
            )

    return model_with_checkpoint


_ls_task = None


def _get_ls_task():
    global _ls_task
    if _ls_task:
        return _ls_task

    from i6_experiments.users.zeyer.datasets.librispeech import (
        get_librispeech_task_bpe10k_raw, get_librispeech_task_raw_v2
    )

    #_ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True) luca's dataloading
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k")
    return _ls_task


py = sis_run_with_prefix  # if run directly via `sis m ...`


def model_warmup(*, model: Model, **_kwargs):
    """warmup, for more reliable timing measures"""
    import torch
    import time
    from returnn.config import get_global_config
    from returnn.tensor import Dim
    import returnn.frontend as rf

    config = get_global_config()
    start_time = time.monotonic()
    limit = start_time + config.float("model_warmup_time", 10.0)

    print("*** warming up...")
    while time.monotonic() < limit:
        batch_dim = Dim(10, name="dummy_batch")
        time_dim = Dim(rf.full(dims=[batch_dim], fill_value=16_000), name="dummy_time")
        feat_dim = Dim(1, name="dummy_feat")
        source = rf.zeros([batch_dim, time_dim, feat_dim])
        res = model.encode(source=source, in_spatial_dim=time_dim)
        if source.raw_tensor.device.type == "cuda":
            torch.cuda.synchronize(source.raw_tensor.device)
        res  # noqa  # keep ref to make sure it is calculated