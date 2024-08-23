"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection
import tree
import math
import numpy as np
import hashlib
import copy
import contextlib
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample

from i6_experiments.users.gaudino.model_interfaces.supports_label_scorer_torch import (
    RFModelWithMakeLabelScorer,
)

from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.configs import *
from i6_experiments.users.yang.torch.albert_exp2024_04_23_baselines.configs import (
    _get_cfg_lrlin_oclr_by_bs_nep,
    _fine_tune_get_cfg_lrlin_oclr_by_bs_nep,
)
# from .trafo_lm import trafo_lm_kazuki_import

from i6_experiments.users.yang.torch.luca_ctc.model_conformer_kd_ctc import from_scratch_model_def, from_scratch_training
#from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy

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

    # lr_1 = 1e-6
    # lr_2 = 1e-5
    # lr_3 = 1e-6
    # ep1 = 28
    # ep2 = 2
    # ep3 = 30
    # lrs = linear_const_linear_learning_rates(lr_1,lr_2,lr_3, ep1,ep2, ep3)
    # top_k = [200]
    # batch_sizes = {
    #     200: 16_000,
    #     1000: 16_000, # still have gpu mem issue
    # }
    # grad_accum_dict = {
    #     16_000: 5,
    #     20_000: 4,
    # }
    # kd_scale = 0.2
    # recog_epoch = np.linspace(0, ep1+ep2+ep3, (ep1+ep2+ep3)//5+1, dtype=int)
    # recog_epoch = list(map(int,list(recog_epoch)[1:]))
    # extra_epochs = [2, 8, 12, 56,57,58,59,52,54]
    # train_lm_scales =  [1.0] #[0.6, 1.0]
    # ############### only eos mask
    # for i in extra_epochs:
    #     recog_epoch.append(i)
    # for k in top_k:
    #     for train_lm_scale in train_lm_scales:
    #         train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
    #             f"debug_eosmask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #             config_24gb_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": ep1+ep2+ep3,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": grad_accum_dict[batch_sizes[k]],
    #                 "batch_size": batch_sizes[k] * 160,
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
    #
    #
    #
    # for k in top_k:
    #     for train_lm_scale in train_lm_scales:
    #         train_exp(  # baseline dev-other 7.03 finetune-24gb-lrlin1e-06_ep10_1e-07_ep30-add-eos, add eos, 40 epochs
    #             f"debug_eos_mask_target_mask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #             config_24gb_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 # "dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": ep1 + ep2 + ep3,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": grad_accum_dict[batch_sizes[k]],
    #                 "batch_size": batch_sizes[k] * 160,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True,
    #                                                 "filename": "/work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.fC2OVTjvivBw/output/models/epoch.040.pt"}},
    #                 "mel_normalization_ted2": False,
    #                 "kd_top_k": k,
    #                 "kd_scale": kd_scale,
    #                 "eos_mask": True,
    #                 "train_lm_scale": train_lm_scale,
    #                 "target_in_top_mask": True,
    #
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch}
    #             }
    #         )

    # lr_1 = 1e-6
    # lr_2 = 1e-5
    # lr_3 = 1e-6
    # ep1 = 28
    # ep2 = 2
    # ep3 = 30
    # lrs = linear_const_linear_learning_rates(lr_1,lr_2,lr_3, ep1,ep2, ep3)
    # top_k = [20]
    # batch_sizes = {
    #     200: 15_000,
    #     1000: 15_000, # still have gpu mem issue
    #     20: 15_000
    # }
    # grad_accum_dict = {
    #     15_000: 5,
    #     20_000: 4,
    # }
    # kd_scale = 0.2
    # #recog_epoch = np.linspace(0, ep1+ep2+ep3, (ep1+ep2+ep3)//5+1, dtype=int)
    # #recog_epoch = list(map(int,list(recog_epoch)[1:]))
    # recog_epoch = [5, 60]
    # #extra_epochs = [2, 8, 12, 56,57,58,59,52,54, 36,38,16,18]
    # extra_epochs= []
    # train_lm_scales =  [1.0] #[0.6, 1.0]
    # # new checkpoint
    # ############### only eos mask
    # base_checkpoint_path = "/u/zyang/setups/rf/alias/conformer_only_target/debug_only_target_kd_scale-0.2_lm_scale-0.6/train/output/models/epoch.024.pt"
    # for i in extra_epochs:
    #     recog_epoch.append(i)
    # for k in top_k:
    #     for train_lm_scale in train_lm_scales:
    #         train_exp(  # baseline dev-other 6.9
    #             f"debug_eosmask_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #             config_24gb_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": ep1+ep2+ep3,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": grad_accum_dict[batch_sizes[k]],
    #                 "batch_size": batch_sizes[k] * 160,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #                 "mel_normalization_ted2": False,
    #                 "kd_top_k": k,
    #                 "kd_scale": kd_scale,
    #                 "eos_mask": True,
    #                 "train_lm_scale": train_lm_scale,
    #                 "target_in_top_mask": False,
    #                 "specaugment_steps": (0,0,0),
    #
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch}
    #             },
    #             greedy_search=True,
    #             length_norm_scale=1.0,
    #         )

        #already trained model:
        #debug_eosmask_kd_scale-0.2-trainlm-1.0-top-20_lr1-1e-06_lr2-1e-05_lr3-1e-06_ep1-28_ep2-2_ep3-30: /work/asr4/zyang/rf/work/i6_core/returnn/training/ReturnnTrainingJob.Zh1t4GvlSqnD
        #

    # for i in extra_epochs:
    #     recog_epoch.append(i)
    # train_lm_scales = [1.0]
    # for k in [20]:
    #     for train_lm_scale in train_lm_scales:
    #         train_exp(  # baseline dev-other 6.9
    #             f"debug_eosmask_freeze_gamma_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}",
    #             config_24gb_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": ep1+ep2+ep3,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": grad_accum_dict[batch_sizes[k]],
    #                 "batch_size": batch_sizes[k] * 160,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #                 "mel_normalization_ted2": False,
    #                 "kd_top_k": k,
    #                 "kd_scale": kd_scale,
    #                 "eos_mask": True,
    #                 "train_lm_scale": train_lm_scale,
    #                 "target_in_top_mask": False,
    #                 "specaugment_steps": (0,0,0),
    #                 "freeze_gamma": True,
    #
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch}
    #             }
    #         )



    # lr_1 = 2e-6
    # lr_2 = 1e-5
    # lr_3 = 1e-6
    # ep1 = 28
    # ep2 = 2
    # ep3 = 30
    # lrs = linear_const_linear_learning_rates(lr_1,lr_2,lr_3, ep1,ep2, ep3)
    # top_k = [20]
    # batch_sizes = {
    #     200: 15_000,
    #     1000: 15_000, # still have gpu mem issue
    #     20: 10_000
    # }
    # grad_accum_dict = {
    #     15_000: 5,
    #     20_000: 4,
    #     10_000: 8,
    # }
    # kd_scales = [0.0, 0.2,0.4]
    # recog_epoch = np.linspace(0, ep1+ep2+ep3, (ep1+ep2+ep3)//5+1, dtype=int)
    # recog_epoch = list(map(int,list(recog_epoch)[1:]))
    # extra_epochs = [1,2, 8, 12, 56,57,58,59,52,54, 36,38,16,18]
    # train_lm_scales = [0.6]
    # kd_layer=8
    # new checkpoint
    ############### only eos mask
    #base_checkpoint_path = "/u/zyang/setups/rf/alias/conformer_only_target/debug_only_target_kd_scale-0.2_lm_scale-0.6/train/output/models/epoch.024.pt"
    base_checkpoint_path = "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt" # trained by Luca, without eos {"dev-clean": 3.02, "dev-other": 6.8, "test-clean": 3.16, "test-other": 7.07},
    # for i in extra_epochs:
    #     recog_epoch.append(i)
    # for kd_scale in kd_scales:
    #     for train_lm_scale in train_lm_scales:
    #         k = 20
    #         train_exp(  # baseline dev-other 6.9
    #             f"debug_eosmask_kd_layer-{kd_layer}_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1-{lr_1}_lr2-{lr_2}_lr3-{lr_3}_ep1-{ep1}_ep2-{ep2}_ep3-{ep3}_singlegpu12",
    #             config_11gb_singalgpu_finetune,
    #             config_updates={
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                 # total steps after 2000 epochs: 982.312
    #                 # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                 # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                 "__num_epochs": ep1+ep2+ep3,
    #                 "mask_eos_output": True,
    #                 "accum_grad_multiple_step": grad_accum_dict[batch_sizes[k]] *2,
    #                 "batch_size": batch_sizes[k] * 80,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #                 "mel_normalization_ted2": False,
    #                 "kd_top_k": k,
    #                 "kd_scale": kd_scale,
    #                 "eos_mask": True,
    #                 "train_lm_scale": train_lm_scale,
    #                 "target_in_top_mask": False,
    #                 "specaugment_steps": (0,0,0),
    #                 "kd_layer": kd_layer,
    #
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch}
    #             },
    #             greedy_search=True,
    #         )
    kd_layer = 12
    kd_scales = [0.5]
    # kd_scale 0.4, dev-other 7.7, broken
    epoch = 125

    recog_epoch = np.linspace(0, epoch, epoch // 5 + 1, dtype=int)
    recog_epoch = list(map(int, list(recog_epoch)[1:]))
    extra_epochs = [118, 122,124,116,114,108]
    for i in extra_epochs:
        recog_epoch.append(i)


    train_lm_scales = [0.6]
    for kd_scale in kd_scales:
        for train_lm_scale in train_lm_scales:
            k = 10
            train_exp(
                f"debug_eosmask_kd_layer-{kd_layer}_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1e-3_4gpu12-epoch{epoch}",
                config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep(8_000, epoch),
                    #"dynamic_learning_rate": dyn_lr_piecewise_linear,
                    # total steps after 2000 epochs: 982.312
                    # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
                    # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
                    "mask_eos_output": True,
                    "add_eos_to_blank": False,
                    "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                    "mel_normalization_ted2": False,
                    "kd_top_k": k,
                    "kd_scale": kd_scale,
                    "eos_mask": True,
                    "train_lm_scale": train_lm_scale,
                    "target_in_top_mask": False,
                    #"specaugment_steps": (0,0,0),
                    "kd_layer": kd_layer,

                },
                 post_config_updates={
                     "cleanup_old_models": {"keep": recog_epoch}
                 },
                greedy_search=True,
                reserve_code="hlt_9"
                #mem_rqmt=30,
            )

    train_lm_scales = [0.6]
    kd_scales = [0.5, 0.2]
    epoch = 125

    recog_epoch = np.linspace(0, epoch, epoch // 5 + 1, dtype=int)
    recog_epoch = list(map(int, list(recog_epoch)[1:]))
    extra_epochs = [118, 122,124,116,114,108]
    for i in extra_epochs:
        recog_epoch.append(i)
    for i in extra_epochs:
        recog_epoch.append(i)
    for kd_scale in kd_scales:
        for train_lm_scale in train_lm_scales:
            k = 10
            train_exp(
                f"debug_eosmask_kd_layer-{kd_layer}_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1e-3_singlegpu12-epoch{epoch}",
                config_11gb_v6_f32_accgrad1_gpu1_wd1e_4,
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep(8_000, epoch),
                    #"dynamic_learning_rate": dyn_lr_piecewise_linear,
                    # total steps after 2000 epochs: 982.312
                    # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
                    # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
                    "mask_eos_output": True,
                    "add_eos_to_blank": False,
                    "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
                    "mel_normalization_ted2": False,
                    "kd_top_k": k,
                    "kd_scale": kd_scale,
                    "eos_mask": True,
                    "train_lm_scale": train_lm_scale,
                    "target_in_top_mask": False,
                    #"specaugment_steps": (0,0,0),
                    "kd_layer": kd_layer,
                    "accum_grad_multiple_step": 1,

                },
                post_config_updates={
                    "cleanup_old_models": {"keep": recog_epoch}
                },
                greedy_search=True,
                mem_rqmt=30,
            )
    # kd_scales = [0.5]
    # for grad_accu in [2,4]:
    #     for kd_scale in kd_scales:
    #         for train_lm_scale in train_lm_scales:
    #             k = 10
    #             train_exp(
    #                 f"debug_eosmask_kd_layer-{kd_layer}_kd_scale-{kd_scale}-trainlm-{train_lm_scale}-top-{k}_lr1e-4_singlegpu12-finetune-epoch{epoch}-gradaccu{grad_accu}",
    #                 config_11gb_v6_f32_accgrad1_gpu1_wd1e_4,
    #                 config_updates={
    #                     **_fine_tune_get_cfg_lrlin_oclr_by_bs_nep(8_000, epoch, peak_lr=1e-4),
    #                     #"dynamic_learning_rate": dyn_lr_piecewise_linear,
    #                     # total steps after 2000 epochs: 982.312
    #                     # "learning_rate_piecewise_steps": [600_000, 900_000, 982_000],
    #                     # "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #                     "mask_eos_output": True,
    #                     "add_eos_to_blank": False,
    #                     "preload_from_files": {"base": {"init_for_train": True, "ignore_missing": True, "filename": base_checkpoint_path}},
    #                     "mel_normalization_ted2": False,
    #                     "kd_top_k": k,
    #                     "kd_scale": kd_scale,
    #                     "eos_mask": True,
    #                     "train_lm_scale": train_lm_scale,
    #                     "target_in_top_mask": False,
    #                     #"specaugment_steps": (0,0,0),
    #                     "kd_layer": kd_layer,
    #                     "accum_grad_multiple_step": grad_accu,
    #
    #                 },
    #                 post_config_updates={
    #                     "cleanup_old_models": {"keep": recog_epoch}
    #                 },
    #                 greedy_search=True,
    #                 mem_rqmt=30,
    #             )


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
    mem_rqmt: Optional[int] = None,
    reserve_code: Optional[str] = None,
    model_avg: bool = False,
    greedy_search=True,
    length_norm_scale=0.0,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    # from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
    #     train,
    # )
    from i6_experiments.users.yang.torch.luca_ctc.train import train
    #from i6_experiments.users.yang.torch.luca_ctc.recog import recog_training_exp
    from i6_experiments.users.yang.torch.decoding.recog import recog_training_exp
    #from i6_experiments.users.zeyer.recog import recog_training_exp

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
        mem_rqmt=mem_rqmt,
        reserve_code=reserve_code,
    )
    # greedy search:
    if greedy_search:
        from i6_experiments.users.yang.torch.decoding.ctc_greedy import model_recog_greedy
        recog_training_exp(
            prefix, task, model_with_checkpoint, recog_def=model_recog_greedy,
        )
    else:
        print('fixed epochs################', model_with_checkpoint.fixed_epochs)
        recog_config_update = {'batch_size':1600000, "length_norm_scale": length_norm_scale}
        from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog
        recog_training_exp(
            prefix + '-label_sync-lennorm_scale_' +f"{length_norm_scale}", task, model_with_checkpoint, search_config=recog_config_update, recog_def=model_recog, model_avg=model_avg
        )

    # ctc label sync search


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
