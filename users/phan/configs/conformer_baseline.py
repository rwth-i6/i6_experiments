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
import itertools
from sisyphus import tk

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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, default_tedlium2_extern_lm_config


from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)

from i6_core.returnn.training import PtCheckpoint

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


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    lr_list = [(1e-3, 1e-4)]
    ep_list = [(40, 80)]
    # ground_truth_weights = [0.5, "average"]
    # recog_epoch = [1] + list(range(20, 120, 20))
    recog_epoch = list(range(20, 120, 20)) + [1]
    ground_truth_weights = [0.5]
    train_exp( 
        f"conformer_baseline",
        config_11gb,
        gpu_mem=11,
        config_updates={
            "batch_size": 1200000,
            "learning_rate": 0.,
            "learning_rates": [],
            "__num_epochs": 0,
            "mask_eos_output": True,
            "add_eos_to_blank": True,
            "preload_from_files": {
                "base": {
                    "init_for_train": True,
                    "ignore_missing": True,
                    "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
                }
            },
            "mel_normalization_ted2": False,
        },
        post_config_updates={
            "cleanup_old_models": {"keep": recog_epoch},
            "torch_dataloader_opts": { # otherwise it will break after every epoch
                "num_workers": 0,
            }
        },
        greedy_search = False,
)



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
    from i6_experiments.users.yang.torch.luca_ctc.recog import recog_training_exp, recog_model
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

    # model_with_checkpoint = train(
    #     prefix,
    #     task=task,
    #     config=config,
    #     post_config=dict_update_deep(post_config, post_config_updates),
    #     model_def=from_scratch_model_def,
    #     train_def=from_scratch_training_kldiv_sample_batch,
    #     num_epochs=num_epochs,
    #     gpu_mem=gpu_mem,
    #     num_processes=num_processes,
    #     distributed_launch_cmd="torchrun" if num_processes else "mpirun",
    #     time_rqmt=time_rqmt,
    #     mem_rqmt = mem_rqmt,
    # )
    # # greedy search: we won't use greedy search anyway
    # if greedy_search:
    #     from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
    #     recog_training_exp(
    #         prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    #     )
    # Luca's label sync search with LM and ILM scale

    recog_config_update = {
        # "torch_amp": "bfloat16",
        # 'batch_size':1200000,
        'batch_size': 200000,
        "length_norm_scale": length_norm_scale,
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "external_language_model": {
            "class": "Trafo_LM_Model",
            "layer_out_dim": 1024,
            "layer_ff_dim": 4096,
            "embed_dim": 128,
            "num_layers": 24,
            "att_num_heads": 8,
            "use_pos_enc": True,
            "ff_activation": "relu",
            "pos_enc_diff_pos": True,
        }, # this to load the external LM only in recog
    }
    # # from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog
    # from i6_experiments.users.phan.recog.ctc_label_sync import model_recog_label_sync as model_recog
    # from i6_experiments.users.phan.recog.ctc_label_sync_v2 import model_recog as model_recog_label_sync_v2
    # from i6_experiments.users.phan.recog.ctc_time_sync import model_recog_time_sync
    # beam_sizes = [32]
    # lm_scales = [0.65, 0.75, 0.85, 0.95]
    # ilm_scales = [0.0, 0.4, 0.5, 0.6, 0.7] # 0.3 not always good, no nede to tune
    # # ilm_scales = [0.0]
    # length_norm_scales = [1.0, 0.0]
    # prior_scales = [0.0, 0.1, 0.2, 0.3] # [0.2, 0.4, 0.6, 0.8] not better
    # # beam_sizes = [32]
    # # lm_scales = [0.65]
    # # ilm_scales = [0.0]
    # # length_norm_scales = [1.0]
    # # prior_scales = [0.0]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     if prior_scale > 0.1 and length_norm_scale != 0.0:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #     }
    #     if prior_scale > 0.0:
    #         search_args.update({
    #             "prior_scale": prior_scale,
    #             "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #             "ctc_log_prior": False,
    #         })
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     if ilm_scale > 0.0:
    #         recog_config_update_extra["preload_from_files"].update({ # transcription ilm
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
    #             }
    #         })
    #         recog_config_update_extra.update({
    #             "internal_language_model": default_ilm_config,
    #         })
    #     # new_checkpoint = PtCheckpoint(tk.Path(""))
    #     suffix = f"_labelSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = prefix + "/conformer_baseline_transIlm_ep5" + suffix
        
    #     res = recog_model(
    #         task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_label_sync_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt=None,
    #         dev_sets=["dev-other", "test-other"],
    #         name=recog_name,
    #     )
    #     tk.register_output(_sis_prefix + "/conformer_baseline_transIlm_ep5" + suffix + "/recog_results_per_epoch/baseline", res.output)

    #     # ep 1 of transcription
    #     if ilm_scale > 0.0:
    #         recog_config_update_extra["preload_from_files"].update({ # transcription ilm
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.Jo4iRk3pZrRS/output/models/epoch.001.pt",
    #             }
    #         })
    #         recog_name_ep1 = prefix + "/conformer_baseline_transIlm_ep1" + suffix
            
    #         res = recog_model(
    #             task,
    #             ModelWithCheckpoint(
    #                 definition=from_scratch_model_def,
    #                 checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #             ),
    #             recog_def=model_recog_label_sync_v2,
    #             config=recog_config_update_extra,
    #             search_rqmt=None,
    #             dev_sets=["dev-other", "test-other"],
    #             name=recog_name_ep1,
    #         )
    #         tk.register_output(_sis_prefix + "/conformer_baseline_transIlm_ep1" + suffix + "/recog_results_per_epoch/baseline", res.output)

    # # ------------------ prior with zero encoder ------------------
    # # beam_sizes = [32]
    # # lm_scales = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    # # length_norm_scales = [1.0]
    # # prior_scales = [0.1, 0.2, 0.3, 0.4, 0.5]
    # beam_sizes = [16, 32]
    # lm_scales = [0.55, 0.65, 0.75, 0.85, 0.95]
    # length_norm_scales = [1.0]
    # prior_scales = [0.1, 0.2, 0.3, 0.4, 0.5] # bad scales: [0.6, 0.7, 0.8, 0.9]
    # for beam_size, lm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, length_norm_scales, prior_scales):
    #     # if prior_scale > lm_scale:
    #     #     continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "prior_type": "zero_encoder",
    #         "prior_scale": prior_scale,
    #     }
    #     exp_name = "/conformer_baseline_zeroEncoderPrior"
    #     suffix = f"_labelSync_beam-{beam_size}_lm-{lm_scale}_ilm-{0.0}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     res = recog_model(
    #         task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_label_sync_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt=None,
    #         dev_sets=["dev-other", "test-other"],
    #         name=recog_name,
    #     )
    #     tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # # ------------------ zero encoder prefix score --------------------
    # # ------- this replaces the 100% masking ILM by directly calculate  
    # # ------- the prefix scores when zeroing out the encoder
    # # -----------------------------------------------------------------
    # # beam_sizes = [32]
    # # lm_scales = [0.55, 0.65, 0.75, 0.85, 0.95, 1.05]
    # # length_norm_scales = [1.0]
    # # prior_scales = [0.1, 0.2, 0.3, 0.4, 0.5]
    # beam_sizes = [16] # must be 16!!!
    # lm_scales = [0.65, 0.75, 0.85, 0.95]
    # ilm_scales = [0.3, 0.4, 0.5, 0.6, 0.7]
    # length_norm_scales = [1.0]
    # prior_scales = [0.0, 0.1] # bad scales: [0.6, 0.7, 0.8, 0.9]
    # from i6_experiments.users.phan.recog.ctc_label_sync_v2_zero_encoder_prefix_score import model_recog as model_recog_label_sync_zero_encoder_prefix
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):
    #     # if prior_scale > lm_scale:
    #     #     continue
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }

    #     exp_name = "/conformer_baseline_zeroEncoderPrefixScore"
    #     suffix = f"_labelSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     res = recog_model(
    #         task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_label_sync_zero_encoder_prefix,
    #         config=recog_config_update_extra,
    #         search_rqmt=None,
    #         dev_sets=["dev-other", "test-other"],
    #         name=recog_name,
    #     )
    #     tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)
        
    # # ------------------- time synchronous search baseline ---------------------
    # # !!!!!!! need fix
    # # Luca's best:
    # # optsr_ctc1.0_trafolm0.7_fix2_prior0.8_fix_beam32/recog_results 
    # # {"dev-clean": 2.14, "dev-other": 4.66, "test-clean": 2.31, "test-other": 5.15} 
    # # Important: turn on LM skip 
    # from i6_experiments.users.phan.recog.ctc_time_sync_v2 import model_recog_time_sync as model_recog_time_sync_v2
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # # lm_scales = [0.6, 0.7, 0.8, 0.9]
    # # ilm_scales = [0.0, 0.4, 0.5, 0.6]
    # # prior_scales = [0.7, 0.8, 0.9]
    # lm_scales = [0.8, 0.9, 1.0, 1.1]
    # # ilm_scales = [0.0, 0.2, 0.4, 0.6] # 0.2 should be best
    # ilm_scales = [0.0, 0.1, 0.2, 0.3, 0.4] 
    # # prior_scales = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # [0.8]
    # prior_scales = [0.0, 0.3, 0.4, 0.5] # [0.8]
    # prior_types = ["precomputed_average", "zero_encoder"]
    # # prior_types = ["precomputed_average"]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
    #     if ilm_scale >= lm_scale:
    #         continue
    #     # Try this first ... We didn't try combination for label sync as well
    #     if prior_type == "zero_encoder":
    #         if ilm_scale != 0.0:
    #             continue
    #     # else:
    #     # if ilm_scale == 0.0 and prior_type == "precomputed_average":
    #     #     if lm_scale != 0.7 or prior_scale != 0.8:
    #     #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "lm_skip": True, # IMPORTANT!!!
    #     }
    #     if prior_scale > 0.0:
    #         search_args["prior_scale"] = prior_scale
    #         if prior_type == "precomputed_average":
    #             search_args.update({
    #                 "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #                 "ctc_log_prior": False,
    #             })
    #         elif prior_type == "zero_encoder":
    #             search_args.update({"prior_type": prior_type})
    #     prior_type_str = {
    #         "precomputed_average": "",
    #         "zero_encoder": "_zeroEncoderPrior",
    #     }
    #     exp_name = "/conformer_baseline_transIlm_ep1" + prior_type_str[prior_type]
    #     suffix = f"_timeSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "batch_size": 600000,
    #         "search_args": search_args,
    #     })
    #     if ilm_scale > 0.0:
    #         recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 1
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.Jo4iRk3pZrRS/output/models/epoch.001.pt",
    #             }
    #         })
    #         recog_config_update_extra.update({
    #             "internal_language_model": default_ilm_config,
    #         })
    #     res = recog_model(
    #         task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_time_sync_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt={"time": 5},
    #         dev_sets=["dev-other", "test-other"],
    #         name=recog_name,
    #     )
    #     tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    #     # ep 5 transcription LM
    #     if ilm_scale > 0.0:
    #         exp_name = "/conformer_baseline_transIlm_ep5" + prior_type_str[prior_type]
    #         suffix = f"_timeSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #         recog_name = prefix + exp_name + suffix
    #         recog_config_update_extra = copy.deepcopy(recog_config_update)
    #         recog_config_update_extra.update({
    #             "batch_size": 600000,
    #             "search_args": search_args,
    #         })
    #         recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 1
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
    #             }
    #         })
    #         recog_config_update_extra.update({
    #             "internal_language_model": default_ilm_config,
    #         })
    #         res = recog_model(
    #             task,
    #             ModelWithCheckpoint(
    #                 definition=from_scratch_model_def,
    #                 checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #             ),
    #             recog_def=model_recog_time_sync_v2,
    #             config=recog_config_update_extra,
    #             search_rqmt={"time": 5},
    #             dev_sets=["dev-other", "test-other"],
    #             name=recog_name,
    #         )
    #         tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)


    # ---------------- Compute statistics -----------------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev-other", "test-other"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "external_language_model": default_extern_lm_config,
        "internal_language_model": default_ilm_config,
    })
    forward_extra_config["preload_from_files"].update({
        "01_trafo_lm": {
            "prefix": "language_model.",
            "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
        },
        "02_lstm_ilm": { # ep 5 of transcription
            "prefix": "ilm.",
            "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
        }
    })
    forward_post_config = dict(
        torch_log_memory_usage=True,
        use_lovely_tensors=True,
    )
    
    for dataset_key in dataset_keys:
        if dataset_key == "train": # not tested
            forward_dataset = task.train_dataset
        else:
            forward_dataset = task.eval_datasets[dataset_key]
        for transcription_lm_epoch in [5]:
            epoch = "baseline"
            checkpoint = ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            )
            stats_job = generic_forward_config.generic_forward_job(
                dataset=forward_dataset,
                model=checkpoint,
                forward_def=compute_kldiv.forward_compute_kldiv,
                forward_callback=compute_kldiv.forward_callback_wrapper,
                forward_extra_config=forward_extra_config,
                forward_post_config=forward_post_config,
                output_files=compute_kldiv.output_files,
                dataset_key=dataset_key,
                job_vis_name=f"Compute ILM stats job, {name}, epoch baseline, {dataset_key}",
                extra_hash={"version": "04/11/2024"}, 
            )
            out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
            stats_job.add_alias(_sis_prefix + f"/conformer_baseline_transIlm_ep{transcription_lm_epoch}" + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
            tk.register_output(_sis_prefix + f"/conformer_baseline_transIlm_ep{transcription_lm_epoch}" + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)
    
    # ------------------- time synchronous search baseline ---------------------
    # !!!!!!! need fix
    # Luca's best:
    # optsr_ctc1.0_trafolm0.7_fix2_prior0.8_fix_beam32/recog_results 
    # {"dev-clean": 2.14, "dev-other": 4.66, "test-clean": 2.31, "test-other": 5.15} 
    # Important: turn on LM skip 
    from i6_experiments.users.phan.recog.ctc_time_sync_v2 import model_recog_time_sync as model_recog_time_sync_v2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.8, 0.9, 1.0] #, 1.1, 1.2]
    ilm_scales = [0.0] 
    prior_scales = [0.2, 0.3, 0.4, 0.5]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        if ilm_scale >= lm_scale:
            continue
        # Try this first ... We didn't try combination for label sync as well
        if prior_type == "zero_encoder":
            if ilm_scale != 0.0:
                continue
        # else:
        # if ilm_scale == 0.0 and prior_type == "precomputed_average":
        #     if lm_scale != 0.7 or prior_scale != 0.8:
        #         continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                    "renormalize_prior": True,
                })
            elif prior_type == "zero_encoder":
                search_args.update({"prior_type": prior_type})
        prior_type_str = {
            "precomputed_average": "",
            "zero_encoder": "_zeroEncoderPrior",
        }
        exp_name = "/conformer_baseline" + prior_type_str[prior_type] + "_renormPrior"
        suffix = f"_timeSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 600000,
            "search_args": search_args,
        })
        res = recog_model(
            task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_v2,
            config=recog_config_update_extra,
            search_rqmt={"time": 5},
            dev_sets=["dev-other", "test-other"],
            name=recog_name,
        )
        tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)
    
    # ------------------------ cross-domain tedlium2 (estimation on LBS, recognition on ted2) -------------------------
    ted2_prefix = "lbs_cross_domain_ted2/" + _sis_prefix + "/" + "conformer_baseline_transIlm_ep5"
    ted2_task = _get_ted2_task()
    ted2_sis_prefix = "lbs_cross_domain_ted2/" + _sis_prefix

    # -------- compute ted2 ILM dev, test PPL --------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev", "test"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "with_extern_lm": False,
        "internal_language_model": default_ilm_config,
    })
    forward_extra_config["preload_from_files"].update({
        "02_lstm_ilm": { # ep 5 of transcription
            "prefix": "ilm.",
            "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
        }
    })
    forward_post_config = dict(
        torch_log_memory_usage=True,
        use_lovely_tensors=True,
    )
    
    for dataset_key in dataset_keys:
        if dataset_key == "train": # not tested
            forward_dataset = ted2_task.train_dataset
        else:
            forward_dataset = ted2_task.eval_datasets[dataset_key]
        checkpoint = ModelWithCheckpoint(
            definition=from_scratch_model_def,
            checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
        )
        stats_job = generic_forward_config.generic_forward_job(
            dataset=forward_dataset,
            model=checkpoint,
            forward_def=compute_kldiv.forward_compute_kldiv,
            forward_callback=compute_kldiv.forward_callback_wrapper,
            forward_extra_config=forward_extra_config,
            forward_post_config=forward_post_config,
            output_files=compute_kldiv.output_files,
            dataset_key=dataset_key,
            job_vis_name=f"Compute ILM stats job on tedlium2, {name}, epoch {epoch}, {dataset_key}",
        )
        out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
        stats_job.add_alias(ted2_prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
        tk.register_output(ted2_prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # ------------------- time synchronous search recombination first baseline ---------------------
    # !!!!!!! need fix
    # Luca's best:
    # optsr_ctc1.0_trafolm0.7_fix2_prior0.8_fix_beam32/recog_results 
    # {"dev-clean": 2.14, "dev-other": 4.66, "test-clean": 2.31, "test-other": 5.15} 
    # Important: turn on LM skip 
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first import model_recog_time_sync_recomb_first
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.8, 0.9, 1.0, 1.1]
    ilm_scales = [0.0, 0.1, 0.2, 0.3, 0.4] 
    prior_scales = [0.0, 0.3, 0.4, 0.5] # [0.8]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                })
        exp_name = "/conformer_baseline_transIlm_ep5"
        suffix = f"_timeSyncRecombFirst_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 600000,
            "search_args": search_args,
        })
        if ilm_scale > 0.0:
            recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                "02_lstm_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                }
            })
            recog_config_update_extra.update({
                "internal_language_model": default_ilm_config,
            })
        res = recog_model(
            task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_recomb_first,
            config=recog_config_update_extra,
            search_rqmt={"time": 5},
            dev_sets=["dev-other", "test-other"],
            name=recog_name,
        )
        tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # ------------------- time synchronous search recombination first baseline (fixed) ---------------------
    # Important: turn on LM skip 
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.8, 0.9, 1.0, 1.1]
    ilm_scales = [0.1, 0.2, 0.3, 0.4] 
    prior_scales = [0.0, 0.3, 0.4, 0.5] # [0.8]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                })
        exp_name = "/conformer_baseline_transIlm_ep5"
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 600000,
            "search_args": search_args,
        })
        if ilm_scale > 0.0:
            recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                "02_lstm_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                }
            })
            recog_config_update_extra.update({
                "internal_language_model": default_ilm_config,
            })
        res = recog_model(
            task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_recomb_first_v2,
            config=recog_config_update_extra,
            search_rqmt={"time": 5},
            dev_sets=["dev-other", "test-other"],
            name=recog_name,
        )
        tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # ------------------- time synchronous search recombination first baseline (fixed) ---------------------
    # Important: turn on LM skip 

    # -------------------------- finish dev-clean and test-clean ------------------------

    # lm 0.0 ilm 0.0 prior 0.0
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog as model_recog_greedy
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.0]
    ilm_scales = [0.0] 
    prior_scales = [0.0] # [0.8]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                })
        exp_name = "/conformer_baseline_transIlm_ep5"
        suffix = f"_greedy_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 1800000,
            "search_args": search_args,
        })
        if ilm_scale > 0.0:
            recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                "02_lstm_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                }
            })
            recog_config_update_extra.update({
                "internal_language_model": default_ilm_config,
            })
        res = recog_model(
            task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_greedy,
            config=recog_config_update_extra,
            search_rqmt={"time": 3},
            dev_sets=["dev-other", "test-other", "dev-clean", "test-clean"],
            name=recog_name,
        )
        tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # lm 0.8, ilm 0.0, prior 0.0
    # lm 0.9, ilm 0.0, prior 0.4
    # lm 1.0, ilm 0.2, prior 0.4
    # lm 0.9, ilm 0.4, prior 0.0
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.0]
    ilm_scales = [0.0] 
    prior_scales = [0.0] # [0.8]
    prior_types = ["precomputed_average"]
    for beam_size, length_norm_scale, prior_type in itertools.product(beam_sizes, length_norm_scales, prior_types):
        for lm_scale, ilm_scale, prior_scale in [(0.8, 0.0, 0.0), (0.9, 0.0, 0.4), (1.0, 0.2, 0.4), (0.9, 0.4, 0.0)]:
            search_args = {
                "beam_size": beam_size,
                "length_normalization_exponent": length_norm_scale, # by default len norm
                "lm_scale": lm_scale,
                "ilm_scale": ilm_scale,
                "lm_skip": True, # IMPORTANT!!!
            }
            if prior_scale > 0.0:
                search_args["prior_scale"] = prior_scale
                if prior_type == "precomputed_average":
                    search_args.update({
                        "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                        "ctc_log_prior": False,
                    })
            exp_name = "/conformer_baseline_transIlm_ep5"
            suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
            recog_name = prefix + exp_name + suffix
            recog_config_update_extra = copy.deepcopy(recog_config_update)
            recog_config_update_extra.update({
                "batch_size": 600000,
                "search_args": search_args,
            })
            if ilm_scale > 0.0:
                recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                    "02_lstm_ilm": {
                        "prefix": "ilm.",
                        "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                    }
                })
                recog_config_update_extra.update({
                    "internal_language_model": default_ilm_config,
                })
            res = recog_model(
                task,
                ModelWithCheckpoint(
                    definition=from_scratch_model_def,
                    checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
                ),
                recog_def=model_recog_time_sync_recomb_first_v2,
                config=recog_config_update_extra,
                search_rqmt={"time": 6},
                dev_sets=["dev-other", "test-other", "dev-clean", "test-clean"],
                name=recog_name,
            )
            tk.register_output(_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # ------------------- Cross domain recognition on tedlium2 ---------------------
    # Important: turn on LM skip 
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    # ilm_scales = [0.0, 0.2, 0.4, 0.6, 0.8] 
    # prior_scales = [0.0, 0.2, 0.4, 0.6]
    # lm_scales = [1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # ilm_scales = [0.5, 0.6, 0.7] 
    # prior_scales = [0.4, 0.5, 0.6]
    lm_scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    ilm_scales = [0.0] 
    # prior_scales = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    prior_scales = [0.0]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                })
        exp_name = "/conformer_baseline_SF_no_prior"
        suffix = f"_timeSyncRecombFirstV2_mergecontraction_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            # "batch_size": 1800000,
            "batch_size": 2400000,
            "search_args": search_args,
            "preload_from_files": {
                "01_lstm_extern_lm": {
                    "prefix": "language_model.",
                    "filename": default_ted2_lstm_extern_lm_checkpoint,
                },
            },
            "external_language_model": default_tedlium2_extern_lm_config,
        })
        if ilm_scale > 0.0:
            recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                "02_lstm_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                }
            })
            recog_config_update_extra.update({
                "internal_language_model": default_ilm_config,
            })
        res = recog_model(
            ted2_task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_recomb_first_v2,
            config=recog_config_update_extra,
            search_rqmt={"time": 5},
            dev_sets=["dev", "test"],
            name=recog_name,
            merge_contraction=True,
        )
        tk.register_output(ted2_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # a little more for the case prior = 0.0???
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
    # ilm_scales = [0.0, 0.2, 0.4, 0.6, 0.8] 
    # prior_scales = [0.0, 0.2, 0.4, 0.6]
    lm_scales = [1.3, 1.4, 1.5, 1.6, 1.7]
    ilm_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3] 
    prior_scales = [0.0]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        if prior_scale > 0.0:
            search_args["prior_scale"] = prior_scale
            if prior_type == "precomputed_average":
                search_args.update({
                    "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                    "ctc_log_prior": False,
                })
        exp_name = "/conformer_baseline_transIlm_ep5"
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 1800000,
            "search_args": search_args,
            "preload_from_files": {
                "01_lstm_extern_lm": {
                    "prefix": "language_model.",
                    "filename": default_ted2_lstm_extern_lm_checkpoint,
                },
            },
            "external_language_model": default_tedlium2_extern_lm_config,
        })
        if ilm_scale > 0.0:
            recog_config_update_extra["preload_from_files"].update({ # transcription ilm ep 5
                "02_lstm_ilm": {
                    "prefix": "ilm.",
                    "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                }
            })
            recog_config_update_extra.update({
                "internal_language_model": default_ilm_config,
            })
        res = recog_model(
            ted2_task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_recomb_first_v2,
            config=recog_config_update_extra,
            search_rqmt={"time": 5},
            dev_sets=["dev", "test"],
            name=recog_name,
        )
        tk.register_output(ted2_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # greedy on ted2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.0]
    ilm_scales = [0.0] 
    prior_scales = [0.0]
    prior_types = ["precomputed_average"]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        exp_name = "/conformer_baseline_greedy"
        suffix = f"_timeSyncRecombFirstV2_mergecontraction_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 2400000,
            "search_args": search_args,
            # "preload_from_files": {
            #     "01_lstm_extern_lm": {
            #         "prefix": "language_model.",
            #         "filename": default_ted2_lstm_extern_lm_checkpoint,
            #     },
            # },
            # "external_language_model": default_tedlium2_extern_lm_config,
        })
        res = recog_model(
            ted2_task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_greedy,
            config=recog_config_update_extra,
            search_rqmt={"time": 2},
            dev_sets=["dev", "test"],
            name=recog_name,
            merge_contraction=True,
        )
        tk.register_output(ted2_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)


    # # ------------------- Cross domain recognition on tedlium2, with ted2 transcription LM ---------------------
    # # Important: turn on LM skip 
    # # first tuning
    # estimation_on_ted2_prefix = "configs/estimation_ted2/conformer_baseline" # hardcoded...

    # # -------- compute ted2 ILM dev, test PPL --------
    # from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    # dataset_keys = ["dev", "test"]
    # forward_extra_config = copy.deepcopy(config)
    # forward_extra_config.update({
    #     "batch_size": 4800000,
    #     "max_seqs": 200,
    #     "with_extern_lm": False,
    #     "internal_language_model": default_ilm_config,
    # })
    # forward_extra_config["preload_from_files"].update({
    #     "02_lstm_ilm": { # ep 5 of transcription
    #         "prefix": "ilm.",
    #         "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.yXLtYuPi8PBC/output/models/epoch.014.pt",
    #     }
    # })
    # forward_post_config = dict(
    #     torch_log_memory_usage=True,
    #     use_lovely_tensors=True,
    # )
    
    # for dataset_key in dataset_keys:
    #     if dataset_key == "train": # not tested
    #         forward_dataset = ted2_task.train_dataset
    #     else:
    #         forward_dataset = ted2_task.eval_datasets[dataset_key]
    #     checkpoint = ModelWithCheckpoint(
    #         definition=from_scratch_model_def,
    #         checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #     )
    #     stats_job = generic_forward_config.generic_forward_job(
    #         dataset=forward_dataset,
    #         model=checkpoint,
    #         forward_def=compute_kldiv.forward_compute_kldiv,
    #         forward_callback=compute_kldiv.forward_callback_wrapper,
    #         forward_extra_config=forward_extra_config,
    #         forward_post_config=forward_post_config,
    #         output_files=compute_kldiv.output_files,
    #         dataset_key=dataset_key,
    #         job_vis_name=f"Compute ILM stats job on tedlium2, tedlium2 transcription LM, epoch {epoch}, {dataset_key}",
    #         forward_time_rqmt=0.5,
    #     )
    #     out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
    #     stats_job.add_alias(estimation_on_ted2_prefix + "/conformer_baseline_transIlm_ted2_ep14" + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
    #     tk.register_output(estimation_on_ted2_prefix + "/conformer_baseline_transIlm_ted2_ep14" + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    # from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    # ilm_scales = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4] 
    # prior_scales = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # prior_types = ["precomputed_average"]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "lm_skip": True, # IMPORTANT!!!
    #     }
    #     if prior_scale > 0.0:
    #         search_args["prior_scale"] = prior_scale
    #         if prior_type == "precomputed_average":
    #             search_args.update({
    #                 "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #                 "ctc_log_prior": False,
    #             })
    #     exp_name = "/conformer_baseline_transIlm_ted2_ep14"
    #     suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = estimation_on_ted2_prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "batch_size": 1800000,
    #         "search_args": search_args,
    #         "preload_from_files": {
    #             "01_lstm_extern_lm": {
    #                 "prefix": "language_model.",
    #                 "filename": default_ted2_lstm_extern_lm_checkpoint,
    #             },
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.yXLtYuPi8PBC/output/models/epoch.014.pt",
    #             }
    #         },
    #         "external_language_model": default_tedlium2_extern_lm_config,
    #         "internal_language_model": default_ilm_config,
    #     })
    #     # if ilm_scale > 0.0:
    #     #     recog_config_update_extra["preload_from_files"].update({ # ted2 transcription ilm ep 14
                
    #     #     })
    #     #     recog_config_update_extra.update({
                
    #     #     })
    #     res = recog_model(
    #         ted2_task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt={"time": 3},
    #         dev_sets=["dev", "test"],
    #         name=recog_name,
    #     )
    #     tk.register_output(estimation_on_ted2_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # # extra tuning without prior correction
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    # ilm_scales = [0.1, 0.2, 0.3, 0.4] 
    # prior_scales = [0.0]
    # prior_types = ["precomputed_average"]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "lm_skip": True, # IMPORTANT!!!
    #     }
    #     if prior_scale > 0.0:
    #         search_args["prior_scale"] = prior_scale
    #         if prior_type == "precomputed_average":
    #             search_args.update({
    #                 "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #                 "ctc_log_prior": False,
    #             })
    #     exp_name = "/conformer_baseline_transIlm_ted2_ep14"
    #     suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = estimation_on_ted2_prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "batch_size": 1800000,
    #         "search_args": search_args,
    #         "preload_from_files": {
    #             "01_lstm_extern_lm": {
    #                 "prefix": "language_model.",
    #                 "filename": default_ted2_lstm_extern_lm_checkpoint,
    #             },
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.yXLtYuPi8PBC/output/models/epoch.014.pt",
    #             }
    #         },
    #         "external_language_model": default_tedlium2_extern_lm_config,
    #         "internal_language_model": default_ilm_config,
    #     })
    #     # if ilm_scale > 0.0:
    #     #     recog_config_update_extra["preload_from_files"].update({ # ted2 transcription ilm ep 14
                
    #     #     })
    #     #     recog_config_update_extra.update({
                
    #     #     })
    #     res = recog_model(
    #         ted2_task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt={"time": 3},
    #         dev_sets=["dev", "test"],
    #         name=recog_name,
    #     )
    #     tk.register_output(estimation_on_ted2_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    # # extra tuning, with prior correction
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [1.6, 1.7, 1.8, 1.9, 2.0]
    # ilm_scales = [0.1, 0.2, 0.3] 
    # prior_scales = [0.6, 0.7, 0.8, 0.9, 1.0]
    # prior_types = ["precomputed_average"]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale, prior_type in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales, prior_types):
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "lm_skip": True, # IMPORTANT!!!
    #     }
    #     if prior_scale > 0.0:
    #         search_args["prior_scale"] = prior_scale
    #         if prior_type == "precomputed_average":
    #             search_args.update({
    #                 "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #                 "ctc_log_prior": False,
    #             })
    #     exp_name = "/conformer_baseline_transIlm_ted2_ep14"
    #     suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
    #     recog_name = estimation_on_ted2_prefix + exp_name + suffix
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "batch_size": 2400000,
    #         "search_args": search_args,
    #         "preload_from_files": {
    #             "01_lstm_extern_lm": {
    #                 "prefix": "language_model.",
    #                 "filename": default_ted2_lstm_extern_lm_checkpoint,
    #             },
    #             "02_lstm_ilm": {
    #                 "prefix": "ilm.",
    #                 "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.yXLtYuPi8PBC/output/models/epoch.014.pt",
    #             }
    #         },
    #         "external_language_model": default_tedlium2_extern_lm_config,
    #         "internal_language_model": default_ilm_config,
    #     })
    #     # if ilm_scale > 0.0:
    #     #     recog_config_update_extra["preload_from_files"].update({ # ted2 transcription ilm ep 14
                
    #     #     })
    #     #     recog_config_update_extra.update({
                
    #     #     })
    #     res = recog_model(
    #         ted2_task,
    #         ModelWithCheckpoint(
    #             definition=from_scratch_model_def,
    #             checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
    #         ),
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         config=recog_config_update_extra,
    #         search_rqmt={"time": 3},
    #         dev_sets=["dev", "test"],
    #         name=recog_name,
    #     )
    #     tk.register_output(estimation_on_ted2_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

    return
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

_ted2_task = None

def _get_ted2_task():
    global _ted2_task
    if _ted2_task:
        return _ted2_task
    from i6_experiments.users.phan.datasets.librispeech_tedlium2 import get_tedlium2_task_libri_bpe10k_raw
    _ted2_task = get_tedlium2_task_libri_bpe10k_raw(with_eos_postfix=False)
    return _ted2_task


py = sis_run_with_prefix  # if run directly via `sis m ...`

