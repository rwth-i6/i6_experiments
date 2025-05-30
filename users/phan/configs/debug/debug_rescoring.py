"""Copied from Albert Zeyer 25.03.2024, then modified
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union, Tuple, Sequence, List, Collection, Callable
import tree
import math
import numpy as np
import hashlib
import copy
import contextlib
import functools
import copy
import itertools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample
from sisyphus import tk

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

#from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, \
    default_tedlium2_extern_lm_config, default_tedlium2_extern_lm_hardcoded_layers_config
from i6_experiments.users.phan.rf_models.trafo_lm_luca import Trafo_LM_Model
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


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)

# set the ILM hyperparams for all experiments
# use trafo ILM for all exps
trafo_ilm_config = { # should be used for transcription LM as well
    "class": "Trafo_LM_Model",
    "layer_out_dim": 256,
    "layer_ff_dim": 1024,
    "embed_dim": 128,
    "num_layers": 6,
    "att_num_heads": 8,
    "use_pos_enc": True,
    "ff_activation": "relu",
    "pos_enc_diff_pos": True,
    "dropout": 0.1,
    "attn_dropout": 0.1,
}

config_11gb.update({"internal_language_model": default_ilm_config})


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    lr_list = [(1e-3, 1e-3)] # [(1e-5, 1e-7), (1e-3, 1e-3)]
    ep_list = [(40, 60)]
    # recog_epoch = [1] + list(range(20, 120, 20)) # 1 is mostly for debugging and getting the baseline
    recog_epoch = [20, 40, 60, 80, 100]
    # Standard KLDiv ILM
    # for lrs in lr_list:
    #     for epochs in ep_list:
    #         lr_1, lr_2 = lrs
    #         ep1, ep2 = epochs
    #         lrs = [lr_1]*ep1
    #         train_exp(
    #             f"conformer_trafo_ilm_kldiv_const_lr_{lr_1}_ep100_fixEos_noSpecAug",
    #             config_11gb,
    #             from_scratch_training_kldiv,
    #             gpu_mem=11,
    #             config_updates={
    #                 "batch_size": 2400000,
    #                 "learning_rate": float(lrs[-1]),
    #                 "learning_rates": lrs,
    #                 "__num_epochs": 100,
    #                 "mask_eos_output": True,
    #                 "add_eos_to_blank": True,
    #                 "preload_from_files": {
    #                     "base": {
    #                         "init_for_train": True,
    #                         "ignore_missing": True,
    #                         "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
    #                     }
    #                 },
    #                 "mel_normalization_ted2": False,
    #                 "use_specaugment": False, # VERY IMPORTANT!!!
    #             },
    #             post_config_updates={
    #                 "cleanup_old_models": {"keep": recog_epoch},
    #                 "torch_dataloader_opts": { # otherwise it will break after every epoch
    #                     "num_workers": 0,
    #                 }
    #             },
    #             greedy_search = False,
    #         )

    for lrs in lr_list:
        for epochs in ep_list:
            lr_1, lr_2 = lrs
            ep1, ep2 = epochs
            lrs = [lr_1]*ep1
            train_exp(
                f"debug_rescoring_first_pass_hdf",
                config_11gb,
                from_scratch_training_kldiv_sample_batch,
                gpu_mem=11,
                config_updates={
                    "batch_size": 1200000,
                    "learning_rate": float(lrs[-1]),
                    "learning_rates": lrs,
                    "__num_epochs": 100,
                    "kldiv_sampling_weight": 0.5,
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
                    "use_specaugment": False, # VERY IMPORTANT!!!
                },
                post_config_updates={
                    "cleanup_old_models": {"keep": recog_epoch},
                    "torch_dataloader_opts": { # otherwise it will break after every epoch
                        "num_workers": 0,
                    }
                },
                greedy_search = False,
            )


    # -------- experiments with shorter epochs but higher learning rates ---------
    # to verify whether lower PPL is really not as good
    # # standard KL Div
    # lr = 1e-3
    # ep = 40
    # recog_epoch_short = [20, 40, 60, 80, 100]
    # # ground_truth_weights = [0.5, "average"]
    # ground_truth_weights = [0.5]
    # for weight in ground_truth_weights:
    #     train_exp(
    #         f"conformer_ilm_kldiv_sampling_weight_{weight}_lr_{lr}_ep_{ep}_fixEos_noSpecAug",
    #         config_11gb,
    #         from_scratch_training_kldiv_sample_batch,
    #         gpu_mem=11,
    #         config_updates={
    #             "batch_size": 1200000,
    #             "learning_rate": lr,
    #             "learning_rates": [lr]*ep,
    #             "__num_epochs": 100,
    #             "mask_eos_output": True,
    #             "add_eos_to_blank": True,
    #             "preload_from_files": {
    #                 "base": {
    #                     "init_for_train": True,
    #                     "ignore_missing": True,
    #                     "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
    #                 }
    #             },
    #             "mel_normalization_ted2": False,
    #             "kldiv_sampling_weight": weight,
    #             "use_specaugment": False, # VERY IMPORTANT!!!
    #         },
    #         post_config_updates={
    #             "cleanup_old_models": {"keep": recog_epoch_short},
    #             "torch_dataloader_opts": { # otherwise it will break after every epoch
    #                 "num_workers": 0,
    #             }
    #         },
    #         greedy_search = False,
    #     )



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
    train_def: TrainDef,
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
    greedy_search=False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    # from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.train import (
    #     train,
    # )
    from i6_experiments.users.yang.torch.luca_ctc.train import train
    from i6_experiments.users.yang.torch.luca_ctc.recog import recog_training_exp
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
        train_def=train_def,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
        mem_rqmt = mem_rqmt,
        disable_epoch_wise_filter=True,
        extra_hash="fix_eos_posterior",
    )
    # # greedy search: we won't use greedy search anyway
    # if greedy_search:
    #     from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
    #     recog_training_exp(
    #         prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    #     )
    # Luca's label sync search with LM and ILM scale

    # -------- compute some statistics related to the KL Div --------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev-other", "test-other"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "external_language_model": default_extern_lm_config,
    })
    forward_post_config = dict(
        torch_log_memory_usage=True,
        use_lovely_tensors=True,
    )
    # ILM stats buggy, don't do it for now
    # for dataset_key in dataset_keys:
    #     if dataset_key == "train": # not tested
    #         forward_dataset = task.train_dataset
    #     else:
    #         forward_dataset = task.eval_datasets[dataset_key]
    #     for epoch in model_with_checkpoint.fixed_epochs:
    #         checkpoint = model_with_checkpoint.get_epoch(epoch)
    #         stats_job = generic_forward_config.generic_forward_job(
    #             dataset=forward_dataset,
    #             model=checkpoint,
    #             forward_def=compute_kldiv.forward_compute_kldiv,
    #             forward_callback=compute_kldiv.forward_callback_wrapper,
    #             forward_extra_config=forward_extra_config,
    #             forward_post_config=forward_post_config,
    #             output_files=compute_kldiv.output_files,
    #             dataset_key=dataset_key,
    #             job_vis_name=f"Compute ILM stats job, {name}, epoch {epoch}, {dataset_key}",
    #         )
    #         out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
    #         stats_job.add_alias(prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
    #         tk.register_output(prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    recog_config_update = {
        'batch_size': 200000, # super slow
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "internal_language_model": default_ilm_config,
        "external_language_model": default_extern_lm_config, # this to load the external LM only in recog
    }
    # from i6_experiments.users.yang.torch.decoding.ctc_aed_joint_simple_version import model_recog
    # ctc label sync search
    # from i6_experiments.users.phan.recog.ctc_label_sync import model_recog_label_sync as model_recog
    # from i6_experiments.users.phan.recog.ctc_label_sync_v2 import model_recog
    # beam_sizes = [32] # to be consistent
    # # ilm_scales = [0.5, 0.6, 0.7, 0.8]
    # # lm_scales = [0.75, 0.85, 0.95, 1.05]
    # lm_scales = [0.85, 0.95, 1.05]
    # # ilm_scales = [0.5, 0.6, 0.7, 0.8, 0.9]
    # ilm_scales = [0.5, 0.6, 0.7, 0.8]
    # length_norm_scales = [1.0, 0.0]
    # prior_scales = [0.0, 0.1, 0.2, 0.4]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     if prior_scale > 0.1 and length_norm_scale != 0.0:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     recog_training_exp(
    #         prefix + f"_labelSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog,
    #         model_avg=False,
    #         exclude_epochs=[],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #     )


    # # --------------- time-synchronous search -----------------
    # from i6_experiments.users.phan.recog.ctc_time_sync_v2 import model_recog_time_sync
    # beam_sizes = [32] # to be consistent [16, 32]
    # length_norm_scales = [0.0] # never use 1.0!
    # lm_scales = [0.9, 1.0, 1.1, 1.2] # [0.9, 1.0, 1.1, 1.2, 1.3]
    # ilm_scales = [0.3, 0.4, 0.5, 0.6, 0.7]
    # prior_scales = [0.0, 0.2, 0.3]
    # # lm_scales = [1.1, 1.2, 1.3]
    # # ilm_scales = [0.6, 0.7, 0.8] # try this ???
    # # prior_scales = [0.0, 0.2, 0.3]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #         "batch_size": 600000,
    #     })
    #     recog_training_exp(
    #         prefix + f"_timeSync_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog_time_sync,
    #         model_avg=False,
    #         exclude_epochs=[],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #     )

    # # --------------- time-synchronous search recomb first -----------------
    # from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first import model_recog_time_sync_recomb_first
    # beam_sizes = [32] # to be consistent [16, 32]
    # length_norm_scales = [0.0] # never use 1.0!
    # lm_scales = [0.9, 1.0, 1.1] # [0.9, 1.0, 1.1, 1.2, 1.3]
    # ilm_scales = [0.3, 0.4, 0.5]
    # prior_scales = [0.2, 0.3]
    # # lm_scales = [1.1, 1.2, 1.3]
    # # ilm_scales = [0.6, 0.7, 0.8] # try this ???
    # # prior_scales = [0.0, 0.2, 0.3]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #         "batch_size": 600000,
    #     })
    #     recog_training_exp(
    #         prefix + f"_timeSyncRecombFirst_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first,
    #         model_avg=False,
    #         exclude_epochs=[],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #     )

    # # -------- check if extern LM PPL is calculated correctly --------
    # # Librispeech LM on ted2 dev/test data
    # from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    # # dataset_keys = ["dev-other", "test-other"]
    # ted2_task = _get_ted2_task()
    # dataset_keys = ["dev", "test"]
    # forward_extra_config = copy.deepcopy(config)
    # forward_extra_config.update({
    #     "batch_size": 4800000,
    #     "max_seqs": 200,
    #     "preload_from_files": {
    #         "01_lstm_lm_ted2": {
    #             "prefix": "language_model.",
    #             # "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
    #             "filename": "/u/zyang/setups/torch_checkpoints/lm/convert/tedlium/network.020.pt",
    #         },
    #     },
    #     # "external_language_model": default_extern_lm_config,
    #     # "external_language_model": default_tedlium2_extern_lm_hardcoded_layers_config,
    #     "external_language_model": default_tedlium2_extern_lm_config,
    #     "with_extern_lm": True,
    # })
    # forward_post_config = dict(
    #     torch_log_memory_usage=True,
    #     use_lovely_tensors=True,
    # )
    
    # for dataset_key in dataset_keys:
    #     if "kldiv_sampling_weight" in config:
    #         continue
    #     if dataset_key == "train": # not tested
    #         forward_dataset = ted2_task.train_dataset
    #     else:
    #         forward_dataset = ted2_task.eval_datasets[dataset_key]
    #     for epoch in [40]:
    #         checkpoint = model_with_checkpoint.get_epoch(epoch)
    #         stats_job = generic_forward_config.generic_forward_job(
    #             dataset=forward_dataset,
    #             model=checkpoint,
    #             forward_def=compute_kldiv.forward_compute_kldiv,
    #             forward_callback=compute_kldiv.forward_callback_wrapper,
    #             forward_extra_config=forward_extra_config,
    #             forward_post_config=forward_post_config,
    #             output_files=compute_kldiv.output_files,
    #             dataset_key=dataset_key,
    #             job_vis_name=f"Compute ILM stats job on tedlium2, {name}, epoch {epoch}, {dataset_key}",
    #         )
    #         out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
    #         stats_job.add_alias("lbs_cross_domain_ted2/" + prefix + "/ilm_stats_with_extern_lm_ppl_zijian_ckpt_fix2" + f"/{dataset_key}/{epoch}")
    #         tk.register_output("lbs_cross_domain_ted2/" + prefix + f"/ilm_stats_with_extern_lm_ppl_zijian_ckpt_fix2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # ------------------------ cross-domain tedlium2 (estimation on LBS, recognition on ted2) -------------------------
    ted2_prefix = "lbs_cross_domain_ted2/" + prefix
    ted2_task = _get_ted2_task()

    # -------- compute ted2 ILM dev, test PPL --------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev", "test"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "with_extern_lm": False,
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
        for epoch in model_with_checkpoint.fixed_epochs:
            checkpoint = model_with_checkpoint.get_epoch(epoch)
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
            # out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
            # stats_job.add_alias(ted2_prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
            # tk.register_output(ted2_prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)


    # ----------------- TED2 time sync recombination before pruning ----------------
    ted2_prefix = "lbs_cross_domain_ted2/" + prefix
    ted2_task = _get_ted2_task()
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    #from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2

    #---------- default ted2 recog config ----------
    ted2_recog_config_update = {
        'batch_size': 2400000,
        "preload_from_files": {
            "01_lstm_extern_lm": {
                "prefix": "language_model.",
                "filename": default_ted2_lstm_extern_lm_checkpoint,
            },
        },
        "internal_language_model": default_ilm_config,
        "external_language_model": default_tedlium2_extern_lm_config, # this to load the external LM only in recog
    }

    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [1.6, 1.7, 1.8]
    # ilm_scales = [0.5, 0.6, 0.7, 0.8] 
    # prior_scales = [0.4, 0.5, 0.6]
    # lm_scales = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # ilm_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    # prior_scales = [0.2, 0.3, 0.4, 0.5, 0.6]
    lm_scales = [1.4]
    ilm_scales = [0.9]
    # prior_scales = [0.0, 0.1, 0.2, 0.3, 0.4]
    prior_scales = [0.0]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        # config_weight = config.get("kldiv_sampling_weight", None)
        # if config_weight == 0.5:
        #     exclude_epochs = [40]
        # elif config_weight == "average":
        #     exclude_epochs = [20]
        # elif config_weight == None:
        #     exclude_epochs = [40]
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "length_norm_scale": length_norm_scale,
            "prior_scale": prior_scale,
            "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "ctc_log_prior": False,
            "lm_skip": True,
        }
        ted2_recog_config_update_extra = copy.deepcopy(ted2_recog_config_update)
        ted2_recog_config_update_extra.update({
            "search_args": search_args,
            "batch_size": 4800000,
        })
        from i6_experiments.users.phan.rescoring.first_pass import recog_first_pass_hdf
        # epoch = list(model_with_checkpoint.fixed_epochs)[0]
        # checkpoint = model_with_checkpoint.get_epoch(epoch)
        # outputs = recog_first_pass_hdf(
        #     task=ted2_task,
        #     model=checkpoint,
        #     recog_def=model_recog_time_sync_recomb_first_v2,
        #     config=ted2_recog_config_update_extra,
        #     dev_sets=['dev']
        # )
        # tk.register_output(prefix+'/debug_out_hyps.hdf', outputs['dev'].out_hyps)
        # tk.register_output(prefix + '/debug_out_lens.hdf', outputs['dev'].out_lens)
        # tk.register_output(prefix + '/debug_out_scores.hdf', outputs['dev'].out_scores)
        # tk.register_output(prefix + '/debug_out_packed_batch_sizes.hdf', outputs['dev'].out_packed_batch_sizes)

        #for debugging:
        ted2_recog_config_update_extra['search_args']['beam_size'] = 5
        checkpoint = model_with_checkpoint.get_epoch(20)
        outputs_v2 = recog_first_pass_hdf(
            task=ted2_task,
            model=checkpoint,
            recog_def=model_recog_time_sync_recomb_first_v2,
            config=ted2_recog_config_update_extra,
            dev_sets=['dev'],
            hdf_output='v2'
        )
        tk.register_output(prefix + '/debug_out_hyps_v2.hdf', outputs_v2['dev'].out_hyps)
        from i6_experiments.users.phan.rescoring.first_pass import debug_hdf_reading
        debug_output = debug_hdf_reading(path=outputs_v2['dev'].out_hyps, model=checkpoint, beam_size=5)
        tk.register_output(prefix +'/debug_read_hdf.txt', debug_output)


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

_ted2_task = None

def _get_ted2_task():
    global _ted2_task
    if _ted2_task:
        return _ted2_task
    #from i6_experiments.users.phan.datasets.librispeech_tedlium2 import get_tedlium2_task_libri_bpe10k_raw
    from i6_experiments.users.phan.datasets.librispeech_tedlium2 import get_tedlium2_task_libri_bpe10k_raw
    _ted2_task = get_tedlium2_task_libri_bpe10k_raw(with_eos_postfix=False)
    return _ted2_task


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