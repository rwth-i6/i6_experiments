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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_lfmmi_context_1
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, default_bigram_config, \
    default_tedlium2_extern_lm_hardcoded_layers_config, default_tedlium2_extern_lm_config

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


config_24gb = copy.deepcopy(config_24gb)
config_24gb.pop("dynamic_learning_rate", None)
config_24gb.pop("learning_rate_piecewise_steps", None)
config_24gb.pop("learning_rate_piecewise_values", None)
config_24gb.pop("learning_rate_invsqrt_norm", None)
config_24gb.pop("learning_rate_warmup_steps", None)
config_24gb.pop("specaugment_steps", None)
config_24gb.pop("torch_amp", None)

# set the ILM hyperparams for all experiments
# hyperparams referrence /u/michel/setups/language_modelling/librispeech/neurallm/decoder_sized_transcripts_only_newrun
config_24gb.update({"internal_language_model": default_bigram_config})


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    # back to 1e-5
    lr_list = [1e-5, 1e-6]
    ep_list = [20]
    recog_epoch = [2, 4, 6, 8, 12, 16, 20] # [20, 40, 80, 100]
    # ---------- lf mmi experiments ---------
    am_scales = [1.0] # 1.5, 0.1 will diverge
    rel_scales = [0.1, 0.2, 0.3] # 0.35 from Willi's paper ????
    top_ks = [160]
    for lr, epoch, am_scale, rel_scale, top_k in itertools.product(lr_list, ep_list, am_scales, rel_scales, top_ks):
        lm_scale = round(am_scale*rel_scale, 3)
        train_exp( 
            f"conformer_lfmmi_context1_strict-topk_noSpecAug_am-{am_scale}_lm-{lm_scale}_topk-{top_k}_lr_{lr}_ep_{epoch}",
            config_24gb,
            from_scratch_training_lfmmi_context_1,
            gpu_mem=24,
            config_updates={
                "batch_size": 1000000,
                "learning_rate": lr,
                "learning_rates": [lr]*epoch,
                "__num_epochs": 20, # 100
                "mask_eos_output": True,
                "add_eos_to_blank": True,
                "preload_from_files": {
                    "base": {
                        "init_for_train": True,
                        "ignore_missing": True,
                        "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
                    },
                    "train_extern_lm": { # import the bigram
                        "init_for_train": True,
                        "prefix": "ilm.",
                        "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.r6yAbhM3T751/output/models/epoch.022.pt",
                    }
                },
                "mel_normalization_ted2": False,
                "am_scale": am_scale,
                "lm_scale": lm_scale,
                "top_k": top_k,
                "using_strict_topk": True,
                "freeze_encoder": False,
                "freeze_ilm": True,
                "use_specaugment": False, # VERY IMPORTANT!!!
            },
            post_config_updates={
                "cleanup_old_models": {"keep": recog_epoch},
                "torch_dataloader_opts": { # otherwise it will break after every epoch
                    "num_workers": 0,
                }
            },
            time_rqmt=50,
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
    )
    # # greedy search: we won't use greedy search anyway
    # if greedy_search:
    #     from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
    #     recog_training_exp(
    #         prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    #     )
    # Luca's label sync search with LM and ILM scale

    # --------------- First we tune using the lstm, and then take the best epoch and use trafo ---------

    recog_config_update_lstm_lm = {
        'batch_size': 2200000, # super fast???
        "preload_from_files": {
            "01_lstm_lm": {
                "ignore_params_prefixes": [
                    "encoder",
                    "ctc",
                    "enc_ctx",
                    "inv_fertility",
                    "target_embed",
                    "s",
                    "weight_feedback",
                    "s_transformed",
                    "energy",
                    "readout_in",
                    "output_prob"
                ],
                "var_name_mapping": {
                    'language_model.input_bias': 'lstm_lm.input_bias',
                    'language_model.input.weight': 'lstm_lm.input.weight',
                    'language_model.lstm_0.ff_weight': 'lstm_lm.lstm_0.ff_weight',
                    'language_model.lstm_0.rec_weight': 'lstm_lm.lstm_0.rec_weight',
                    'language_model.lstm_0.bias': 'lstm_lm.lstm_0.bias',
                    'language_model.lstm_1.ff_weight': 'lstm_lm.lstm_1.ff_weight',
                    'language_model.lstm_1.rec_weight': 'lstm_lm.lstm_1.rec_weight',
                    'language_model.lstm_1.bias': 'lstm_lm.lstm_1.bias',
                    'language_model.lstm_2.ff_weight': 'lstm_lm.lstm_2.ff_weight',
                    'language_model.lstm_2.rec_weight': 'lstm_lm.lstm_2.rec_weight',
                    'language_model.lstm_2.bias': 'lstm_lm.lstm_2.bias',
                    'language_model.lstm_3.ff_weight': 'lstm_lm.lstm_3.ff_weight',
                    'language_model.lstm_3.rec_weight': 'lstm_lm.lstm_3.rec_weight',
                    'language_model.lstm_3.bias': 'lstm_lm.lstm_3.bias',
                    'language_model.output.weight': 'lstm_lm.output.weight',
                    'language_model.output.bias': 'lstm_lm.output.bias',
                },
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/full_w_lm_import_2023_10_18/average.pt",
            },
        },
        "internal_language_model": default_bigram_config,
        "external_language_model": default_tedlium2_extern_lm_hardcoded_layers_config,
    }

    recog_config_update_trafo_lm = {
        'batch_size': 600000, # super slow!!!
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "internal_language_model": default_bigram_config,
        "external_language_model": default_extern_lm_config, # this to load the external LM only in recog
    }

    # # --------------- decoding with trafo LM from LibriSpeech -----------------
    # from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    # beam_sizes = [32] # to be consistent
    # # lm_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # ilm_scales = [0.0]
    # # length_norm_scales = [0.0]
    # # prior_scales = [0.0, 0.3, 0.4, 0.5]

    # # lm_scales = [0.7, 0.8, 0.9, 1.0]
    # # ilm_scales = [0.0]
    # # length_norm_scales = [0.0]
    # # prior_scales = [0.3, 0.4, 0.5]

    # lm_scales = [1.0, 1.1, 1.2]
    # ilm_scales = [0.0]
    # length_norm_scales = [0.0]
    # prior_scales = [0.4, 0.5, 0.6]

    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     if config["learning_rate"] != 1e-6:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update_trafo_lm)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     recompute_prior = True if prior_scale > 0.0 else False
    #     prior_config = None
    #     if recompute_prior:
    #         prior_config = copy.deepcopy(recog_config_update_extra)
    #         prior_config.pop("external_language_model", None)
    #         prior_config.pop("preload_from_files", None)
    #         prior_config.pop("search_args", None)
    #         prior_config.pop("hash_override", None)
    #         # prior_config.pop("internal_language_model", None)
    #         prior_config["batch_size"] = int(38400000)
    #         prior_config["batching"] = "sorted_reverse"
    #         prior_config["hash_override"] = "18/11/2024"
    #     recog_training_exp(
    #         prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         model_avg=False,
    #         exclude_epochs=[6, 8, 12, 16, 20],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #         recompute_prior=recompute_prior,
    #         prior_config=prior_config,
    #         search_rqmt={"time": 5},
    #     )

    # # no prior -> lower LM scale
    # lm_scales = [0.4, 0.5, 0.6, 0.7]
    # ilm_scales = [0.0]
    # length_norm_scales = [0.0]
    # prior_scales = [0.0]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if config["learning_rate"] != 1e-6:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_normalization_exponent": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update_trafo_lm)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     recompute_prior = True if prior_scale > 0.0 else False
    #     prior_config = None
    #     if recompute_prior:
    #         prior_config = copy.deepcopy(recog_config_update_extra)
    #         prior_config.pop("external_language_model", None)
    #         prior_config.pop("preload_from_files", None)
    #         prior_config.pop("search_args", None)
    #         prior_config.pop("hash_override", None)
    #         # prior_config.pop("internal_language_model", None)
    #         prior_config["batch_size"] = int(38400000)
    #         prior_config["batching"] = "sorted_reverse"
    #         prior_config["hash_override"] = "18/11/2024"
    #     recog_training_exp(
    #         prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         model_avg=False,
    #         exclude_epochs=[6, 8, 12, 16, 20],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #         recompute_prior=recompute_prior,
    #         prior_config=prior_config,
    #         search_rqmt={"time": 5},
    #     )

    # ----------------- cross-domain tedlium2 recognition ---------------------
    # ----------------- time sync recombination before pruning ----------------
    ted2_prefix = "lbs_cross_domain_ted2/" + prefix
    ted2_task = _get_ted2_task()
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
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
        "internal_language_model": default_bigram_config,
        "external_language_model": default_tedlium2_extern_lm_config, # this to load the external LM only in recog
    }

    # with prior
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    ilm_scales = [0.0]
    # lm_scales = [1.0, 1.1, 1.2, 1.3]
    # prior_scales = [0.5, 0.6, 0.7, 0.8]
    lm_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    prior_scales = [0.0, 0.3, 0.4, 0.5, 0.6]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):
        if prior_scale >= lm_scale:
            continue       
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "length_norm_scale": length_norm_scale,
            "prior_scale": prior_scale,
            "ctc_log_prior": False,
            "lm_skip": True,
        }
        ted2_recog_config_update_extra = copy.deepcopy(ted2_recog_config_update)
        ted2_recog_config_update_extra.update({
            "search_args": search_args,
        })
        recompute_prior = True if prior_scale > 0.0 else False
        prior_config = None
        if recompute_prior:
            prior_config = copy.deepcopy(ted2_recog_config_update_extra)
            prior_config.pop("external_language_model", None)
            prior_config.pop("preload_from_files", None)
            prior_config.pop("search_args", None)
            prior_config.pop("hash_override", None)
            # prior_config.pop("internal_language_model", None)
            prior_config["batch_size"] = int(25600000)
            prior_config["batching"] = "sorted_reverse"
            prior_config["hash_override"] = "30/11/2024"
        recog_training_exp(
            ted2_prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            ted2_task,
            model_with_checkpoint,
            search_config=ted2_recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[2, 4, 6, 8, 16, 20],
            train_exp_name=name,
            dev_sets=["dev", "test"],
            # dev_sets=["dev"],
            recompute_prior=recompute_prior,
            prior_config=prior_config,
            prior_task=task,
        )

    # # without prior
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # # lm_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # lm_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    # ilm_scales = [0.0] 
    # prior_scales = [0.0]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):
    #     if config["am_scale"] != 1.2 or config["lm_scale"] != 0.12:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale,
    #         "prior_scale": prior_scale,
    #         "ctc_log_prior": False,
    #         "lm_skip": True,
    #     }
    #     ted2_recog_config_update_extra = copy.deepcopy(ted2_recog_config_update)
    #     ted2_recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     recompute_prior = True if prior_scale > 0.0 else False
    #     prior_config = None
    #     if recompute_prior:
    #         prior_config = copy.deepcopy(recog_config_update_extra)
    #         prior_config.pop("external_language_model", None)
    #         prior_config.pop("preload_from_files", None)
    #         prior_config.pop("search_args", None)
    #         prior_config.pop("hash_override", None)
    #         # prior_config.pop("internal_language_model", None)
    #         prior_config["batch_size"] = int(25600000)
    #         prior_config["batching"] = "sorted_reverse"
    #         prior_config["hash_override"] = "30/11/2024"
    #     recog_training_exp(
    #         ted2_prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         ted2_task,
    #         model_with_checkpoint,
    #         search_config=ted2_recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         model_avg=False,
    #         exclude_epochs=[2, 4, 6, 8, 16, 20],
    #         train_exp_name=name,
    #         # dev_sets=["dev", "test"],
    #         dev_sets=["dev", "test"],
    #         recompute_prior=recompute_prior,
    #         prior_config=prior_config,
    #     )

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
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k", main_key="train")
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
