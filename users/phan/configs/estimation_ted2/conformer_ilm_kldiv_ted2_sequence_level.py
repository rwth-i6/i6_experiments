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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, \
    ilm_kldiv_sequence_level
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, \
    default_tedlium2_extern_lm_config, default_tedlium2_extern_lm_hardcoded_layers_config

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
# hyperparams referrence /u/michel/setups/language_modelling/librispeech/neurallm/decoder_sized_transcripts_only_newrun
config_11gb.update({"internal_language_model": default_ilm_config})


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""

    # 
    lr = 1e-3
    ep = 16
    recog_epoch_short = [4, 8, 12, 16]
    ground_truth_weights = [1.0, 0.5, "average"]
    for weight in ground_truth_weights:
        if weight == 1.0:
            batch_size = 2400000
        else:
            batch_size = 1800000
        train_exp( 
            f"conformer_ilm_kldiv_ted2_sequence_level_weight_{weight}_lr_{lr}_ep_{ep}",
            config_11gb,
            ilm_kldiv_sequence_level,
            gpu_mem=24,
            config_updates={
                "batch_size": batch_size,
                "learning_rate": lr,
                "learning_rates": [lr]*ep,
                "__num_epochs": ep,
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
                "sequence_sampling_weight": weight,
                "use_specaugment": False, # VERY IMPORTANT!!!
            },
            post_config_updates={
                "cleanup_old_models": {"keep": recog_epoch_short},
                "torch_dataloader_opts": { # otherwise it will break after every epoch
                    "num_workers": 0,
                }
            },
            greedy_search = False,
            time_rqmt=100,
        )



_sis_prefix: Optional[str] = None

def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

        prefix_name = get_setup_prefix_for_module(__name__)
    global _sis_prefix
    _sis_prefix = prefix_name



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
    ted2_task = _get_ted2_task()
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
        task=ted2_task,
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
    )


    # ----------------- TED2 time sync recombination before pruning ----------------
    ted2_task = _get_ted2_task()
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2

    ted2_task = _get_ted2_task()

     # ----------------- TED2 time sync recombination before pruning ----------------
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2

    #---------- default ted2 recog config ----------
    ted2_recog_config_update = {
        'batch_size': 1800000,
        "preload_from_files": {
            "01_lstm_extern_lm": {
                "prefix": "language_model.",
                "filename": default_ted2_lstm_extern_lm_checkpoint,
            },
        },
        "internal_language_model": default_ilm_config,
        "external_language_model": default_tedlium2_extern_lm_config, # this to load the external LM only in recog
    }

    # with prior
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [1.6, 1.7, 1.8]
    # ilm_scales = [0.5, 0.6, 0.7, 0.8] 
    # prior_scales = [0.4, 0.5, 0.6]
    # lm_scales = [1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
    # ilm_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 
    # prior_scales = [0.2, 0.3, 0.4, 0.5, 0.6]
    # lm_scales = [1.4, 1.5, 1.6, 1.7]
    # ilm_scales = [0.6, 0.7, 0.8, 0.9, 1.0] 
    # prior_scales = [0.0, 0.2, 0.4, 0.6]
    # lm_scales = [1.3, 1.4, 1.5, 1.6]
    # ilm_scales = [0.5, 0.6, 0.7] 
    # prior_scales = [0.3, 0.4, 0.5]
    # lm_scales = [1.3, 1.4, 1.5]
    # ilm_scales = [0.2, 0.3, 0.4] 
    # prior_scales = [0.3, 0.4, 0.5]
    lm_scales = [1.0, 1.1, 1.2]
    ilm_scales = [0.1, 0.2, 0.3] 
    prior_scales = [0.5, 0.6, 0.7]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
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
        })
        recog_training_exp(
            prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            ted2_task,
            model_with_checkpoint,
            search_config=ted2_recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[8, 12, 16],
            train_exp_name=name,
            dev_sets=["dev", "test"],
        )

    # # no prior
    # beam_sizes = [32]
    # length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    # lm_scales = [0.6, 0.8, 1.0, 1.2, 1.4]
    # ilm_scales = [0.2, 0.4, 0.6]
    # # lm_scales = [0.6, 0.7, 0.8, 1.0, 1.1, 1.2]
    # # ilm_scales = [0.1, 0.2, 0.3, 0.4]  
    # prior_scales = [0.0]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale,
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #         "lm_skip": True,
    #     }
    #     ted2_recog_config_update_extra = copy.deepcopy(ted2_recog_config_update)
    #     ted2_recog_config_update_extra.update({
    #         "search_args": search_args,
    #     })
    #     recog_training_exp(
    #         prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         ted2_task,
    #         model_with_checkpoint,
    #         search_config=ted2_recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         model_avg=False,
    #         exclude_epochs=[8, 12, 16],
    #         train_exp_name=name,
    #         dev_sets=["dev", "test"],
    #     )

    return model_with_checkpoint

_ted2_task = None

def _get_ted2_task():
    global _ted2_task
    if _ted2_task:
        return _ted2_task
    from i6_experiments.users.phan.datasets.librispeech_tedlium2 import get_tedlium2_task_libri_bpe10k_raw
    _ted2_task = get_tedlium2_task_libri_bpe10k_raw(with_eos_postfix=False)
    return _ted2_task


py = sis_run_with_prefix  # if run directly via `sis m ...`


