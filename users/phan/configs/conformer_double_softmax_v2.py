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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_double_softmax
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config

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
config_24gb.update({"internal_language_model": default_ilm_config})


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    # lr_list = [(5e-6, 1e-7)]
    # ep_list = [(40, 40)]
    # recog_epoch = [20, 40, 80] # [20, 40, 80, 100]
    # # ---------- double softmax experiments ---------
    # am_scales = [1.0] 
    # rel_scales = [0.2, 0.35] # just start with this
    # for lrs, epochs, am_scale, rel_scale in itertools.product(lr_list, ep_list, am_scales, rel_scales):
    #     lm_scale = round(am_scale*rel_scale, 2)
    #     lr_1, lr_2 = lrs
    #     ep1, ep2 = epochs
    #     lrs = const_linear_learning_rates(lr_1,lr_2,ep1,ep2)
    #     train_exp( 
    #         f"conformer_double_softmax_v2_am-{am_scale}_lm-{lm_scale}_lr1_{lr_1}_lr2_{lr_2}_ep1_{ep1}_ep2_{ep2}",
    #         config_24gb,
    #         from_scratch_training_double_softmax,
    #         gpu_mem=24,
    #         config_updates={
    #             "batch_size": 1000000,
    #             "learning_rate": lrs[-1],
    #             "learning_rates": lrs,
    #             "__num_epochs": ep1+ep2,
    #             "mask_eos_output": True,
    #             "add_eos_to_blank": True,
    #             "preload_from_files": {
    #                 "base": {
    #                     "init_for_train": True,
    #                     "ignore_missing": True,
    #                     "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
    #                 },
    #                 "train_extern_lm": { # import transcription lm
    #                     "init_for_train": True,
    #                     "prefix": "ilm.",
    #                     "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
    #                 }
    #             },
    #             "mel_normalization_ted2": False,
    #             "am_scale": am_scale,
    #             "lm_scale": lm_scale,
    #             "freeze_encoder": False,
    #             "freeze_ilm": True,
    #             "hash_override": 1,
    #         },
    #         time_rqmt=100,
    #         post_config_updates={
    #             "cleanup_old_models": {"keep": recog_epoch},
    #             "torch_dataloader_opts": { # otherwise it will break after every epoch
    #                 "num_workers": 0,
    #             }
    #         },
    #         greedy_search = False,
    #     )

    lr_list = [1e-5]
    ep_list = [40]
    recog_epoch = [20, 40] # [20, 40, 80, 100]
    # ---------- double softmax experiments ---------
    am_scales = [1.0] 
    rel_scales = [0.35] # just start with this
    for lr, epoch, am_scale, rel_scale in itertools.product(lr_list, ep_list, am_scales, rel_scales):
        lm_scale = round(am_scale*rel_scale, 2)
        train_exp( 
            f"conformer_double_softmax_v2_am-{am_scale}_lm-{lm_scale}_lr1_{lr}_ep_{epoch}",
            config_24gb,
            from_scratch_training_double_softmax,
            gpu_mem=24,
            config_updates={
                "batch_size": 1000000,
                "learning_rate": lr,
                "learning_rates": [lr] * epoch,
                "__num_epochs": epoch,
                "mask_eos_output": True,
                "add_eos_to_blank": True,
                "preload_from_files": {
                    "base": {
                        "init_for_train": True,
                        "ignore_missing": True,
                        "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
                    },
                    "train_extern_lm": { # import transcription lm
                        "init_for_train": True,
                        "prefix": "ilm.",
                        "filename": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/training/ReturnnTrainingJob.3uMurj5onrWB/output/models/epoch.005.pt",
                    }
                },
                "mel_normalization_ted2": False,
                "am_scale": am_scale,
                "lm_scale": lm_scale,
                "freeze_encoder": False,
                "freeze_ilm": True,
                "hash_override": 1,
            },
            time_rqmt=100,
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

    recog_config_update = {
        'batch_size': 600000, # super slow
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "internal_language_model": default_ilm_config,
        "external_language_model": default_extern_lm_config, # this to load the external LM only in recog
    }

    # --------------- time-synchronous search recomb first -----------------
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32] # to be consistent
    lm_scales = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ilm_scales = [0.0]
    length_norm_scales = [0.0]
    prior_scales = [0.0, 0.3, 0.4, 0.5, 0.6]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "prior_scale": prior_scale,
            "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "ctc_log_prior": False,
        }
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "search_args": search_args,
        })
        recompute_prior = True if prior_scale > 0.0 else False
        prior_config = None
        if recompute_prior:
            prior_config = copy.deepcopy(recog_config_update_extra)
            prior_config.pop("external_language_model", None)
            prior_config.pop("search_args", None)
            prior_config.pop("hash_override", None)
            # prior_config.pop("internal_language_model", None)
            prior_config["batch_size"] = int(25600000)
            prior_config["batching"] = "sorted_reverse"
        recog_training_exp(
            prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            task,
            model_with_checkpoint,
            search_config=recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[],
            train_exp_name=name,
            dev_sets=["dev-other", "test-other"],
            recompute_prior=recompute_prior,
            prior_config=prior_config,
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
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k", main_key="train")
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
