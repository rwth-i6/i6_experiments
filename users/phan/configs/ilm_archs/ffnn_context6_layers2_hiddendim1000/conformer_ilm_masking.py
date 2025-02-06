"""
CTC ILM masking method
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


from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv, \
    from_scratch_training_kldiv_sample_batch, from_scratch_training_kldiv_masking
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, default_tedlium2_extern_lm_config


if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
    from i6_experiments.users.zeyer.model_with_checkpoints import (
        ModelWithCheckpoints,
        ModelWithCheckpoint,
    )


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
from i6_experiments.users.phan.configs.ilm_archs.ffnn_context6_layers2_hiddendim1000.ilm_config import default_ilm_config as \
    default_ffnn_context6_layers2_hiddendim1000_ilm_config
config_11gb.update({"internal_language_model": default_ffnn_context6_layers2_hiddendim1000_ilm_config})

def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    lr_list = [1e-3]
    ep_list = [100]
    recog_epoch = [20, 40, 60, 80, 100]
    masking_rates = [0.4]
    for lr, masking_rate in zip(lr_list, masking_rates):
        for epoch in ep_list:
            lrs = [lr]*epoch
            train_exp( 
                f"conformer_ilm_kldiv_masking_maskRate_{masking_rate}_lr_{lr}_ep_{epoch}",
                config_11gb,
                from_scratch_training_kldiv_masking,
                gpu_mem=11,
                config_updates={
                    "batch_size": 2400000,
                    "learning_rate": float(lrs[-1]),
                    # "learning_rates": lrs,
                    "__num_epochs": epoch,
                    "mask_eos_output": True,
                    "add_eos_to_blank": True,
                    "preload_from_files": {
                        "base": {
                            "init_for_train": True,
                            "ignore_missing": True,
                            "filename": "/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt",
                        }
                    },
                    "input_masking_rate": masking_rate,
                    "mel_normalization_ted2": False,
                    "load_vocab": "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
                    "use_specaugment": False, # VERY IMPORTANT!!!
                },
                post_config_updates={
                    "cleanup_old_models": {"keep": recog_epoch},
                    "torch_dataloader_opts": { # otherwise it will break after every epoch
                        "num_workers": 0,
                    }
                },
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
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.yang.torch.luca_ctc.train import train
    from i6_experiments.users.yang.torch.luca_ctc.recog import recog_training_exp

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
        disable_epoch_wise_filter=False,
        use_multiproc_dataset=True,
        train_dataset_key="train-other-960",
    )

    # -------- compute some ILM statistics --------
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
    
    for dataset_key in dataset_keys:
        if dataset_key == "train": # not tested
            forward_dataset = task.train_dataset
        else:
            forward_dataset = task.eval_datasets[dataset_key]
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
                job_vis_name=f"Compute ILM stats job, {name}, epoch {epoch}, {dataset_key}",
            )
            out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
            stats_job.add_alias(prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
            tk.register_output(prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    recog_config_update = {
        'batch_size': 1800000, # super slow
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "internal_language_model": default_ffnn_context6_layers2_hiddendim1000_ilm_config,
        "external_language_model": default_extern_lm_config, # this to load the external LM only in recog
    }

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
        #"with_extern_lm": False,
        "with_extern_lm": True,
        "preload_from_files": {
            "01_lstm_lm_ted2": {
                "prefix": "language_model.",
                "filename": "/work/asr4/zyang/torch_checkpoints/lm/convert/tedlium/network.020.pt",
            },
        },
        "external_language_model": default_tedlium2_extern_lm_config,
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
            out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
            stats_job.add_alias(ted2_prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
            tk.register_output(ted2_prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # # --------------- time-synchronous recombination first search (fixed) -----------------
    # from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    # beam_sizes = [32]
    # lm_scales = [0.8, 0.9, 1.0, 1.1]
    # ilm_scales = [0.4, 0.5, 0.6, 0.7]
    # length_norm_scales = [0.0] # never use 1.0!
    # prior_scales = [0.0, 0.1, 0.2]
    # for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
    #     if ilm_scale >= lm_scale:
    #         continue
    #     if config["input_masking_rate"] not in [0.2, 0.4, 0.6]:
    #         continue
    #     search_args = {
    #         "beam_size": beam_size,
    #         "lm_scale": lm_scale,
    #         "ilm_scale": ilm_scale,
    #         "length_norm_scale": length_norm_scale, # by default len norm
    #         "prior_scale": prior_scale,
    #         "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
    #         "ctc_log_prior": False,
    #         "lm_skip": True,
    #     }
    #     recog_config_update_extra = copy.deepcopy(recog_config_update)
    #     recog_config_update_extra.update({
    #         "search_args": search_args,
    #         "batch_size": 600000,
    #     })
    #     recog_training_exp(
    #         prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
    #         task,
    #         model_with_checkpoint,
    #         search_config=recog_config_update_extra,
    #         recog_def=model_recog_time_sync_recomb_first_v2,
    #         model_avg=False,
    #         exclude_epochs=[20],
    #         train_exp_name=name,
    #         dev_sets=["dev-other", "test-other"],
    #     )


    # ----------------- TED2 time sync recombination before pruning ----------------
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
        "internal_language_model": default_ffnn_context6_layers2_hiddendim1000_ilm_config,
        "external_language_model": default_tedlium2_extern_lm_config, # this to load the external LM only in recog
    }

    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [1.3, 1.4, 1.5, 1.6, 1.7, 1.8]
    ilm_scales = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4] 
    # prior_scales = [0.0, 0.2, 0.4]
    prior_scales = [0.0]
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
            ted2_prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            ted2_task,
            model_with_checkpoint,
            search_config=ted2_recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[],
            train_exp_name=name,
            dev_sets=["dev", "test"],
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
    # from i6_experiments.users.phan.alignment.convert import get_gmm_pseudo_word_alignments
    alignments = {}
    for dataset_key in ["train-other-960"]:
        alignments[dataset_key] = ["/work/asr3/zyang/share/mnphan/alignment_data/lbs960/gmm_pseudo_word_alignments.hdf"]
    #_ls_task = get_librispeech_task_bpe10k_raw(with_eos_postfix=True) luca's dataloading
    _ls_task = get_librispeech_task_raw_v2(vocab="bpe10k", alignments=alignments, train_epoch_wise_filter=None)
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

