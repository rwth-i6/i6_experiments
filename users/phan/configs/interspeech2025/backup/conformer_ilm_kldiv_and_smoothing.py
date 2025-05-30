"""
Config for training and recognition for LSTM ILM with the KLDiv and smoothing method
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Sequence, List

import copy
import copy
import itertools

from sisyphus import tk

from i6_experiments.users.yang.torch.luca_ctc.configs import *
from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, from_scratch_training_kldiv, from_scratch_training_kldiv_sample_batch
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_extern_lm_config, \
    default_tedlium2_extern_lm_config

from i6_experiments.users.zeyer.model_interfaces import TrainDef
from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints

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
    lr_list = [(1e-3, 1e-3)] # [(1e-5, 1e-7), (1e-3, 1e-3)]
    ep_list = [(40, 60)]
    # recog_epoch = [1] + list(range(20, 120, 20)) # 1 is mostly for debugging and getting the baseline
    recog_epoch = [20, 40, 60, 80, 100]
    # Standard KLDiv ILM
    for lrs in lr_list:
        for epochs in ep_list:
            lr_1, _ = lrs
            ep1, _ = epochs
            lrs = [lr_1]*ep1
            train_exp( 
                f"conformer_ilm_kldiv_lr_{lr_1}_ep_{ep1}",
                config_11gb,
                from_scratch_training_kldiv,
                gpu_mem=11,
                config_updates={
                    "batch_size": 2400000,
                    "learning_rate": float(lrs[-1]),
                    "learning_rates": lrs,
                    "__num_epochs": 100,
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
    # standard KL Div
    lr = 1e-3
    ep = 40
    recog_epoch_short = [20, 40, 60, 80, 100]
    # ground_truth_weights = [0.5, "average"]
    ground_truth_weights = [0.5]
    for weight in ground_truth_weights:
        train_exp( 
            f"conformer_ilm_kldiv_smoothing_weight_{weight}_lr_{lr}_ep_{ep}",
            config_11gb,
            from_scratch_training_kldiv_sample_batch,
            gpu_mem=11,
            config_updates={
                "batch_size": 1200000,
                "learning_rate": lr,
                "learning_rates": [lr]*ep,
                "__num_epochs": 100,
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
                "kldiv_sampling_weight": weight,
                "use_specaugment": False, # VERY IMPORTANT!!!
            },
            post_config_updates={
                "cleanup_old_models": {"keep": recog_epoch_short},
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
        disable_epoch_wise_filter=True,
        extra_hash="fix_eos_posterior",
    )

    # -------- compute some ILM statistics --------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev-other", "test-other", "dev-clean"]
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
        'batch_size': 600000,
        "preload_from_files": {
            "01_trafo_lm": {
                "prefix": "language_model.",
                "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
            },
        },
        "internal_language_model": default_ilm_config,
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
            out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
            stats_job.add_alias(ted2_prefix + "/ilm_stats_v2" + f"/{dataset_key}/{epoch}")
            tk.register_output(ted2_prefix + f"/ilm_stats_v2/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # --------------- LibriSpeech recognitions with frame-level prior -----------------
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32] 
    length_norm_scales = [0.0] # never use 1.0!
    lm_scales = [1.0]
    ilm_scales = [0.4, 0.5]
    prior_scales = [0.3]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        config_weight = config.get("kldiv_sampling_weight", None)
        if config_weight is None and (lm_scale, ilm_scale, prior_scale) != (1.0, 0.5, 0.3):
            continue
        elif config_weight == 0.5 and (lm_scale, ilm_scale, prior_scale) != (1.0, 0.4, 0.3):
            continue
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "length_norm_scale": length_norm_scale, # by default len norm
            "prior_scale": prior_scale,
            # this is the correct prior
            "prior_file": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/forward/ReturnnForwardJobV2.Wug49TgveO2b/output/prior.txt",
            "ctc_log_prior": False,
        }
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "search_args": search_args,
            "batch_size": 600000,
        })
        recog_training_exp(
            prefix + f"_timeSyncRecombFirstV2_correctprior_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            task,
            model_with_checkpoint,
            search_config=recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[60, 80, 100],
            train_exp_name=name,
            dev_sets=["dev-other", "test-other", "dev-clean", "test-clean"],
        )

    # --------------- LibriSpeech recognitions without frame level priors -----------------
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [0.9]
    ilm_scales = [0.4, 0.5]
    prior_scales = [0.0]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "length_norm_scale": length_norm_scale, # by default len norm
            "prior_scale": prior_scale,
            "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "ctc_log_prior": False,
        }
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "search_args": search_args,
            "batch_size": 600000,
        })
        recog_training_exp(
            prefix + f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}",
            task,
            model_with_checkpoint,
            search_config=recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[60, 80, 100],
            train_exp_name=name,
            dev_sets=["dev-other", "test-other", "dev-clean", "test-clean"],
        )


    # ----------------- TED2 recognitions ----------------
    ted2_prefix = "lbs_cross_domain_ted2/" + prefix
    ted2_task = _get_ted2_task()
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

    # ------------- Ted2 recognitions without frame-level prior --------------
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [1.6, 1.7]
    ilm_scales = [1.1, 1.4] 
    prior_scales = [0.0]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        config_weight = config.get("kldiv_sampling_weight", None)
        exclude_epochs = [40, 60, 80, 100]
        if config_weight == 0.5:
            if (lm_scale, ilm_scale) != (1.7, 1.4):
                continue
        elif config_weight == None:
            if (lm_scale, ilm_scale) != (1.6, 1.1):
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
            exclude_epochs=exclude_epochs,
            train_exp_name=name,
            dev_sets=["dev", "test"],
        )

    # ------------- Ted2 recognitions with frame-level prior --------------
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [1.6, 1.7]
    ilm_scales = [0.8, 1.3] 
    prior_scales = [0.4, 0.1]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):                
        if ilm_scale >= lm_scale:
            continue
        scales = (lm_scale, ilm_scale, prior_scale)
        config_weight = config.get("kldiv_sampling_weight", None)
        exclude_epochs = [40, 60, 80, 100]
        if config_weight == 0.5:
            if scales != (1.7, 1.3, 0.1):
                continue
        elif config_weight == None:
            if scales != (1.6, 0.8, 0.4):
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
            exclude_epochs=exclude_epochs,
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
