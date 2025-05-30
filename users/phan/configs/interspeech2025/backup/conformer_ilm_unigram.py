"""
Config for doing recognition with ELM and unigram ILM (renormalized frame-level prior)
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Sequence, List
import copy
import itertools
from sisyphus import tk

from i6_experiments.users.yang.torch.luca_ctc.configs import *

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def
from i6_experiments.users.phan.rf_models.default_model_configs import default_extern_lm_config, default_tedlium2_extern_lm_config

from i6_experiments.users.zeyer.model_with_checkpoints import (
    ModelWithCheckpoints,
    ModelWithCheckpoint,
)


config_11gb = copy.deepcopy(config_11gb)
config_11gb.pop("dynamic_learning_rate", None)
config_11gb.pop("learning_rate_piecewise_steps", None)
config_11gb.pop("learning_rate_piecewise_values", None)
config_11gb.pop("learning_rate_invsqrt_norm", None)
config_11gb.pop("learning_rate_warmup_steps", None)


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    train_exp( 
        f"conformer_ilm_unigram_renorm_prior",
        config_11gb,
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
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,

    length_norm_scale=0.0,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from i6_experiments.users.yang.torch.luca_ctc.recog import recog_model

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

    recog_config_update = {
        # "torch_amp": "bfloat16",
        # 'batch_size':1200000,
        'batch_size': 600000,
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

    unigram_renorm_prior_config = {
        "class": "Unigram_LM_RF",
        "preload_tensor": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/forward/ReturnnForwardJobV2.Wug49TgveO2b/output/prior.txt",
        "preload_is_log_prob": False,
        "preload_blank_idx": 10025,
    }
    ilm_arch = "unigram_renorm_prior"
    epoch = 0

    # ---------------- Compute unigram ILM statistics on librispeech -----------------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev-other", "test-other", "dev-clean"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "external_language_model": default_extern_lm_config,
        "internal_language_model": unigram_renorm_prior_config,
        "no_bos_eos": True,
    })
    forward_extra_config["preload_from_files"].update({
        "01_trafo_lm": {
            "prefix": "language_model.",
            "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
        },
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
            job_vis_name=f"Compute ILM stats job, {ilm_arch}, epoch {epoch}, {dataset_key}",
            extra_hash={"version": "04/11/2024"}, 
        )
        out_stat_file = stats_job.out_files[compute_kldiv.default_out_file_name]
        stats_job.add_alias(_sis_prefix + f"/{ilm_arch}" + "/ilm_stats_v2_no_eos" + f"/{dataset_key}/{epoch}")
        tk.register_output(_sis_prefix + f"/{ilm_arch}" + f"/ilm_stats_v2_no_eos/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    
    # # ------------------------ cross-domain tedlium2 (estimation on LBS, recognition on ted2) -------------------------
    ted2_prefix = "lbs_cross_domain_ted2/" + _sis_prefix + "/" + ilm_arch
    ted2_task = _get_ted2_task()
    ted2_sis_prefix = "lbs_cross_domain_ted2/" + _sis_prefix

    # -------- compute ted2 ILM dev, test PPL --------
    from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    dataset_keys = ["dev", "test"]
    forward_extra_config = copy.deepcopy(config)
    forward_extra_config.update({
        "batch_size": 4800000,
        "max_seqs": 200,
        "with_extern_lm": True,
        "external_language_model": default_tedlium2_extern_lm_config,
        "internal_language_model": unigram_renorm_prior_config,
        "no_bos_eos": True,
    })
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    forward_extra_config["preload_from_files"].update({
        "01_extern_lstm_lm": {
            "prefix": "language_model.",
            "filename": default_ted2_lstm_extern_lm_checkpoint,
        },
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
        stats_job.add_alias(ted2_prefix + "/ilm_stats_v2_no_eos" + f"/{dataset_key}/{epoch}")
        tk.register_output(ted2_prefix + f"/ilm_stats_v2_no_eos/{dataset_key}/{epoch}/{compute_kldiv.default_out_file_name}" , out_stat_file)

    # # # ------------- Librispeech recognition without frame-level prior ------------
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0] # we don't need 1.0 for time sync search!!!
    lm_scales = [0.8]
    ilm_scales = [0.4] 
    prior_scales = [0.0]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):
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
            search_args.update({
                "prior_scale": prior_scale,
                "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
                "ctc_log_prior": False,
            })
        exp_name = f"/{ilm_arch}"
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 600000,
            "search_args": search_args,
            "internal_language_model": unigram_renorm_prior_config,
        })
        res = recog_model(
            task,
            ModelWithCheckpoint(
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            ),
            recog_def=model_recog_time_sync_recomb_first_v2,
            config=recog_config_update_extra,
            search_rqmt={"time": 7},
            dev_sets=["dev-other", "test-other", "dev-clean", "test-clean"],
            name=recog_name,
            epoch=epoch
        )
        tk.register_output(_sis_prefix + exp_name + suffix + f"/recog_results_per_epoch/{epoch}", res.output)

    # --------------------- Cross-domain recognition on Tedlium 2, without frame-level prior ---------------------
    # prior = 0.0
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [1.4]
    ilm_scales = [0.9] 
    prior_scales = [0.0]
    for beam_size, lm_scale, ilm_scale, length_norm_scale, prior_scale in itertools.product(beam_sizes, lm_scales, ilm_scales, length_norm_scales, prior_scales):
        if ilm_scale >= lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "length_normalization_exponent": length_norm_scale, # by default len norm
            "lm_scale": lm_scale,
            "ilm_scale": ilm_scale,
            "lm_skip": True, # IMPORTANT!!!
        }
        search_args.update({
            "prior_scale": prior_scale,
            "prior_file": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_core/returnn/forward/ReturnnForwardJobV2.OSftOYzAjRUg/output/prior.txt",
            "ctc_log_prior": False,
        })
        exp_name = f"/{ilm_arch}"
        suffix = f"_timeSyncRecombFirstV2_mergecontraction_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 2400000,
            "search_args": search_args,
            "preload_from_files": {
                "01_lstm_extern_lm": {
                    "prefix": "language_model.",
                    "filename": default_ted2_lstm_extern_lm_checkpoint,
                }
            },
            "external_language_model": default_tedlium2_extern_lm_config,
            "internal_language_model": unigram_renorm_prior_config,
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
            epoch=epoch,
            merge_contraction=True,
        )
        tk.register_output(ted2_sis_prefix + exp_name + suffix + f"/recog_results_per_epoch/{epoch}", res.output)

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
