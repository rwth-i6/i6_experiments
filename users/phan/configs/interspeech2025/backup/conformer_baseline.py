"""
Config for running baseline recognition (with and without ELM and frame-level prior, no ILM here)
"""

from __future__ import annotations

from typing import Optional, Union, Tuple, Sequence, List
import itertools
import copy
from sisyphus import tk

from i6_experiments.users.yang.torch.luca_ctc.configs import *

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def
from i6_experiments.users.phan.rf_models.default_model_configs import default_ilm_config, default_tedlium2_extern_lm_config

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
    lr_list = [(1e-3, 1e-4)]
    ep_list = [(40, 80)]
    # ground_truth_weights = [0.5, "average"]
    # recog_epoch = [1] + list(range(20, 120, 20))
    recog_epoch = list(range(20, 120, 20)) + [1]
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

    ted2_prefix = "lbs_cross_domain_ted2/" + _sis_prefix 
    ted2_task = _get_ted2_task()
    ted2_sis_prefix = "lbs_cross_domain_ted2/" + _sis_prefix

    recog_config_update = {
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

    # ------------------- LibriSpeech: baseline CTC + only ELM, no frame-level prior ---------------------
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [0.8]
    ilm_scales = [0.0] 
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
        exp_name = "/conformer_baseline"
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


    # -------------------------- LibriSpeech no ELM (greedy search) ------------------------
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
        exp_name = "/conformer_baseline"
        suffix = f"_greedy_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 1800000,
            "search_args": search_args,
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



    # ------------------- Ted2 recognitions: CTC + only ELM, no frame-level prior ---------------------
    # Important: turn on LM skip 
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [0.7]
    ilm_scales = [0.0]
    prior_scales = [0.0]
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
        exp_name = "/conformer_baseline_SF_no_prior"
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
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

        # ------------------- Tedlium2: CTC + only ELM + frame-level prior ---------------------
    # Important: turn on LM skip 
    from i6_experiments.users.phan.rf_models.default_checkpoints import default_ted2_lstm_extern_lm_checkpoint
    from i6_experiments.users.phan.recog.ctc_time_sync_recomb_first_v2 import model_recog_time_sync_recomb_first_v2
    beam_sizes = [32]
    length_norm_scales = [0.0]
    lm_scales = [1.1]
    ilm_scales = [0.0]
    prior_scales = [0.6]
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
        exp_name = "/conformer_baseline_SF_with_prior"
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
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

    # ----------- ted2 recognitions without ELM (is basically greedy search) ---------------
    beam_sizes = [32]
    length_norm_scales = [0.0]
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
        suffix = f"_timeSyncRecombFirstV2_beam-{beam_size}_lm-{lm_scale}_ilm-{ilm_scale}_lenNorm-{length_norm_scale}_prior-{prior_scale}"
        recog_name = ted2_prefix + exp_name + suffix
        recog_config_update_extra = copy.deepcopy(recog_config_update)
        recog_config_update_extra.update({
            "batch_size": 2400000,
            "search_args": search_args,
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
        )
        tk.register_output(ted2_sis_prefix + exp_name + suffix + "/recog_results_per_epoch/baseline", res.output)

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

