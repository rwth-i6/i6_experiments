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

from i6_experiments.users.phan.rf_models.model_conformer_with_ilm_v2 import from_scratch_model_def, train_masked_bi_ilm_masking
from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog

from i6_experiments.users.phan.rf_models.default_model_configs import default_bidirectional_ilm_config, default_extern_lm_config, \
    default_tedlium2_extern_lm_config, default_tedlium2_extern_lm_hardcoded_layers_config


from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef
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

# set the ILM hyperparams for all experiments
# hyperparams referrence /u/michel/setups/language_modelling/librispeech/neurallm/decoder_sized_transcripts_only_newrun
config_11gb.update({"internal_language_model": default_bidirectional_ilm_config})

# config 24GB for higher target masking rate
config_24gb = copy.deepcopy(config_24gb)
config_24gb.pop("dynamic_learning_rate", None)
config_24gb.pop("learning_rate_piecewise_steps", None)
config_24gb.pop("learning_rate_piecewise_values", None)
config_24gb.pop("learning_rate_invsqrt_norm", None)
config_24gb.pop("learning_rate_warmup_steps", None)
config_24gb.pop("specaugment_steps", None)
config_24gb.pop("torch_amp", None)
config_24gb.update({"internal_language_model": default_bidirectional_ilm_config})

# set the ILM hyperparams for all experiments
# hyperparams referrence /u/michel/setups/language_modelling/librispeech/neurallm/decoder_sized_transcripts_only_newrun
config_11gb.update({"internal_language_model": default_bidirectional_ilm_config})

def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    lr_list = [1e-3, 1e-4] # [(1e-5, 1e-7), (1e-3, 1e-3)]
    ep_list = [100]
    # recog_epoch = [1] + list(range(20, 120, 20)) # 1 is mostly for debugging and getting the baseline
    recog_epoch = [20, 40, 60, 80, 100]
    target_masking_rates = [0.2, 0.4] # this for now
    # Standard KLDiv ILM
    for lr in lr_list:
        for epoch in ep_list:
            for target_masking_rate in target_masking_rates:
                if target_masking_rate > 0.2:
                    gpu_config = config_24gb
                    gpu_mem = 11
                else:
                    gpu_config = config_11gb
                    gpu_mem =11
                train_exp( 
                    f"conformer_bi_ilm_kldiv_masking_targetMaskRate_{target_masking_rate}_lr_{lr}_ep_{epoch}",
                    gpu_config,
                    train_masked_bi_ilm_masking,
                    gpu_mem=gpu_mem,
                    config_updates={
                        "batch_size": 1200000,
                        "learning_rate": lr,
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
                        "load_vocab": "/work/common/asr/librispeech/data/sisyphus_work_dir/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab",
                        "mel_normalization_ted2": False,
                        "use_specaugment": False, # VERY IMPORTANT!!!
                        "target_masking_rate": target_masking_rate,
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
        use_multiproc_dataset=True,
        train_dataset_key="train-other-960",
    )
    learning_rates = model_with_checkpoint.get_training_job()
    tk.register_output(prefix + "/train/learning_rates", learning_rates.out_learning_rates)

    # # greedy search: we won't use greedy search anyway
    # if greedy_search:
    #     from i6_experiments.users.yang.torch.luca_ctc.model_recog_ctc_greedy import model_recog
    #     recog_training_exp(
    #         prefix, task, model_with_checkpoint, recog_def=model_recog, model_avg=model_avg
    #     )
    # Luca's label sync search with LM and ILM scale

    # # -------- compute some statistics related to the KL Div --------
    # from i6_experiments.users.phan.forward_misc import generic_forward_config, compute_kldiv
    # dataset_keys = ["dev-other", "test-other", "dev-clean"]
    # forward_extra_config = copy.deepcopy(config)
    # forward_extra_config.update({
    #     "batch_size": 4800000,
    #     "max_seqs": 200,
    #     "preload_from_files": {
    #         "01_trafo_lm": {
    #             "prefix": "language_model.",
    #             "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
    #         },
    #     },
    #     "external_language_model": default_extern_lm_config,
    # })
    # forward_post_config = dict(
    #     torch_log_memory_usage=True,
    #     use_lovely_tensors=True,
    # )
    
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

    # recog_config_update = {
    #     'batch_size': 600000, # super slow
    #     "preload_from_files": {
    #         "01_trafo_lm": {
    #             "prefix": "language_model.",
    #             "filename": "/work/asr3/zeineldeen/hiwis/luca.gaudino/setups-data/2023-08-10--rf-librispeech/work/i6_experiments/users/gaudino/returnn/convert_ckpt_rf/librispeech/trafo_lm_only_24_02_06/network.023.pt",
    #         },
    #     },
    #     "internal_language_model": default_bidirectional_ilm_config,
    #     "external_language_model": default_extern_lm_config, # this to load the external LM only in recog
    # }


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


    # ----------------- TED2 time sync search and ILM rescoring ----------------
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
        "internal_language_model": default_bidirectional_ilm_config,
        "external_language_model": default_tedlium2_extern_lm_config, # this to load the external LM only in recog
    }

    beam_sizes = [32]
    first_pass_lm_scales = [0.6, 0.7, 0.8]
    scale_diff = [0.5, 0.6, 0.7]
    second_pass_lm_scales = [round(lm + diff, 2) for lm in first_pass_lm_scales for diff in scale_diff]
    ilm_scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for beam_size, first_pass_lm_scale, second_pass_lm_scale, ilm_scale in itertools.product(beam_sizes, first_pass_lm_scales, second_pass_lm_scales, ilm_scales):                
        if ilm_scale >= second_pass_lm_scale:
            continue
        search_args = {
            "beam_size": beam_size,
            "lm_scale": first_pass_lm_scale,
            "length_norm_scale": 0.0,
            "prior_scale": 0.0,
            # "prior_file": "/work/asr3/zyang/share/mnphan/work_rf_ctc/work/i6_core/returnn/forward/ReturnnForwardJobV2.Wug49TgveO2b/output/prior.txt",
            "ctc_log_prior": False,
            "lm_skip": True,
        }
        
        ted2_recog_config_update_extra = copy.deepcopy(ted2_recog_config_update)
        ted2_recog_config_update_extra.update({
            "search_args": search_args,
            "split_am_lm_score": True,
        })
        rescoring_args = {
            "beam_size": beam_size,
            "lm_scale": second_pass_lm_scale,
            "ilm_scale": ilm_scale,
        }
        rescoring_config = copy.deepcopy(ted2_recog_config_update_extra)
        rescoring_config.update({
            "batch_size": 45000,
            "rescoring_args": rescoring_args,
            "load_vocab": "/u/zyang/setups/vocab/librispeech/bpe10k/lower_case.bpe.vocab",
        })
        ted2_recog_config_update_extra.pop("internal_language_model", None)
        recog_training_exp(
            ted2_prefix + f"_beam-{beam_size}_firstPassLM-{first_pass_lm_scale}_secondPassLM-{second_pass_lm_scale}_ilm-{ilm_scale}",
            ted2_task,
            model_with_checkpoint,
            search_config=ted2_recog_config_update_extra,
            recog_def=model_recog_time_sync_recomb_first_v2,
            model_avg=False,
            exclude_epochs=[],
            train_exp_name=name,
            dev_sets=["dev", "test"],
            hyps_rescoring=True,
            rescoring_config=rescoring_config,
            overwrite_first_pass_model=ModelWithCheckpoint( # because we only train the ILM, use the baseline for the first pass
                definition=from_scratch_model_def,
                checkpoint=tk.Path("/work/asr4/zyang/torch_checkpoints/ctc/luca_20240617_noeos/epoch.1982.pt"),
            )
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
        # alignments[dataset_key] = [get_alignment(dataset_key=dataset_key)] # str to list
        # alignments[dataset_key] = [get_gmm_pseudo_word_alignments().out_hdf]
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
