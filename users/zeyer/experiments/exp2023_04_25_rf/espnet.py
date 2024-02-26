"""
Attention-based encoder-decoder (AED) experiments, using ESPnet models
"""

from __future__ import annotations

import os
import copy
import sys
import logging
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, List

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from i6_experiments.users.zeyer.model_interfaces import ModelDef, RecogDef, TrainDef, ModelDefWithCfg
from i6_experiments.users.zeyer.accum_grad_schedules.piecewise_linear import dyn_accum_grad_piecewise_linear

from .configs import *
from .configs import _get_cfg_lrlin_oclr_by_bs_nep

if TYPE_CHECKING:
    import torch
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints, ModelWithCheckpoint
    from i6_experiments.users.zeyer.datasets.task import Task
    from espnet2.asr.espnet_model import ESPnetASRModel
    from espnet.nets.scorer_interface import BatchScorerInterface

# The model gets raw features (16khz) and does feature extraction internally.
_log_mel_feature_dim = 80


def sis_run_with_prefix(prefix_name: Optional[str] = None):
    """run the exp"""
    _sis_setup_global_prefix(prefix_name)

    train_exp(  # 5.15
        "v6-24gb-bs30k-wd1e_6-lrlin1e_5_587k-EBranchformer",
        config_24gb_v6,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(30_000, 2000),
        },
        model_avg=True,
    )

    # uncomment this to get the CUDA OOM error in dist.all_reduce: https://github.com/rwth-i6/returnn/issues/1482
    # train_exp(
    #     "v6-11gb-f32-bs8k-accgrad1-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-ncclError",
    #     config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
    #         "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
    #     },
    # )

    train_exp(
        "v6-11gb-f32-bs8k-accgrad100-mgpu4-wd1e_4-lrlin1e_5_558k-EBranchformer",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed": {"options": {"find_unused_parameters": True}},
            "accum_grad_multiple_step": 100,
        },
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},  # OOM in epoch 28 and later...
    )

    train_exp(  # 6.52
        "v6-11gb-f32-bs8k-accgrad100-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": 100,
        },
    )

    train_exp(  # 5.58
        "v6-11gb-f32-bs8k-accgrad10-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": 10,
        },
    )

    train_exp(  # 5.23
        "v6-11gb-f32-bs8k-accgrad4-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": 4,
        },
    )

    train_exp(  # 5.38
        "v6-11gb-f32-bs8k-lr2e_3-warmup500k-accgrad4-mgpu4-pavg100-wd1e_4-EBranchformer",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            "batch_size": 8_000 * _batch_size_factor,
            "learning_rate": 2e-3,
            "dynamic_learning_rate": dyn_lr_lin_warmup_invsqrt_decay,
            "learning_rate_warmup_steps": 500_000,
            "learning_rate_invsqrt_norm": 40_000,
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": 4,
        },
    )

    train_exp(  # 5.17
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": _dyn_accum_grad_multiple_step_v2,
        },
        model_avg=True,
    )

    train_exp(  # 5.56
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin2e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500, peak_lr=2e-3),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": _dyn_accum_grad_multiple_step_v2,
        },
        model_avg=True,
    )

    # TODO also try model average

    train_exp(  # 5.29
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV1a",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": _dyn_accum_grad_multiple_step_v1a,
        },
    )

    model = train_exp(  # 5.11
        "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_2-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )
    for name, recog_config in {
        "ctc03-beam12-batch200": {
            # {"dev-clean": 2.24, "dev-other": 5.12, "test-clean": 2.35, "test-other": 5.23}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0.3},
        },
        "ctc03-beam12-batch50": {
            # {"dev-clean": 2.24, "dev-other": 5.13, "test-clean": 2.35, "test-other": 5.22}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0.3},
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "ctc03-beam12-batch1": {
            # {"dev-clean": 2.24, "dev-other": 5.14, "test-clean": 2.35, "test-other": 5.21}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0.3},
            "max_seqs": 1,
            "_trigger_hash_change": 1,
        },
        "ctc03-beam60-batch1": {
            # {"dev-clean": 2.22, "dev-other": 5.13, "test-clean": 2.34, "test-other": 5.16}
            "beam_search_opts": {"beam_size": 60, "ctc_weight": 0.3},
            "max_seqs": 1,
        },
        "ctc0-beam12-batch200": {
            # {"dev-clean": 2.85, "dev-other": 5.54, "test-clean": 3.02, "test-other": 5.62}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0},
        },
        "ctc0-beam12-batch50": {
            # {"dev-clean": 2.74, "dev-other": 5.51, "test-clean": 3.25, "test-other": 5.61}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0},
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "ctc0-beam12-batch1": {
            # {"dev-clean": 2.74, "dev-other": 5.51, "test-clean": 3.25, "test-other": 5.6}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0},
            "max_seqs": 1,
        },
        # "ctc1-beam12-batch200": {"beam_search_opts": {"beam_size": 12, "ctc_weight": 1}},  # TODO why OOM?
        "ctc1-beam12-batch50": {
            # {"dev-clean": 2.83, "dev-other": 6.61, "test-clean": 3.02, "test-other": 6.61}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 1},
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
        },
        "ctc1-beam12-batch1": {
            # {"dev-clean": 2.84, "dev-other": 6.61, "test-clean": 3.03, "test-other": 6.6}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 1},
            "max_seqs": 1,
        },
    }.items():
        _recog(
            "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_2-lrlin1e_5_558k-EBranchformer-dynGradAccumV2/recog-last-espnet-"
            + name,
            model.get_last_fixed_epoch(),
            model_recog,
            # TODO trigger new hash to get timing logs...
            {"search_version": 4, "__batch_size_dependent": True, **recog_config},
        )
    for name, recog_config in {
        # OOM with batch50 once there is CTC...
        "ctc03-beam12-batch20": {
            # {"dev-clean": 2.17, "dev-other": 5.12, "test-clean": 2.31, "test-other": 5.12}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0.3},
            "max_seqs": 20,
            "batch_size": 2000 * _batch_size_factor,
            "_trigger_hash_change": 1,
        },
        "ctc03-beam12-batch1": {
            # {"dev-clean": 2.17, "dev-other": 5.12, "test-clean": 2.31, "test-other": 5.12}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0.3},
            "max_seqs": 1,
        },
        "ctc0-beam12-batch50": {
            # {"dev-clean": 2.75, "dev-other": 5.36, "test-clean": 3.25, "test-other": 5.52}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0},
            "max_seqs": 50,
            "batch_size": 5000 * _batch_size_factor,
            "_trigger_hash_change": 1,
        },
        "ctc0-beam12-batch1": {
            # {"dev-clean": 2.74, "dev-other": 5.35, "test-clean": 3.25, "test-other": 5.51}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 0},
            "max_seqs": 1,
        },
        "ctc1-beam12-batch20": {
            # {"dev-clean": 2.83, "dev-other": 6.62, "test-clean": 3.03, "test-other": 6.6}
            "beam_search_opts": {"beam_size": 12, "ctc_weight": 1},
            "max_seqs": 20,
            "batch_size": 2000 * _batch_size_factor,
        },
    }.items():
        _recog(
            "v6-11gb-f32-bs8k-mgpu4-pavg100-wd1e_2-lrlin1e_5_558k-EBranchformer-dynGradAccumV2/recog-last-our-" + name,
            model.get_last_fixed_epoch(),
            model_recog_our,
            {"__batch_size_dependent": True, "beam_search_collect_individual_seq_scores": True, **recog_config},
        )

    train_exp(  # 6.13
        "v6-11gb-f32-bs8k-mgpu2-nep500-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            "__num_processes": 2,
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )

    train_exp(  # 8.62 (interestingly, mgpu4-nep125 below is better than this)
        "v6-11gb-f32-bs8k-nep500-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            "__num_processes": None,
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 500),
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000, 100_000, 1_100_000, 1_242_000],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
        config_deletes=["__num_processes", "torch_distributed"],
    )

    train_exp(  # 5.85
        "v6-11gb-f32-bs8k-mgpu4-nep250-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 250),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000 // 2, 100_000 // 2, 1_100_000 // 2, 1_242_000 // 2],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )

    train_exp(  # 7.26
        "v6-11gb-f32-bs8k-mgpu4-nep125-pavg100-wd1e_4-lrlin1e_5_558k-EBranchformer-dynGradAccumV2",
        config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
        {
            "espnet_config": "egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml",
            "espnet_fixed_sos_eos": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep(8_000, 125),
            "torch_distributed.sync_on_cpu": True,  # https://github.com/rwth-i6/returnn/issues/1482
            "accum_grad_multiple_step": dyn_accum_grad_piecewise_linear,
            "accum_grad_piecewise_steps": [50_000 // 4, 100_000 // 4, 1_100_000 // 4, 1_242_000 // 4],
            "accum_grad_piecewise_values": [1, 100, 1, 1, 10],
        },
    )


_sis_prefix: Optional[str] = None


def _sis_setup_global_prefix(prefix_name: Optional[str] = None):
    if not prefix_name:
        from .sis_setup import get_prefix_for_config

        prefix_name = get_prefix_for_config(__file__)
    global _sis_prefix
    _sis_prefix = prefix_name


def _recog(
    name: str,
    model_with_checkpoint: ModelWithCheckpoint,
    recog_def: RecogDef,
    recog_config: Optional[Dict[str, Any]] = None,
):
    from sisyphus import tk
    from i6_experiments.users.zeyer.recog import recog_model

    task = _get_ls_task()

    res = recog_model(task, model_with_checkpoint, recog_def=recog_def, config=recog_config)
    tk.register_output(_sis_prefix + "/" + name, res.output)


# noinspection PyShadowingNames
def train_exp(
    name: str,
    config: Dict[str, Any],
    model_config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    post_config_updates: Optional[Dict[str, Any]] = None,
    env_updates: Optional[Dict[str, str]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 24,
    num_processes: Optional[int] = None,
    time_rqmt: Optional[int] = None,  # set this to 1 or below to get the fast test queue
    with_eos_postfix: bool = False,
    model_avg: bool = False,
) -> ModelWithCheckpoints:
    """
    Train experiment
    """
    from .train import train
    from i6_experiments.users.zeyer.recog import recog_training_exp

    if _sis_prefix is None:
        _sis_setup_global_prefix()

    prefix = _sis_prefix + "/" + name
    task = _get_ls_task(with_eos_postfix=with_eos_postfix)
    config = config.copy()
    config = dict_update_deep(config, config_updates, config_deletes)
    if "__num_epochs" in config:
        num_epochs = config.pop("__num_epochs")
    if "__gpu_mem" in config:
        gpu_mem = config.pop("__gpu_mem")
    if "__num_processes" in config:
        num_processes = config.pop("__num_processes")
    if "__train_audio_preprocess" in config:
        task: Task = copy.copy(task)
        task.train_dataset = copy.copy(task.train_dataset)
        task.train_dataset.train_audio_preprocess = config.pop("__train_audio_preprocess")

    model_with_checkpoint = train(
        prefix,
        task=task,
        config=config,
        post_config=dict_update_deep(post_config, post_config_updates),
        env_updates=env_updates,
        model_def=ModelDefWithCfg(from_scratch_model_def, model_config),
        train_def=from_scratch_training,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        distributed_launch_cmd="torchrun" if num_processes else "mpirun",
        time_rqmt=time_rqmt,
    )
    recog_training_exp(
        prefix,
        task,
        model_with_checkpoint,
        recog_def=model_recog,
        search_config={"search_version": 4, "num_epochs": num_epochs},
        model_avg=model_avg,
    )

    return model_with_checkpoint


_ls_task: Dict[bool, Task] = {}  # with_eos_postfix -> Task


def _get_ls_task(*, with_eos_postfix: bool = False):
    if with_eos_postfix in _ls_task:
        return _ls_task[with_eos_postfix]

    from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_bpe10k_raw

    _ls_task[with_eos_postfix] = get_librispeech_task_bpe10k_raw(with_eos_postfix=with_eos_postfix)
    return _ls_task[with_eos_postfix]


py = sis_run_with_prefix  # if run directly via `sis m ...`

_batch_size_factor = 160


def _get_bos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.bos_label_id is not None:
        bos_idx = target_dim.vocab.bos_label_id
    elif target_dim.vocab.eos_label_id is not None:
        bos_idx = target_dim.vocab.eos_label_id
    elif "<sil>" in target_dim.vocab.user_defined_symbol_ids:
        bos_idx = target_dim.vocab.user_defined_symbol_ids["<sil>"]
    else:
        raise Exception(f"cannot determine bos_idx from vocab {target_dim.vocab}")
    return bos_idx


def _get_eos_idx(target_dim: Dim) -> int:
    """for non-blank labels"""
    assert target_dim.vocab
    if target_dim.vocab.eos_label_id is not None:
        eos_idx = target_dim.vocab.eos_label_id
    else:
        raise Exception(f"cannot determine eos_idx from vocab {target_dim.vocab}")
    return eos_idx


def _dyn_accum_grad_multiple_step(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    if global_train_step <= 10_000:
        return 4
    if global_train_step <= 50_000:
        return 2
    return 1


def _dyn_accum_grad_multiple_step_v1a(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    if global_train_step <= 20_000:
        return 4
    if global_train_step <= 100_000:
        return 2
    return 1


def _dyn_accum_grad_multiple_step_v2(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    # Schedule:
    # start low (to get from random init somewhere more sensible fast),
    # increase to almost 100 (to get it to convergence),
    # decay again (to get faster convergence),
    # and then maybe at the very end increase again (for finetuning).
    # Assume ~1.242k steps in total.

    steps = [50_000, 100_000, 1_100_000, 1_242_000]
    values = [1, 100, 1, 1, 10]
    assert len(steps) + 1 == len(values)

    last_step = 0
    for i, step in enumerate(steps):
        assert step > last_step
        assert global_train_step >= last_step
        if global_train_step < step:
            factor = (global_train_step + 1 - last_step) / (step - last_step)
            return int(values[i + 1] * factor + values[i] * (1 - factor))
        last_step = step

    return values[-1]


def from_scratch_model_def(*, epoch: int, in_dim: Dim, target_dim: Dim) -> ESPnetASRModel:
    """Function is run within RETURNN."""
    import returnn
    from returnn.config import get_global_config

    in_dim, epoch  # noqa
    config = get_global_config(return_empty_if_none=True)  # noqa

    # Load some train yaml file for model def.
    # References:
    # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/run.sh
    # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh
    # https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_train.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/asr.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/abs_task.py

    tools_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__))))
    print("tools dir:", tools_dir)
    sys.path.append(tools_dir + "/espnet")

    import espnet2

    espnet_repo_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(espnet2.__file__)))

    from espnet2.tasks.asr import ASRTask
    from espnet2.asr.espnet_model import ESPnetASRModel

    enc_aux_logits = config.typed_value("aux_loss_layers")
    pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
    # real input is raw audio, internally it does logmel
    in_dim = Dim(name="logmel", dimension=_log_mel_feature_dim, kind=Dim.Types.Feature)

    espnet_config_file = config.value("espnet_config", None)
    assert espnet_config_file
    parser = ASRTask.get_parser()
    args = parser.parse_args(["--config", espnet_repo_root_dir + "/" + espnet_config_file])
    args.token_list = target_dim.vocab.labels
    assert config.bool("espnet_fixed_sos_eos", False)
    args.model_conf["sym_sos"] = target_dim.vocab.labels[_get_bos_idx(target_dim)]
    args.model_conf["sym_eos"] = target_dim.vocab.labels[_get_eos_idx(target_dim)]

    # TODO any of these relevant?
    #             --use_preprocessor true \
    #             --bpemodel "${bpemodel}" \
    #             --token_type "${token_type}" \
    #             --token_list "${token_list}" \
    #             --non_linguistic_symbols "${nlsyms_txt}" \
    #             --cleaner "${cleaner}" \
    #             --g2p "${g2p}" \
    #             --valid_data_path_and_name_and_type "${_asr_valid_dir}/${_scp},speech,${_type}" \
    #             --valid_shape_file "${asr_stats_dir}/valid/speech_shape" \
    #             --resume true \
    #             ${pretrained_model:+--init_param $pretrained_model} \
    #             --ignore_init_mismatch ${ignore_init_mismatch} \
    #             --fold_length "${_fold_length}" \

    model = ASRTask.build_model(args)
    assert isinstance(model, ESPnetASRModel)
    print("Target dim:", target_dim)
    print("Vocab size:", model.vocab_size)
    print("Vocab:", target_dim.vocab.labels[:5], "...", target_dim.vocab.labels[-5:])
    print("Ignore:", model.ignore_id)
    print("Blank:", model.blank_id)
    print("SOS/EOS:", model.sos, model.eos)
    model.returnn_epoch = epoch
    model.returnn_target_dim = target_dim
    return model


from_scratch_model_def: ModelDef[ESPnetASRModel]
from_scratch_model_def.behavior_version = 16
from_scratch_model_def.backend = "torch"
from_scratch_model_def.batch_size_factor = _batch_size_factor


def from_scratch_training(
    *, model: ESPnetASRModel, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim
):
    """Function is run within RETURNN."""
    import torch
    import returnn.frontend as rf

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    loss, stats, weight = model(
        speech=data.raw_tensor,
        speech_lengths=data_spatial_dim.dyn_size,
        text=targets.raw_tensor.to(torch.int64),
        text_lengths=targets_spatial_dim.dyn_size,
    )
    # TODO the following is correct for CE and CTC, but not correct for CER and probably others, need to check...
    # ESPnet usually does divide the loss by num seqs (batch dim) but not by seq length.
    custom_inv_norm_factor = targets_spatial_dim.get_size_tensor()
    custom_inv_norm_factor = rf.cast(custom_inv_norm_factor, "float32")
    batch_dim_value = custom_inv_norm_factor.dims[0].get_dim_value_tensor()
    custom_inv_norm_factor /= rf.cast(batch_dim_value, "float32")
    rf.get_run_ctx().mark_as_loss(loss, "total", custom_inv_norm_factor=custom_inv_norm_factor)
    for k, v in stats.items():
        if v is not None:
            rf.get_run_ctx().mark_as_loss(v, k, as_error=True)


from_scratch_training: TrainDef[ESPnetASRModel]
from_scratch_training.learning_rate_control_error_measure = "dev_score_full_sum"


def model_recog(
    *,
    model: ESPnetASRModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    from returnn.config import get_global_config

    config = get_global_config()
    search_version = config.int("search_version", 0)
    assert search_version >= 3, f"search version {search_version} unsupported, likely there was a bug earlier..."
    # version 3 was setting RETURNN_FIX_BLANK to have ESPnet blank fixed.
    #   But now this has been merged in ESPnet. https://github.com/espnet/espnet/pull/5620
    #   We maybe should check for the right ESPnet version... Look out for unusual long recognized seqs.

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    # References:
    # https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/run.sh
    # https://github.com/espnet/espnet/blob/master/egs2/TEMPLATE/asr1/asr.sh
    # https://github.com/espnet/espnet/blob/master/espnet2/bin/asr_inference.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/asr.py
    # https://github.com/espnet/espnet/blob/master/espnet2/tasks/abs_task.py

    # decode_asr.yaml:
    # beam_size: 60
    # ctc_weight: 0.3
    # lm_weight: 0.6

    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    print("beam search opts:", beam_search_opts)

    beam_size = beam_search_opts.pop("beam_size", 12)  # like RETURNN, not 60 for now...
    ctc_weight = beam_search_opts.pop("ctc_weight", 0.3)
    # lm_weight = beam_search_opts.pop("lm_weight", 0.6)  # not used currently...
    # ngram_weight = beam_search_opts.pop("ngram_weight", 0.9)  # not used currently...
    penalty = beam_search_opts.pop("length_reward", 0.0)
    normalize_length = beam_search_opts.pop("normalize_length", False)  # note: only at the end
    maxlenratio = beam_search_opts.pop("maxlenratio", 0.0)
    minlenratio = beam_search_opts.pop("minlenratio", 0.0)
    assert not beam_search_opts, f"found unused opts: {beam_search_opts}"

    # Partly taking code from espnet2.bin.asr_inference.Speech2Text.

    import time
    import torch
    from espnet.nets.scorers.ctc import CTCPrefixScorer
    from espnet.nets.scorers.length_bonus import LengthBonus
    from espnet.nets.scorer_interface import BatchScorerInterface
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.beam_search import Hypothesis

    asr_model = model

    ctc = CTCPrefixScorer(ctc=asr_model.ctc, eos=asr_model.eos)
    token_list = asr_model.token_list
    scorers, weights = {}, {}
    if ctc_weight != 1:
        scorers["decoder"] = model.decoder
        weights["decoder"] = 1.0 - ctc_weight
    if ctc_weight != 0:
        scorers["ctc"] = ctc
        weights["ctc"] = ctc_weight
    if penalty != 0:
        scorers["length_bonus"] = LengthBonus(len(token_list))
        weights["length_bonus"] = penalty

    assert all(isinstance(v, BatchScorerInterface) for k, v in scorers.items()), f"non-batch scorers: {scorers}"

    beam_search = BatchBeamSearch(
        beam_size=beam_size,
        weights=weights,
        scorers=scorers,
        sos=asr_model.sos,
        eos=asr_model.eos,
        vocab_size=len(token_list),
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "full",
        normalize_length=normalize_length,
    )

    start_time = time.perf_counter_ns()

    speech = data.raw_tensor  # [B, Nsamples]
    print("Speech shape:", speech.shape, "device:", speech.device)
    lengths = data_spatial_dim.dyn_size  # [B]
    batch = {"speech": speech, "speech_lengths": lengths}
    logging.info("speech length: " + str(speech.size(1)))

    # Encoder forward (batched)
    enc, enc_olens = asr_model.encode(**batch)
    print("Encoded shape:", enc.shape, "device:", enc.device)

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    batch_dim = data.dims[0]
    batch_size = speech.size(0)
    beam_dim = Dim(beam_size, name="beam")
    olens = torch.zeros([batch_size, beam_size], dtype=torch.int32)
    out_spatial_dim = Dim(Tensor("out_spatial", [batch_dim, beam_dim], "int32", raw_tensor=olens))
    outputs = [[] for _ in range(batch_size)]
    oscores = torch.zeros([batch_size, beam_size], dtype=torch.float32)
    seq_log_prob = Tensor("scores", [batch_dim, beam_dim], "float32", raw_tensor=oscores)

    # BatchBeamSearch is misleading: It still only operates on a single sequence,
    # but just handles all hypotheses in a batched way.
    # So we must iterate over all the sequences here from the input.
    for i in range(batch_size):
        nbest_hyps: List[Hypothesis] = beam_search(
            x=enc[i, : enc_olens[i]], maxlenratio=maxlenratio, minlenratio=minlenratio
        )
        print("best:", " ".join(token_list[v] for v in nbest_hyps[0].yseq))
        # I'm not exactly sure why, but sometimes we get even more hyps?
        # And then also sometimes, we get less hyps?
        very_bad_score = min(-1e32, nbest_hyps[-1].score - 1)  # not -inf because of serialization issues
        while len(nbest_hyps) < beam_size:
            nbest_hyps.append(Hypothesis(score=very_bad_score, yseq=torch.zeros(0, dtype=torch.int32)))
        for j in range(beam_size):
            hyp: Hypothesis = nbest_hyps[j]
            olens[i, j] = hyp.yseq.size(0)
            outputs[i].append(hyp.yseq)
            oscores[i, j] = hyp.score

    search_end_time = time.perf_counter_ns()
    data_seq_len_sum = rf.reduce_sum(data_spatial_dim.dyn_size_ext, axis=data_spatial_dim.dyn_size_ext.dims)
    data_seq_len_sum_secs = data_seq_len_sum.raw_tensor / _batch_size_factor / 100.0
    data_seq_len_max_seqs = data_spatial_dim.get_dim_value() / _batch_size_factor / 100.0
    out_len_longest_sum = rf.reduce_sum(rf.reduce_max(out_spatial_dim.dyn_size_ext, axis=beam_dim), axis=batch_dim)
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"data len max {data_spatial_dim.get_dim_value()} ({data_seq_len_max_seqs:.2f} secs)",
                f"data len sum {data_seq_len_sum.raw_tensor} ({data_seq_len_sum_secs:.2f} secs)",
                f"enc {enc_end_time - start_time} ns",
                f"enc len max {torch.max(enc_olens)}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    outputs_t = torch.zeros([batch_size, beam_size, torch.max(olens)], dtype=torch.int32)
    for i in range(batch_size):
        for j in range(beam_size):
            outputs_t[i, j, : olens[i, j]] = outputs[i][j]
    seq_targets = Tensor("outputs", [batch_dim, beam_dim, out_spatial_dim], "int32", raw_tensor=outputs_t)

    from returnn.datasets.util.vocabulary import Vocabulary

    target_dim = Dim(name="target", dimension=len(token_list), kind=Dim.Types.Feature)
    target_dim.vocab = Vocabulary.create_vocab_from_labels(token_list, eos_label=model.eos)
    seq_targets.sparse_dim = target_dim

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog: RecogDef[ESPnetASRModel]
model_recog.output_with_beam = True
model_recog.output_blank_label = "<s>"
model_recog.batch_size_dependent = False


def model_recog_our(
    *,
    model: ESPnetASRModel,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dim, Dim]:
    """
    Function is run within RETURNN.

    Earlier we used the generic beam_search function,
    but now we just directly perform the search here,
    as this is overall simpler and shorter.

    :return:
        recog results including beam {batch, beam, out_spatial},
        log probs {batch, beam},
        recog results info: key -> {batch, beam},
        out_spatial_dim,
        final beam_dim
    """
    import torch
    import time
    from i6_experiments.users.zeyer.decoding.beam_search_torch.beam_search_v5 import BeamSearchOptsV5, beam_search_v5
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.length_reward import LengthRewardScorer
    from i6_experiments.users.zeyer.decoding.beam_search_torch.scorers.shallow_fusion import ShallowFusedLabelScorers
    from returnn.config import get_global_config

    config = get_global_config()

    start_time = time.perf_counter_ns()

    speech = data.raw_tensor  # [B, Nsamples]
    print("Speech shape:", speech.shape, "device:", speech.device)
    lengths = data_spatial_dim.dyn_size  # [B]
    batch = {"speech": speech, "speech_lengths": lengths}
    logging.info("speech length: " + str(speech.size(1)))

    # Encoder forward (batched)
    enc, enc_olens = model.encode(**batch)
    print("Encoded shape:", enc.shape, "device:", enc.device)

    batch_dim = data.dims[0]

    max_seq_len = enc_olens
    print("** max seq len:", max_seq_len)

    if data.raw_tensor.device.type == "cuda":
        # Just so that timing of encoder is correct.
        torch.cuda.synchronize(data.raw_tensor.device)

    enc_end_time = time.perf_counter_ns()

    from espnet.nets.scorers.ctc import CTCPrefixScorer

    # Note: ESPnet distinguishes between full scorers and partial scorers.
    #             if isinstance(v, PartialScorerInterface):
    #                 self.part_scorers[k] = v
    #             else:
    #                 self.full_scorers[k] = v
    # Full scorer: Given hyp (incl states, scores, etc),
    #   score all vocab labels, and calc new state.
    # Partial scorer: Given hyp and some set of labels ("part_ids"),
    #   score all vocab labels (or only the given part_ids labels, and score for others is 0), and calc new state.

    beam_search_version = config.int("beam_search_version", 5)
    beam_search_func = {5: beam_search_v5}[beam_search_version]
    beam_search_opts_cls = BeamSearchOptsV5
    beam_search_opts = (config.typed_value("beam_search_opts", None) or {}).copy()
    extra = {}
    out_individual_seq_scores = None
    if config.bool("beam_search_collect_individual_seq_scores", False):
        out_individual_seq_scores = {}
        extra["out_individual_seq_scores"] = out_individual_seq_scores
    label_scorer = ShallowFusedLabelScorers()
    ctc_weight = beam_search_opts.pop("ctc_weight", 0.3)
    if ctc_weight != 1:
        label_scorer.label_scorers["decoder"] = (
            get_our_label_scorer_intf(model.decoder, enc=enc, enc_olens=enc_olens),
            1.0 - ctc_weight,
        )
    if ctc_weight:
        label_scorer.label_scorers["ctc"] = (
            get_our_label_scorer_intf(
                CTCPrefixScorer(ctc=model.ctc, eos=model.eos),
                enc=enc,
                enc_olens=enc_olens,
            ),
            ctc_weight,
        )
    len_reward = beam_search_opts.pop("length_reward", 0.0)
    if len_reward:
        label_scorer.label_scorers["length_reward"] = (LengthRewardScorer(), len_reward)
    beam_search_opts.setdefault("length_normalization_exponent", config.float("length_normalization_exponent", 0.0))

    # Beam search happening here:
    (
        seq_targets,  # [Batch,FinalBeam,OutSeqLen]
        seq_log_prob,  # [Batch,FinalBeam]
        out_seq_len,  # [Batch,FinalBeam]
    ) = beam_search_func(
        label_scorer,
        batch_size=batch_dim.get_dim_value(),
        max_seq_len=max_seq_len,
        device=data.raw_tensor.device,
        opts=beam_search_opts_cls(
            **beam_search_opts,
            bos_label=model.sos,
            eos_label=model.eos,
            num_labels=model.vocab_size,
        ),
        **extra,
    )

    beam_dim = Dim(seq_log_prob.shape[1], name="beam")
    out_spatial_dim = Dim(rf.convert_to_tensor(out_seq_len, dims=[batch_dim, beam_dim], name="out_spatial"))
    seq_targets_t = rf.convert_to_tensor(
        seq_targets, dims=[batch_dim, beam_dim, out_spatial_dim], sparse_dim=model.returnn_target_dim
    )
    seq_log_prob_t = rf.convert_to_tensor(seq_log_prob, dims=[batch_dim, beam_dim])

    search_end_time = time.perf_counter_ns()
    data_seq_len_sum = rf.reduce_sum(data_spatial_dim.dyn_size_ext, axis=data_spatial_dim.dyn_size_ext.dims)
    data_seq_len_sum_secs = data_seq_len_sum.raw_tensor / _batch_size_factor / 100.0
    data_seq_len_max_seqs = data_spatial_dim.get_dim_value() / _batch_size_factor / 100.0
    out_len_longest_sum = rf.reduce_sum(rf.reduce_max(out_spatial_dim.dyn_size_ext, axis=beam_dim), axis=batch_dim)
    print(
        "TIMINGS:",
        ", ".join(
            (
                f"batch size {data.get_batch_dim_tag().get_dim_value()}",
                f"data len max {data_spatial_dim.get_dim_value()} ({data_seq_len_max_seqs:.2f} secs)",
                f"data len sum {data_seq_len_sum.raw_tensor} ({data_seq_len_sum_secs:.2f} secs)",
                f"enc {enc_end_time - start_time} ns",
                f"enc len max {torch.max(enc_olens)}",
                f"dec {search_end_time - enc_end_time} ns",
                f"out len max {out_spatial_dim.get_dim_value()}",
                f"out len longest sum {out_len_longest_sum.raw_tensor}",
            )
        ),
    )

    extra_recog_results = {}
    if out_individual_seq_scores:
        for k, v in out_individual_seq_scores.items():
            extra_recog_results[f"score:{k}"] = rf.convert_to_tensor(
                v.expand(batch_dim.get_dim_value(), beam_dim.get_dim_value()), dims=[batch_dim, beam_dim]
            )

    return seq_targets_t, seq_log_prob_t, extra_recog_results, out_spatial_dim, beam_dim


def get_our_label_scorer_intf(espnet_scorer: BatchScorerInterface, *, enc: torch.Tensor, enc_olens: torch.Tensor):
    """
    :param espnet_scorer:
    :param enc: [Batch,EncTime,Dim]
    :param enc_olens: [Batch] -> [0..EncTime]
    """
    import torch
    import tree

    from espnet.nets.scorer_interface import BatchScorerInterface
    from espnet.nets.scorer_interface import BatchPartialScorerInterface, PartialScorerInterface
    from espnet.nets.scorers.ctc import CTCPrefixScorer
    from espnet2.asr.decoder.transformer_decoder import BaseTransformerDecoder
    from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
    from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

    # Just not implemented otherwise, but also should be the case for all cases we expect here.
    assert isinstance(espnet_scorer, BatchScorerInterface)
    if isinstance(espnet_scorer, PartialScorerInterface):
        assert isinstance(espnet_scorer, BatchPartialScorerInterface)

    from i6_experiments.users.zeyer.decoding.beam_search_torch.interface import (
        LabelScorerIntf,
        StateObjTensorExt,
        StateObjIgnored,
    )
    from i6_experiments.users.zeyer.decoding.beam_search_torch.utils import batch_gather

    class EspnetLabelScorer(LabelScorerIntf):
        """ESPnet label scorer"""

        def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
            """
            :param batch_size:
            :param device:
            :return: state. all tensors are expected to have shape [Batch, Beam=1, ...].
            """
            # ESPnet espnet.nets.batch_beam_search.BatchBeamSearch.init_hyp (slightly simplified):
            #         return self.batchfy(
            #             [
            #                 Hypothesis(
            #                     score=0.0,
            #                     scores={k: 0.0 for k in self.scorers},
            #                     states={k: d.batch_init_state(x) for k, d in self.scorers.items()},
            #                     hs=[],
            #                     yseq=torch.tensor([self.sos], device=x.device),
            #                 )
            #             ]
            #         )

            if isinstance(espnet_scorer, CTCPrefixScorer):
                # espnet.nets.scorers.ctc.CTCPrefixScorer.batch_init_state incorrectly assumes batch_size=1,
                # and is wrong otherwise.
                # It's anyway ugly - we don't store any state here... but we assign the internal attributes.
                from espnet.nets.ctc_prefix_score import CTCPrefixScoreTH

                logp = espnet_scorer.ctc.log_softmax(enc)
                espnet_scorer.impl = CTCPrefixScoreTH(logp, enc_olens, 0, espnet_scorer.eos)
                return None

            # Note: batch_init_state in most cases is just init_state,
            # and init_state in many cases just returns None.
            # E.g. TransformerDecoder and many others handle the case of initial state in batch_score,
            # where state=None is for the initial state.
            state = espnet_scorer.batch_init_state(enc)

            # Note: Need to add beam_dim=1 (not really a problem).
            # Then also need to make sure that the batch_size is correct
            # (unclear, probably not possible in general).
            assert state is None, f"not implemented: {state!r}"
            return None

        def score_and_update_state(
            self,
            *,
            prev_state: Any,
            prev_label: torch.Tensor,
        ) -> Tuple[torch.Tensor, Any]:
            """
            :param prev_state: state of the scorer (decoder). any nested structure.
                all tensors are expected to have shape [Batch, Beam, ...].
            :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
            :return: (scores, state).
                scores: shape [Batch, Beam, Label], log-prob-like scores.
                    Broadcasting is allowed for any of the dims (e.g. think of :class:`LengthRewardScorer`).
                state: all tensors are expected to have shape [Batch, Beam, ...].
            """
            batch_size, beam_size = prev_label.shape

            if prev_state is not None:
                ys, prev_state = prev_state
                ys = torch.concat([ys, prev_label[:, :, None]], dim=2)  # [batch,beam,out_len]
            else:
                ys = prev_label[:, :, None]  # [batch,beam,out_len]
            ys_ = ys.flatten(0, 1)  # [batch*beam,out_len]

            # Convert all [batch,beam,...] tensors to [batch*beam,...].
            def _map(x):
                if x is None:
                    return None
                assert isinstance(x, torch.Tensor) and x.shape[:2] == (batch_size, beam_size)
                return x.flatten(0, 1)

            prev_state = tree.map_structure(_map, prev_state)

            if isinstance(espnet_scorer, CTCPrefixScorer):
                enc_ = None  # not needed
            else:
                enc_ = enc.unsqueeze(1).expand(batch_size, beam_size, *enc.shape[1:]).flatten(0, 1)

            if isinstance(espnet_scorer, CTCPrefixScorer):
                # Unfortunately the CTCPrefixScorer breaks our assumption that the batch dim is the first dim.
                # Thus, we must permute the corresponding entries in the state.
                # Also, the initial state is None, so we need to cover this case as well.
                if prev_state is not None:
                    # 4-tuple. first has batch in dim=2, second has batch in dim=0, third and forth don't have batch?
                    # n_bh = self.batch * n_hyps. snum = odim.
                    # first: r: (self.input_length, 2, n_bh, snum) in func,
                    #   then with select_state resulting in: (in_len, 2, batch * new_n_hyps)
                    #   or: r_prev: (self.input_length, 2, self.batch * n_hyps)
                    # second: log_psi: (n_bh, self.odim) in func,
                    #   then with select_state resulting in: (batch * new_n_hyps, self.odim) ?
                    # third/forth: f_min, f_max: scalars, no batch, only used anyway with att_w, can just set 0 and 1.
                    # we even get a fifth as output: scoring_idmap: but not used.
                    # So, only care about first, second.
                    # Apply the select_state logic here, i.e. espnet.nets.scorers.ctc.CTCPrefixScorer.select_state.
                    r, log_psi = prev_state
                    r: torch.Tensor  # [batch*beam,in_len,2,snum]
                    r = batch_gather(r, indices=prev_label.flatten(), index_dim=3)  # [batch*beam,in_len,2]
                    r = r.permute(1, 2, 0)  # [in_len,2,batch*beam]
                    log_psi: torch.Tensor  # [batch*beam,odim]
                    log_psi = batch_gather(log_psi, indices=prev_label.flatten())  # [batch*beam]
                    log_psi = log_psi[:, None]  # [batch*beam,1]. must broadcast to [batch*beam,odim]
                    prev_state = (r, log_psi, 0, 1)

                # Inline espnet.nets.scorers.ctc.CTCPrefixScorer.batch_score_partial,
                # as we already have it batched.
                scores, states = espnet_scorer.impl(ys_, prev_state)
                # scores: (n_bh, vocab)
                r, log_psi = states[:2]
                r: torch.Tensor  # [in_len,2,batch*beam,snum]
                r = r.permute(2, 0, 1, 3)  # [batch*beam,in_len,2,snum]
                states = (r, log_psi)

            elif isinstance(espnet_scorer, BaseTransformerDecoder):
                # Inlined and adapted espnet2.asr.decoder.transformer_decoder.BaseTransformerDecoder.batch_score.
                # We avoid the state transformation here, as we anyway have it already in the right way.
                ys_mask = subsequent_mask(ys_.size(-1), device=enc.device).unsqueeze(0)
                enc_olens_ = enc_olens.unsqueeze(1).expand(batch_size, beam_size).flatten(0, 1)  # [batch*beam]
                enc_mask = (~make_pad_mask(enc_olens_, maxlen=enc_.size(1)))[:, None, :].to(
                    enc_.device
                )  # [batch*beam,1,time]
                scores, states = espnet_scorer.forward_one_step(ys_, ys_mask, enc_, enc_mask, cache=prev_state)

            else:
                # Note: No select_state needed. This is already done by the outer logic.
                # prev_state_ls must be list over batch entries, thus convert out prev_state.
                prev_state_ls = []
                for batch_idx in range(batch_size):
                    for beam_idx in range(beam_size):

                        def _map(x):
                            assert isinstance(x, torch.Tensor) and x.shape[:2] == (batch_size, beam_size)
                            return x[batch_idx, beam_idx]

                        prev_state_ls.append(tree.map_structure(_map, prev_state))

                # WARNING: This is without a mask for enc.
                if isinstance(espnet_scorer, BatchPartialScorerInterface):
                    scores, states_ls = espnet_scorer.batch_score_partial(ys_, None, prev_state_ls, enc_)
                else:
                    scores, states_ls = espnet_scorer.batch_score(ys_, prev_state_ls, enc_)

                # We get back a list over batch entries, stack all tensors to single state object.
                def _map(*xs):
                    assert all(isinstance(x, torch.Tensor) for x in xs)
                    return torch.stack(xs)

                states = tree.map_structure(_map, *states_ls)

            # Convert all [batch*beam,...] tensors to [batch,beam,...].
            def _map(x):
                assert isinstance(x, torch.Tensor) and x.shape[:1] == (batch_size * beam_size,)
                return x.unflatten(0, (batch_size, beam_size))

            scores = _map(scores)
            states = tree.map_structure(_map, states)
            return scores, (ys, states)

    return EspnetLabelScorer()


# RecogDef API
model_recog_our: RecogDef[ESPnetASRModel]
model_recog_our.output_with_beam = True
model_recog_our.output_blank_label = "<s>"
model_recog_our.batch_size_dependent = False
