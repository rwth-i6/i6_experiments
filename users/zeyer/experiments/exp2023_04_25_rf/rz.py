"""
Experiments in RWTH ITC
"""

from __future__ import annotations
from .configs import (
    config_24gb_v4,
    config_24gb_v6,
    _batch_size_factor,
    _get_cfg_lrlin_oclr_by_bs_nep,
)
from .conformer_import_moh_att_2023_06_30 import train_exp as train_exp_aed_lstm
from .aed import train_exp as train_exp_aed_trafo
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config


# run directly via `sis m ...`
def py():
    from i6_experiments.users.zeyer import tools_paths

    tools_paths.monkey_patch_i6_core()

    train_exp_aed_lstm(
        "v4-f32-bs20k-accgrad4",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 4,
        },
        gpu_mem=16,
    )
    train_exp_aed_lstm(
        "v4-f32-bs20k-accgrad4-mgpu2",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 4,
            "torch_distributed": {},
        },
        gpu_mem=16,
        num_processes=2,
    )
    train_exp_aed_lstm(
        "v4-f32-bs20k-accgrad2",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
        },
        gpu_mem=16,
    )
    train_exp_aed_lstm(
        "v4-f32-bs20k-accgrad2-mgpu2-adam",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
            "torch_distributed": {},
            "optimizer.class": "adam",
        },
        gpu_mem=16,
        num_processes=2,
    )
    train_exp_aed_lstm(  # 6.31
        "v6-f32-bs20k-accgrad2-mgpu2-wd1e_4",
        config_v6_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
            "torch_distributed": {},
            "optimizer.weight_decay": 1e-4,
        },
        gpu_mem=16,
        num_processes=2,
    )

    train_exp_aed_trafo(
        "v6-bhv20-f32-bs20k-accgrad1-mgpu2-wd1e_2-lrlin1e_5_438k-speedpertV2",
        config_v6_f32_bs20k,
        model_config={"behavior_version": 20},
        config_updates={
            "__gpu_mem": 16,
            "__num_processes": 2,
            **_get_cfg_lrlin_oclr_by_bs_nep(20_000, 1000),
            "accum_grad_multiple_step": 1,
            "torch_distributed": {},
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
    )

    # currently NVLINK seems broken, always NCCL error... or I could fallback to slower communication...
    # train_exp(
    #     "v4-f32-mgpu16",
    #     config_v4_f32,
    #     config_updates={"torch_distributed": {}},
    #     gpu_mem=32,
    #     num_processes=16,
    # )
    # train_exp(
    #     "v4-f32-mgpu2",
    #     config_v4_f32,
    #     config_updates={"torch_distributed": {}},
    #     gpu_mem=32,
    #     num_processes=2,
    # )
    # train_exp(  # 6.12
    #     "v6-f32-accgrad1-mgpu4-wd1e_4-lrlin1e_5_111k",
    #     config_v6_f32,
    #     config_updates={
    #         "accum_grad_multiple_step": 1,
    #         "torch_distributed": {},
    #         "optimizer.weight_decay": 1e-4,
    #         # bs15k steps/epoch: ~493, total num of steps for 500 epochs: ~247k
    #         "learning_rate_piecewise_steps": [111_000, 222_000, 247_000],
    #         "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    #     },
    #     gpu_mem=32,
    #     num_processes=4,
    #     num_epochs=500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    # )


config_v4_f32 = dict_update_deep(config_24gb_v4, None, ["torch_amp"])
config_v4_f32_bs20k = dict_update_deep(
    config_v4_f32,
    {
        "batch_size": 20_000 * _batch_size_factor,  # 30k gives OOM on the 16GB GPU
    },
)
config_v6_f32 = dict_update_deep(config_24gb_v6, None, ["torch_amp"])
config_v6_f32_bs20k = dict_update_deep(
    config_v6_f32,
    {
        "batch_size": 20_000 * _batch_size_factor,  # 30k gives OOM on the 16GB GPU
    },
)
