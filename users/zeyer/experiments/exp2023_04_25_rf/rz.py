"""
Experiments in RWTH ITC
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence
from .conformer_import_moh_att_2023_06_30 import (
    train_exp as _train_exp,
    config_24gb_v4,
    config_24gb_v6,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep, dict_update_delete_deep

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_with_checkpoints import ModelWithCheckpoints


# run directly via `sis m ...`
def py():
    train_exp(
        "v4-f32-bs20k-accgrad4",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 4,
        },
    )
    train_exp(
        "v4-f32-bs20k-accgrad4-mgpu2",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 4,
            "torch_distributed": {},
        },
        num_processes=2,
    )
    train_exp(
        "v4-f32-bs20k-accgrad2",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
        },
    )
    train_exp(
        "v4-f32-bs20k-accgrad2-mgpu2-adam",
        config_v4_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
            "torch_distributed": {},
            "optimizer.class": "adam",
        },
        num_processes=2,
    )
    train_exp(
        "v6-f32-bs20k-accgrad2-mgpu2-wd1e-4",
        config_v6_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 2,
            "torch_distributed": {},
            "optimizer.weight_decay": 1e-4,
        },
        num_processes=2,
    )
    train_exp(
        "v6-f32-bs20k-accgrad1-mgpu4-wd1e-4-lrlin1e_5_220k",
        config_v6_f32_bs20k,
        config_updates={
            "accum_grad_multiple_step": 1,
            "torch_distributed": {},
            "optimizer.weight_decay": 1e-4,
            # bs15k steps/epoch: ~980, total num of steps for 500 epochs: ~490k
            "learning_rate_piecewise_steps": [220_000, 440_000, 489_000],
            "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
        },
        num_processes=4,
        num_epochs=500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    )
    train_exp(
        "v4-f32-mgpu16",
        config_v4_f32,
        config_updates={"torch_distributed": {}},
        gpu_mem=32,
        num_processes=16,
    )
    train_exp(
        "v4-f32-mgpu2",
        config_v4_f32,
        config_updates={"torch_distributed": {}},
        gpu_mem=32,
        num_processes=2,
    )


config_v4_f32 = dict_update_delete_deep(config_24gb_v4, ["torch_amp"])
config_v4_f32_bs20k = dict_update_deep(
    config_v4_f32,
    {
        "batch_size": 20_000 * _batch_size_factor,  # 30k gives OOM on the 16GB GPU"
    },
)
config_v6_f32 = dict_update_delete_deep(config_24gb_v6, ["torch_amp"])
config_v6_f32_bs20k = dict_update_deep(
    config_v6_f32,
    {
        "batch_size": 20_000 * _batch_size_factor,  # 30k gives OOM on the 16GB GPU"
    },
)


def train_exp(
    name: str,
    config: Dict[str, Any],
    *,
    config_updates: Optional[Dict[str, Any]] = None,
    config_deletes: Optional[Sequence[str]] = None,
    num_epochs: int = 2000,
    gpu_mem: Optional[int] = 16,
    num_processes: Optional[int] = None,
    **kwargs,
) -> ModelWithCheckpoints:
    return _train_exp(
        name,
        config,
        config_updates=config_updates,
        config_deletes=config_deletes,
        num_epochs=num_epochs,
        gpu_mem=gpu_mem,
        num_processes=num_processes,
        **kwargs,
    )
