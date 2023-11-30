"""
Experiments in RWTH ITC
"""

from .conformer_import_moh_att_2023_06_30 import train_exp, config_24gb_v4, _batch_size_factor


# run directly via `sis m ...`
def py():
    train_exp(
        "base-24gb-v4-f32-bs20k-accgrad4",
        config_24gb_v4,
        config_updates={
            "batch_size": 20_000 * _batch_size_factor,  # 30k gives OOM on the 16GB GPU
            "accum_grad_multiple_step": 4,
        },
        config_deletes=["torch_amp"],
        gpu_mem=16,
    )
    train_exp(
        "base-24gb-v4-f32-bs20k-accgrad4-mgpu2",
        config_24gb_v4,
        config_updates={
            "batch_size": 20_000 * _batch_size_factor,
            "accum_grad_multiple_step": 4,
            "torch_distributed": {},
        },
        config_deletes=["torch_amp"],
        gpu_mem=16,
        num_processes=2,
    )
    train_exp(
        "base-24gb-v4-f32-bs20k-accgrad2",
        config_24gb_v4,
        config_updates={
            "batch_size": 20_000 * _batch_size_factor,
            "accum_grad_multiple_step": 2,
        },
        config_deletes=["torch_amp"],
        gpu_mem=16,
    )
    train_exp(
        "base-24gb-v4-f32-mgpu16",
        config_24gb_v4,
        config_updates={"torch_distributed": {}},
        config_deletes=["torch_amp"],
        gpu_mem=32,
        num_processes=16,
    )
    train_exp(
        "base-24gb-v4-f32-mgpu2",
        config_24gb_v4,
        config_updates={"torch_distributed": {}},
        config_deletes=["torch_amp"],
        gpu_mem=32,
        num_processes=2,
    )
