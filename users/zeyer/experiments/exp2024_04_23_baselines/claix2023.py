"""
Config for RWTH IPC CLAIX-2023 cluster experiments.
"""

from __future__ import annotations
from typing import Dict, Any
from .configs import config_24gb_v6
from .aed import train_exp, dict_update_deep, speed_pert_librosa_config, _batch_size_factor, dyn_lr_piecewise_linear


def py():
    # TODO fix LR schedule
    # TODO only 50% compute time... need to increase multiproc dataset workers
    train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_4-lrlin1e_5_295kWrong-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_dummy(200_000, 2000),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    )


# https://help.itc.rwth-aachen.de/service/rhr4fjjutttf/article/9108f4a6f43c40a3a168919afd36839d/
# TODO check weight decay...
config_96gb_bf16_accgrad1 = dict_update_deep(
    config_24gb_v6,
    {
        "__gpu_mem": 96,
        "__cpu_rqmt": 24,  # the whole c23g node has 96 CPUs, and 4 GPUs
        "__mem_rqmt": 100,  # the whole node should have more than 500GB
        "accum_grad_multiple_step": 1,  # per single GPU
    },
)


def _get_cfg_lrlin_oclr_by_bs_nep_dummy(bs_feat: int, n_ep: int, *, peak_lr: float = 1e-3) -> Dict[str, Any]:
    """
    :param bs_feat: batch size for features (not including _batch_size_factor)
    :param n_ep: num epochs
    """
    tot_num_steps = 1_000_000  # dummy value
    steps = [tot_num_steps * 0.45, tot_num_steps * 0.9, tot_num_steps]
    steps = [int(s) for s in steps]

    return {
        "__num_epochs": n_ep,
        "batch_size": bs_feat * _batch_size_factor,
        "learning_rate": 1.0,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        # If the dict has no entry for the bs_feat,n_ep combination, see above.
        "learning_rate_piecewise_steps": steps,
        "learning_rate_piecewise_values": [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
    }
