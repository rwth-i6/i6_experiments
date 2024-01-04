"""
Configurations, i.e. RETURNN settings,
shared across several setups here in this directory.
"""

from __future__ import annotations
from typing import Any, Dict
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.lr_schedules.lin_warmup_invsqrt_decay import dyn_lr_lin_warmup_invsqrt_decay
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear


_batch_size_factor = 160

config = dict(
    batching="laplace:.1000",
    batch_size=15_000 * _batch_size_factor,
    max_seqs=200,
    max_seq_length_default_target=75,
    specaugment_steps=(10_000, 20_000, 40_000),
    # gradient_clip=0,
    # gradient_clip_global_norm = 1.0
    optimizer={
        "class": "adamw",
        "epsilon": 1e-8,
        "weight_decay": 1e-6,
    },
    accum_grad_multiple_step=4,
    # gradient_noise=0.0,
    learning_rate=2.5e-3,
    dynamic_learning_rate=dyn_lr_lin_warmup_invsqrt_decay,
    learning_rate_warmup_steps=40_000,
    learning_rate_invsqrt_norm=40_000,
    aux_loss_layers=[4, 8],
)
post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
    torch_dataloader_opts=dict(num_workers=1),
)

config_24gb = config.copy()
config_24gb.update(
    dict(
        torch_amp="bfloat16",
        batch_size=40_000 * _batch_size_factor,
        accum_grad_multiple_step=2,
        learning_rate=2e-3,
        learning_rate_warmup_steps=20_000,
        learning_rate_invsqrt_norm=20_000,
        specaugment_steps=(5_000, 15_000, 25_000),
    )
)
# base-24gb (using config_24gb): converged, but stagnated, and hiccups

config_24gb_v2 = dict_update_deep(
    config_24gb,
    {
        "optimizer.epsilon": 1e-16,
        "specaugment_num_spatial_mask_factor": 200,
        "specaugment_max_consecutive_feature_dims": 10,
    },
)

config_24gb_v3 = config_24gb_v2.copy()
config_24gb_v3.update(
    dict(
        learning_rate=2.5e-3,
        grad_scaler=None,
        gradient_clip_global_norm=5.0,
    )
)

config_24gb_v4 = dict_update_deep(
    config_24gb_v3,
    {
        "learning_rate": 1e-3,
        "optimizer.weight_decay_modules_blacklist": [
            "rf.BatchNorm",  # unclear if really good
            "rf.LayerNorm",  # unclear if really good
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
    },
)

config_24gb_v5 = dict_update_deep(
    config_24gb_v4,
    {
        "pretrain_opts": {  # pretrain
            "steps": [(8 * 500, {"num_layers": 2}), (4 * 500, {"num_layers": 4}), (4 * 500, {"num_layers": 8})]
        },
        "pos_emb_dropout": 0.1,  # posdrop01
        "optimizer.weight_decay_modules_blacklist": [  # wdblacklist2
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
        "rf_att_dropout_broadcast": False,  # attdropfixbc
    },
    [
        # specaugorig
        "specaugment_num_spatial_mask_factor",
        "specaugment_max_consecutive_feature_dims",
    ],
)

config_24gb_v6 = dict_update_deep(config_24gb_v5, None, ["pretrain_opts"])


# By batch size (in k) and num (sub)epochs.
_lrlin_oclr_steps_by_bs_nep = {
    (8, 500): [558_000, 1_117_000, 1_242_000],  # ~2485steps/ep, 500 eps -> 1.242k steps in total
    (15, 500): [295_000, 590_000, 652_000],  # total steps after 500 epochs: ~652k
    (40, 2000): [450_000, 900_000, 982_000],  # total steps after 2000 epochs: 982.312
}


def _get_cfg_lrlin_oclr_by_bs_nep(bs_feat: int, n_ep: int) -> Dict[str, Any]:
    """
    :param bs_feat: batch size for features (not including _batch_size_factor)
    :param n_ep: num epochs
    """
    return {
        "batch_size": bs_feat * _batch_size_factor,
        "learning_rate": 1.0,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        "learning_rate_piecewise_steps": _lrlin_oclr_steps_by_bs_nep[(bs_feat // 1000, n_ep)],
        "learning_rate_piecewise_values": [1e-5, 1e-3, 1e-5, 1e-6],
    }


_cfg_lrlin1e_5_295k = _get_cfg_lrlin_oclr_by_bs_nep(15_000, 500)

config_11gb_v6_f32_bs15k_accgrad4_mgpu = dict_update_deep(
    config_24gb_v6,
    {
        "batch_size": 15_000 * _batch_size_factor,  # ~1305 steps/epoch
        "accum_grad_multiple_step": 4,  # per single GPU
        "torch_distributed": {},  # multi-GPU
        "__gpu_mem": 11,
    },
    [
        "torch_amp",  # f32
    ],
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad4_mgpu,
    {
        "accum_grad_multiple_step": 1,
    },
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad1_mgpu,
    {
        "optimizer.weight_decay": 1e-4,
        **_cfg_lrlin1e_5_295k,
        "__num_processes": 4,  # multi-GPU
        "__num_epochs": 500,  # because of multi-GPU, 1 subepoch here is like 4 subepochs in single-GPU
    },
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad1_mgpu4_wd1e_4_lrlin1e_5_295k,
    {
        "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},  # multi-GPU
    },
)
config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_100k = dict_update_deep(
    config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k,
    {
        # total steps after 500 epochs: ~652k
        "learning_rate_piecewise_steps": [100_000, 590_000, 652_000],
    },
)

# TODO lrlin
# TODO lr09e_3
