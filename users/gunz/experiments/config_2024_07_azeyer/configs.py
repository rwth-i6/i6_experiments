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

config_24gb_v6 = dict(
    torch_amp="bfloat16",
    grad_scaler=None,
    batching="laplace:.1000",
    max_seqs=200,
    max_seq_length_default_target=75,
    specaugment_steps=(5_000, 15_000, 25_000),
    gradient_clip_global_norm=5.0,
    optimizer={
        "class": "adamw",
        "epsilon": 1e-16,
        "weight_decay": 1e-6,
        "weight_decay_modules_blacklist": [
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
        ],
    },
    accum_grad_multiple_step=2,
    learning_rate=1e-3,
    dynamic_learning_rate=dyn_lr_lin_warmup_invsqrt_decay,
    learning_rate_warmup_steps=20_000,
    learning_rate_invsqrt_norm=20_000,
    aux_loss_layers=[4, 8],
    pos_emb_dropout=0.1,
    rf_att_dropout_broadcast=False,
)

post_config = dict(
    cleanup_old_models=dict(keep_last_n=5),
    torch_dataloader_opts=dict(num_workers=1),
)


# By batch size (in k) and num (sub)epochs.
# 500 subepochs is usually for multi-GPU with 4 GPUs,
# i.e. the same as single-GPU 2000 subepochs.
# If the dict is missing some entry,
# unfortunately there is currently no good automatic way to get the number.
# I need to log at the stats of some setup with this batch size.
# I just run some setup with some arbitrary LR scheduling (calling it "wrongLr"),
# or maybe with sqrt-decay, and then look at the stats (steps/ep, or total num steps),
# and give some estimates for the steps here, i.e. 45%, 90%, almost 100%,
# making sure the last number is slightly below the real total number of steps.
_lrlin_oclr_steps_by_bs_nep = {
    (8, 125): [139_000, 279_000, 310_000],  # ~2485steps/ep, 125 eps -> 310k steps in total
    (8, 250): [279_000, 558_000, 621_000],  # ~2485steps/ep, 250 eps -> 621k steps in total
    (8, 500): [558_000, 1_117_000, 1_242_000],  # ~2485steps/ep, 500 eps -> 1.242k steps in total
    (10, 500): [443_000, 887_000, 986_000],  # ~1973 steps/epoch, total steps after 500 epochs: ~986k
    (15, 400): [234_000, 469_000, 521_000],  # total steps after 400 epochs: ~521k
    (15, 500): [295_000, 590_000, 652_000],  # total steps after 500 epochs: ~652k
    (15, 600): [352_000, 704_000, 782_000],  # total steps after 600 epochs: ~782k
    (20, 1000): [438_000, 877_000, 974_000],  # total steps after 1000 epochs: 974.953
    (20, 2000): [878_000, 1_757_000, 1_952_000],  # total steps after 2000 epochs: 1.952.394
    (30, 2000): [587_000, 1_174_000, 1_305_000],  # total steps after 2000 epochs: 1.305.182
    (40, 2000): [450_000, 900_000, 982_000],  # total steps after 2000 epochs: 982.312
}


def _get_cfg_lrlin_oclr_by_bs_nep(bs_feat: int, n_ep: int, *, peak_lr: float = 1e-3) -> Dict[str, Any]:
    """
    :param bs_feat: batch size for features (not including _batch_size_factor)
    :param n_ep: num epochs
    """
    return {
        "__num_epochs": n_ep,
        "batch_size": bs_feat * _batch_size_factor,
        "learning_rate": 1.0,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        # If the dict has no entry for the bs_feat,n_ep combination, see above.
        "learning_rate_piecewise_steps": _lrlin_oclr_steps_by_bs_nep[(bs_feat // 1000, n_ep)],
        "learning_rate_piecewise_values": [peak_lr * 1e-2, peak_lr, peak_lr * 1e-2, peak_lr * 1e-3],
    }


# combine this for example with _get_cfg_lrlin_oclr_by_bs_nep(15_000, 500)
config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4 = dict_update_deep(
    config_24gb_v6,
    {
        "__gpu_mem": 11,
        "accum_grad_multiple_step": 1,  # per single GPU
        "torch_distributed": {"reduce_type": "param", "param_sync_step": 100},  # multi-GPU
        "__num_processes": 4,  # multi-GPU
        "optimizer.weight_decay": 1e-4,
    },
    [
        "torch_amp",  # f32
    ],
)


def test():
    from ..exp2023_04_25_rf import configs as configs_old

    config_old = configs_old.config_11gb_v6_f32_bs15k_accgrad1_mgpu4_pavg100_wd1e_4_lrlin1e_5_295k
    config_new = {**config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4, **_get_cfg_lrlin_oclr_by_bs_nep(15_000, 500)}
    assert config_old == config_new
