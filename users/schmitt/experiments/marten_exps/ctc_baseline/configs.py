"""
Configurations, i.e. RETURNN settings,
shared across several setups here in this directory.
"""

from __future__ import annotations
from typing import Any, Dict, List
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
    pos_emb_dropout=0.1,  # WARNING: when the self-att or conformer opts are custom, this is ignored! also for CTC!
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
# ~1245 /steps/ep
_lrlin_oclr_steps_by_bs_nep = {
    (15, 1): [560, 1_120, 1_245],  # total steps after 1 epoch: ~1245
    (15, 9): [5_000, 10_100, 11_200],  # total steps after 9 epochs: ~11k
    (15, 10): [5_600, 11_200, 12_450],  # total steps after 10 epochs: ~12k
    (15, 18): [10_000, 20_200, 22_400],  # total steps after 18 epochs: ~22k
    (15, 20): [11_200, 22_400, 24_900],  # total steps after 20 epochs: ~25k
    (15, 45): [25_200, 50_400, 56_000],  # total steps after 45 epochs: ~56k
    (15, 50): [27_900, 55_800, 62_000],  # total steps after 50 epochs: ~62k
    (15, 56): [31_500, 63_000, 70_000],  # total steps after 113 epochs: ~140k
    (15, 75): [41_850, 83_700, 93_000],  # total steps after 75 epochs: ~93k
    (15, 113): [63_000, 126_000, 140_000],  # total steps after 113 epochs: ~140k
    (15, 125): [70_000, 140_000, 155_000],  # total steps after 125 epochs: ~155k
    (15, 225): [126_000, 252_000, 280_000],  # total steps after 225 epochs: ~281k
    (15, 450): [253_000, 506_000, 562_000],  # total steps after 450 epochs: ~562k
    (15, 500): [295_000, 590_000, 652_000],  # total steps after 500 epochs: ~652k
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


# Just specify avg num steps per (sub)epoch (partition epoch 20) for batch_size // 1000.
# (Assumes max_seqs 200, spm10k, max_seq_len 75, multi-GPU 4.)
# Estimated via some existing log, or alternatively via:
# tools/calc-avg-num-train-steps-per-ep-from-seq-lens.py \
#   output/datasets/LibriSpeech/seq_len_audio-features=raw-sampleRate=16000-peakNormalization=True.txt \
#   --seq_lens_file_for_max_seq_len \
#     output/datasets/LibriSpeech/seq_len_target-spm10k-class=SamplingBytePairEncoding-breadthProb=0.001.txt \
#   --partition_epoch 20 --seq_ordering "laplace:.1000" \
#   --max_seq_len 75 --multi_gpu 4 --num_epochs 20 \
#   --max_seqs 200 --batch_size (N * 1000 * 160)
# Then using p10 (10% percentile) from the output.
# Using some lower number than the real one should be safe.
# It means we might reach the end of the LR schedule slightly earlier than in the real case.
_tot_num_steps_by_bs = {
    5: 3898,
    8: 2485,
    10: 1973,
    15: 1303,
    20: 976,
    30: 652,
    40: 491,
}


def _get_cfg_lrlin_oclr_by_bs_nep_v2(bs_feat: int, n_ep: int, *, peak_lr: float = 1e-3) -> Dict[str, Any]:
    """
    :param bs_feat: batch size for features (not including _batch_size_factor)
    :param n_ep: num epochs
    """
    tot_num_steps = _tot_num_steps_by_bs[bs_feat // 1000] * n_ep
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


def _get_cfg_lrlin_oclr_by_bs_nep_v4(
    n_ep: int,
    *,
    base_lr: float = 1.0,
    peak_lr: float = 1e-3,
    low_lr: float = 1e-5,
    lowest_lr: float = 1e-6,
    step_peak_fraction: float = 0.45,
    step_finetune_fraction: float = 0.9,
) -> Dict[str, Any]:
    """
    :param n_ep: num epochs
    """
    return {
        "__num_epochs": n_ep,
        "learning_rate": base_lr,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        "learning_rate_piecewise_by_epoch_continuous": True,
        "learning_rate_piecewise_steps": [step_peak_fraction * n_ep, step_finetune_fraction * n_ep, n_ep],
        "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
    }


def _get_cfg_lrlin_oclr_by_bs_nep_v5(
    n_ep: int,
    *,
    base_lr: float = 1.0,
    const_lr: float = 1e-3,
    low_lr: float = 1e-5,
    # lowest_lr: float = 1e-6,
    step_peak_fraction: float = 0.45,
    step_const_fraction: float = 0.9,
) -> Dict[str, Any]:
    """
    This warms up, stays constant for a while, and then decays.
    :param n_ep: num epochs
    """
    return {
        "__num_epochs": n_ep,
        "learning_rate": base_lr,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        "learning_rate_piecewise_by_epoch_continuous": True,
        "learning_rate_piecewise_steps": [step_peak_fraction * n_ep, step_const_fraction * n_ep, n_ep],
        "learning_rate_piecewise_values": [low_lr, const_lr, const_lr, low_lr],
    }


def generic_piecewise_linear(
    name: str,
    global_train_step: int,
) -> float:
    """
    Piecewise linear
    """
    from returnn.config import get_global_config
    from returnn.util.math import PiecewiseLinear

    config = get_global_config()
    f = config.typed_dict.get(f"_{name}_piecewise_cache")
    if f is None:
        steps = config.float_list(f"{name}_piecewise_steps")
        values = config.float_list(f"{name}_piecewise_values")
        assert len(steps) + 1 == len(values)
        last_step = 0
        for i, step in enumerate(steps):
            assert step > last_step
            last_step = step
        f = PiecewiseLinear(dict(zip([0] + list(steps), values)))
        config.typed_dict[f"_{name}_piecewise_cache"] = f

    return f(global_train_step + 1)


def _get_cfg_generic_piecewise_linear(
        name: str,
        steps: List[int],
        values: List[float],
):
    """
    This warms up, stays constant for a while, and then decays.
    """
    return {
        name: generic_piecewise_linear,
        f"{name}_piecewise_steps": steps,
        f"{name}_piecewise_values": values,
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
