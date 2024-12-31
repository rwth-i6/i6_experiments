"""
Config for RWTH IPC CLAIX-2023 cluster experiments for AED models
"""

from __future__ import annotations

from typing import Any, Dict

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from .configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from .aed import train_exp as aed_train_exp

from i6_experiments.users.zeyer.train_v4 import train, ModelDefWithCfg

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerPositionwiseFeedForward


def py():
    # Note: From the baseline experiment (optimized for 4x11GB GPUs with float32),
    # we made the following changes (partly taking our 1x24GB GPU settings into account)
    # for the H100 GPU with 96GB memory (nodes c23g in the CLAIX-2023 cluster):
    # - __gpu_mem = 96
    # - batch size was increased to 200k (takes about 60-70GB of GPU memory)
    # - bf16 again (AMP) (also checking pure bf16 now...)
    # - (grad accum 1 (no change actually; and obviously, batch size is already large enough...))
    # - (LR scheduling now based on seq_idx (this is not really related to the new GPU, but just simplifies things))
    # - (weight decay = 1e-2 still, no change so far, but checking now...)
    # - partition epoch to 1 (dataset_train_opts.train_epoch_split=1)
    #   (because the GPU is so fast that it trains a single epoch in 20mins;
    #    otherwise, eval is just too often, takes too much time)
    # - more workers for data loading (__multi_proc_dataset_opts.num_workers=25) (check computation time in log!)
    # - __cpu_rqmt = 24 (the whole c23g node has 96 CPUs, and 4 GPUs)
    # - __mem_rqmt = 100 (the whole node should have more than 500GB)
    # Due to the larger batch size, we have less steps per epoch. With bs200k, it is 2016 steps per full epoch.
    # Baseline: SubEp21 start: 21660, SubEp40 end: 46627, thus 24967 steps per full epoch.
    # (But first full epoch has a bit less due to filtering: 21660)

    # Baseline: v6-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-speedpertV2-spm10k-spmSample07
    # {"dev-clean": 2.35, "dev-other": 4.98, "test-clean": 2.21, "test-other": 5.49}
    # Final 'devtrain_loss_ce': 0.11065730461399945, 'devtrain_loss_fer': 0.006211603513944155,
    # -----

    # Note: epoch filtering is wrong, should not do that for 5 full epochs...
    # {"dev-clean": 2.36, "dev-other": 5.35, "test-clean": 2.4, "test-other": 5.72}
    # Final 'devtrain_loss_ce': 0.11504180265534825, 'devtrain_loss_fer': 0.005836691916394713,
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1},
    )

    # No curriculum learning (epoch filtering) (-> train_epoch_wise_filter=None)
    # {"dev-clean": 2.4, "dev-other": 5.22, "test-clean": 2.5, "test-other": 5.55}
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # TODO instead of having SpecAugment step-based, we can also make this dependent on the continuous epoch

    # SpecAugment adapted
    # {"dev-clean": 2.32, "dev-other": 5.24, "test-clean": 2.41, "test-other": 5.66}
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "specaugment_steps": (500, 1_000, 2_000),
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # Test default_float_dtype="bfloat16" (bfloat16A) instead of AMP.
    # (96gb-bf16A-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07)
    # Consumes about 40GB of GPU memory.
    # {"dev-clean": 4.37, "dev-other": 10.07, "test-clean": 4.39, "test-other": 10.32}
    # TODO what's the problem?

    # bfloat16A with larger batch.
    # Batch size 400k: OOM after some epochs.
    # bfloat16A with larger batch V2.
    # {"dev-clean": 4.28, "dev-other": 10.35, "test-clean": 4.22, "test-other": 10.08}
    aed_train_exp(
        f"96gb-bf16A-bs300k-bsSeq400-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            "torch_amp": None,
            "default_float_dtype": "bfloat16",
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(300_000, 100, batch_size_factor=_batch_size_factor),
            "max_seqs": 400,
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "specaugment_steps": (500, 1_000, 2_000),
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid oom
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
    )

    # Higher peak LR (96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-lr1e_2-noCrl-speedpertV2-spm10k-spmSample07)
    # {"dev-clean": 5.62, "dev-other": 12.41, "test-clean": 5.56, "test-other": 12.88} -> bad

    # More weight decay
    # {"dev-clean": 2.4, "dev-other": 5.26, "test-clean": 2.54, "test-other": 5.63}
    # -> unclear, maybe slightly better?
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd5e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 5e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # Lion optimizer (https://arxiv.org/abs/2302.06675, https://github.com/lucidrains/lion-pytorch/)
    # {"dev-clean": 2.37, "dev-other": 5.52, "test-clean": 2.5, "test-other": 5.68}
    # (Baseline with Adam: "dev-other": 5.22)
    # TODO maybe needs more tuning? different wd, lr
    lion_lr_factor = 0.3
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd0.0333-lrlinEpCont-lr3e_4-optLion-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                200_000,
                100,
                peak_lr=1e-3 * lion_lr_factor,
                low_lr=1e-5 * lion_lr_factor,
                lowest_lr=1e-6 * lion_lr_factor,
                batch_size_factor=_batch_size_factor,
            ),
            "optimizer.class": "returnn.torch.optim.lion.Lion",
            "optimizer.weight_decay": 1e-2 / lion_lr_factor,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        config_deletes=["optimizer.epsilon"],  # no eps in Lion
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # lossSeqNorm.
    # Baseline: {"dev-clean": 2.4, "dev-other": 5.22, "test-clean": 2.5, "test-other": 5.55}
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07-lossSeqNorm",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "use_normalized_loss": "seqs",
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # TODO larger batch?
    # TODO shuffle batches
    # TODO loss Seq norm
    # TODO log_grad_norm
    # TODO better grad clip?
    # TODO init out layer to all zeros
