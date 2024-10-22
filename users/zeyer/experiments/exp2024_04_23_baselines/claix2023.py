"""
Config for RWTH IPC CLAIX-2023 cluster experiments.
"""

from __future__ import annotations

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

from .configs import config_24gb_v6, _get_cfg_lrlin_oclr_by_bs_nep_v3
from .aed import train_exp as aed_train_exp
from .ctc import train_exp as ctc_train_exp

from i6_experiments.users.zeyer.experiments.exp2024_10_16_consistency_reg_ctc import cr_ctc_training

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerPositionwiseFeedForward


def py():
    # Note: From the baseline experiment (optimized for 4x11GB GPUs with float32),
    # we made the following changes (partly taking our 1x24GB GPU settings into account)
    # for the H100 GPU with 96GB memory (nodes c23g in the CLAIX-2023 cluster):
    # - __gpu_mem = 96
    # - batch size was increased to 200k (takes about 60-70GB of GPU memory)
    # - bf16 again (AMP)
    # - (grad accum 1 (no change actually; and obviously, batch size is already large enough...))
    # - (LR scheduling now based on seq_idx (this is not really related to the new GPU, but just simplifies things))
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
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100),
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
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # SpecAugment adapted
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-noCrl-specAug2k-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100),
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

    # Higher peak LR
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd1e_2-lrlinEpCont-lr1e_2-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, peak_lr=1e-2),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
        },
        post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    )

    # More weight decay
    aed_train_exp(
        f"96gb-bf16-bs200k-accgrad1-wd5e_2-lrlinEpCont-noCrl-speedpertV2-spm10k-spmSample07",
        config_96gb_bf16_accgrad1,
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100),
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

    for opts in [
        # Baseline (n12) has {"dev-clean": 2.35, "dev-other": 5.65, "test-clean": 2.66, "test-other": 5.94}.
        # CLAIX baseline: {"dev-clean": 2.54, "dev-other": 5.93, "test-clean": 2.68, "test-other": 6.27}
        # CLAIX CR: {"dev-clean": 2.49, "dev-other": 5.99, "test-clean": 2.68, "test-other": 6.05}
        # v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        {
            "name": "v6-relPosAttDef-noBias-aedLoss-bhv20-96gb-bf16-bs200k-accgrad1-wd1e_2"
            "-lrlinEpCont-featBN-speedpertV2-spm10k-bpeSample001",
            "num_enc_layers": 12,
            "batch_size": 200_000,
        },
        # Baseline (n16) has {"dev-clean": 2.26, "dev-other": 5.44, "test-clean": 2.5, "test-other": 5.62}.
        # v6-n16-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs10k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        # {"num_enc_layers": 16, "batch_size": 10_000},
    ]:
        for cr_ctc in [None, {"cr_loss_scale": 0.2}]:
            # TODO also adapt specaug for CR...
            use_cr_ctc = cr_ctc is not None
            if use_cr_ctc:
                name = f"n{opts['num_enc_layers']}"
                name += f"-crLoss{cr_ctc['cr_loss_scale']}"
            else:
                name = opts["name"]
            ctc_train_exp(
                name,
                config_96gb_bf16_accgrad1,
                train_def=cr_ctc_training if use_cr_ctc else None,
                model_config={
                    "enc_conformer_layer": rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                    "feature_batch_norm": True,
                    "num_enc_layers": opts["num_enc_layers"],
                },
                config_updates={
                    **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                        opts["batch_size"] // (2 if use_cr_ctc else 1), 100 // (2 if use_cr_ctc else 1)
                    ),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    # purely used for training
                    "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                    **(cr_ctc if use_cr_ctc else {}),
                    **({"aed_loss_bug_fix": True} if use_cr_ctc else {}),
                },
                post_config_updates={"__multi_proc_dataset_opts": {"num_workers": 25}},
                vocab="spm10k",
                train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
                dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
                # avoid OOM
                # env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
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
