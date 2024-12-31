"""
Config for RWTH IPC CLAIX-2023 cluster experiments.
"""

from __future__ import annotations

from typing import Any, Dict

from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear

from .configs import (
    config_24gb_v6,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from .aed import train_exp as aed_train_exp
from .ctc import train_exp as ctc_train_exp, _raw_sample_rate
from .lm import lm_train_def, lm_model_def

from i6_experiments.users.zeyer.experiments.exp2024_10_16_consistency_reg_ctc import cr_ctc_training
from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_lm_dataset
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

    from returnn.frontend.encoder.conformer import ConformerConvSubsample

    for opts in [
        # Baseline (n12) has {"dev-clean": 2.35, "dev-other": 5.65, "test-clean": 2.66, "test-other": 5.94}.
        # CLAIX baseline: {"dev-clean": 2.54, "dev-other": 5.93, "test-clean": 2.68, "test-other": 6.27}
        # CLAIX CR: {"dev-clean": 2.49, "dev-other": 5.99, "test-clean": 2.68, "test-other": 6.05}
        # v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        # {"num_enc_layers": 12, "batch_size": 200_000, "vocab": "spm10k"},
        # Baseline (n16, spm10k) has {"dev-clean": 2.26, "dev-other": 5.44, "test-clean": 2.5, "test-other": 5.62}.
        # v6-n16-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs10k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        # This here is now spm512 though.
        # Note: In the original CR paper, they don't have time-downsampling!
        # {"num_enc_layers": 16, "batch_size": 10_000, "vocab": "spm512"},
        {"num_enc_layers": 12, "batch_size": 200_000, "vocab": "spm512"},
        # {"num_enc_layers": 12, "batch_size": 150_000, "vocab": "spm512", "time_downsampling": 4},
        # {"num_enc_layers": 12, "batch_size": 75_000, "vocab": "spm512", "time_downsampling": 2},
    ]:
        for cr_ctc in [None, {"cr_loss_scale": 0.2}, {"cr_loss_scale": 0.5}, {"cr_loss_scale": 1.0}]:
            # TODO also adapt specaug for CR...
            use_cr_ctc = cr_ctc is not None
            name = f"crLoss{cr_ctc['cr_loss_scale']}-" if use_cr_ctc else ""
            if opts.get("time_downsampling"):
                name += f"time{opts['time_downsampling']}-"
            name += f"n{opts['num_enc_layers']}-{opts['vocab']}-auxAED"
            ctc_train_exp(
                name,
                config_96gb_bf16_accgrad1,
                train_def=cr_ctc_training if use_cr_ctc else None,
                model_config={
                    **{
                        2: {
                            "enc_input_layer": rf.build_dict(
                                ConformerConvSubsample,
                                out_dims=[32, 64, 64],
                                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                                pool_sizes=[(1, 2)],
                                strides=[(1, 1), (2, 1), (1, 1)],
                            ),
                        },
                        4: {
                            "enc_input_layer": rf.build_dict(
                                ConformerConvSubsample,
                                out_dims=[32, 64, 64],
                                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                                pool_sizes=[(1, 2)],
                                strides=[(1, 1), (2, 1), (2, 1)],
                            ),
                        },
                        None: {},
                    }[opts.get("time_downsampling")],
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
                        opts["batch_size"] // (2 if use_cr_ctc else 1),
                        100 // (2 if use_cr_ctc else 1),
                        batch_size_factor=_batch_size_factor,
                    ),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    # purely used for training
                    "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),
                    **(cr_ctc if use_cr_ctc else {}),
                    **({"use_fixed_ctc_grad": "v2", "aed_loss_bug_fix": True} if use_cr_ctc else {}),
                    "max_seq_length_default_target": None,
                    # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                    # out of 281241 seqs in train, we removed only 71 seqs.
                    # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                    "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                },
                post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
                vocab=opts["vocab"],
                train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
                dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
                # avoid OOM
                # env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
            )
        del opts, use_cr_ctc, name

    # shuffleBatch100. But not so relevant here? No large laplace, also max 200 seqs in batch.
    # (n12-b250k-shuffleBatch100-spm10k) 6.15

    # Small vocab, now time downsampling 4.
    ctc_train_exp(  # 5.99
        "time4-n12-spm512-auxAED-b150k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_input_layer": rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (2, 1), (2, 1)],
            ),
            "enc_conformer_layer": rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    ctc_train_exp(
        "time4-n12-spm512-blankSep-auxAED-b150k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_input_layer": rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (2, 1), (2, 1)],
            ),
            "enc_conformer_layer": rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
            "out_blank_separated": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            "use_fixed_ctc_grad": "v2",
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    for am_scale, prior_scale, prior_type in [
        (0.7, 0.0, None),
        (0.5, 0.2, "batch"),
        (0.5, 0.5, "batch"),
        (1.0, 1.0, "batch"),
    ]:
        ctc_train_exp(
            f"time4-n12-spm512-blankSep-am{am_scale}-prior{prior_scale}-priorType{prior_type}-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_input_layer": rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (2, 1)],
                ),
                "enc_conformer_layer": rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
                "num_enc_layers": 12,
                "out_blank_separated": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
                # Only for training:
                "ctc_am_scale": am_scale,
                "ctc_prior_scale": prior_scale,
                "ctc_prior_type": prior_type,
                "use_fixed_ctc_grad": "v2",
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm512",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    for am_scale, prior_scale, prior_type in [
        (0.7, 0.0, None),
        (0.5, 0.2, "batch"),
        (0.5, 0.5, "batch"),
        (1.0, 1.0, "batch"),
    ]:
        ctc_train_exp(
            f"time4-n12-spm512-am{am_scale}-prior{prior_scale}-priorType{prior_type}-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_input_layer": rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (2, 1)],
                ),
                "enc_conformer_layer": rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
                "num_enc_layers": 12,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
                # Only for training:
                "ctc_am_scale": am_scale,
                "ctc_prior_scale": prior_scale,
                "ctc_prior_type": prior_type,
                "use_fixed_ctc_grad": "v2",
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm512",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    ctc_train_exp(
        "time4-n12-spm10k-auxAED-b150k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_input_layer": rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (2, 1), (2, 1)],
            ),
            "enc_conformer_layer": rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    ctc_train_exp(
        "time4-n12-spm10k-blankSep-auxAED-b150k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_input_layer": rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 64, 64],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (2, 1), (2, 1)],
            ),
            "enc_conformer_layer": rf.build_dict(
                ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
            "out_blank_separated": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            "use_fixed_ctc_grad": "v2",
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    for am_scale, prior_scale, prior_type in [
        # (0.7, 0.0, None),
        (0.5, 0.2, "batch"),
        # (0.5, 0.5, "batch"),
        # (1.0, 1.0, "batch"),
    ]:
        ctc_train_exp(
            f"time4-n12-spm10k-am{am_scale}-prior{prior_scale}-priorType{prior_type}-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_input_layer": rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (2, 1)],
                ),
                "enc_conformer_layer": rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
                "num_enc_layers": 12,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
                # Only for training:
                "ctc_am_scale": am_scale,
                "ctc_prior_scale": prior_scale,
                "ctc_prior_type": prior_type,
                "use_fixed_ctc_grad": "v2",
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm512",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # TODO normed grad...

    for vocab, sample, alpha in [
        ("spm10k", "bpe", 0.01),  # 6.01 (5.93 without max seq len on audio)
        # ("spm512", None, None),  # 6.08 (11gb)
        # ("spm512", "bpe", 0.001),  # 6.05 (11gb)
        # ("spm512", "bpe", 0.005),  # 6.01 (11gb)
        # ("spm512", "bpe", 0.01),  # 6.08 (but test-* is better than spm512 without sampling) (11gb)
        ("spm512", None, None),
        # ("spm128", None, None),  # 6.37 (11gb)
        # TODO ("spm128", "bpe", 0.001),
        # ("spm128", "bpe", 0.01),  # 6.40 (11gb)
        # TODO ("spm128", "bpe", 0.005),
        # ("bpe128", None, None),
        # ("spm64", None, None),
        # ("bpe64", None, None),
        # ("utf8", None, None),
        # ("char", None, None),
        # ("bpe0", None, None),
    ]:
        ctc_train_exp(
            f"n12-auxAED-b200k"
            f"-{vocab}" + (f"-{sample}Sample{str(alpha).replace('.', '').replace('-','_')}" if sample else ""),
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_conformer_layer": rf.build_dict(
                    rf.encoder.conformer.ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
                "num_enc_layers": 12,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab=vocab,
            train_vocab_opts=(
                {
                    "other_opts": (
                        {
                            "spm": {"enable_sampling": True, "alpha": alpha},
                            "bpe": {"class": "SamplingBytePairEncoding", "breadth_prob": alpha},
                        }[sample]
                    )
                }
                if sample
                else None
            ),
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Another baseline.
    ctc_train_exp(  # {"dev-clean": 2.36, "dev-other": 6.14, "test-clean": 2.56, "test-other": 6.15}
        f"n12-auxAED-b200k-spm512-bpeSample0005",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_conformer_layer": rf.build_dict(
                rf.encoder.conformer.ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.005}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    from .optim_ext.zipformer_scaled_adam import ScaledAdam

    # ScaledAdam.
    # n12-auxAED-b200k-optScaledAdam-spm512-bpeSample0005: 13.89
    # TODO what now? (see also below the LM exp) maybe also impl weight decay?
    # Try higher lr.
    ctc_train_exp(
        f"n12-auxAED-b200k-optScaledAdam-lr1e_2-spm512-bpeSample0005",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_conformer_layer": rf.build_dict(
                rf.encoder.conformer.ConformerEncoderLayer,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                ),
                num_heads=8,
            ),
            "feature_batch_norm": True,
            "num_enc_layers": 12,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(200_000, 100, batch_size_factor=_batch_size_factor, peak_lr=1e-2),
            "optimizer.class": rf.build_dict(ScaledAdam)["class"],
            "optimizer.clipping_scale": 2.0,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=[
            # ScaledAdam does not have weight decay (??) (TODO...)
            "optimizer.weight_decay",
            "optimizer.weight_decay_modules_blacklist",
        ],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.005}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    from .model_ext.zipformer import RFZipFormerEncoder

    # ZipFormer medium
    ctc_train_exp(
        "zipformer-medium-spm512-auxAED-b500k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": rf.build_dict(
                RFZipFormerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (1, 1)],
                ),
                # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#medium-scale-model-number-of-model-parameters-89987295-ie-900-m
                # As I see it, it just uses the defaults, so no params.
                params=dict(),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(500_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=["aux_loss_layers"],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # ZipFormer medium with CR
    for cr_ctc in [None, {"cr_loss_scale": 0.2}]:
        use_cr_ctc = cr_ctc is not None
        ctc_train_exp(
            f"zipformer-medium-spm512-auxAED-b500k-cr{use_cr_ctc}",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_build_dict": rf.build_dict(
                    RFZipFormerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (2, 1), (1, 1)],
                    ),
                    # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#medium-scale-model-number-of-model-parameters-89987295-ie-900-m
                    # As I see it, it just uses the defaults, so no params.
                    params=dict(),
                ),
                "feature_batch_norm": True,
            },
            train_def=cr_ctc_training if use_cr_ctc else None,
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(
                    500_000 // (2 if use_cr_ctc else 1),
                    100 // (2 if use_cr_ctc else 1),
                    batch_size_factor=_batch_size_factor,
                ),
                "optimizer.weight_decay": 1e-2,
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
                **(cr_ctc if use_cr_ctc else {}),
                **({"use_fixed_ctc_grad": "v2", "aed_loss_bug_fix": True} if use_cr_ctc else {}),
            },
            config_deletes=["aux_loss_layers"],
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm512",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # ZipFormer medium with ScaledAdam
    ctc_train_exp(
        "zipformer-medium-spm512-auxAED-b500k-optScaledAdam-lr1e_2",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": rf.build_dict(
                RFZipFormerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (1, 1)],
                ),
                # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#medium-scale-model-number-of-model-parameters-89987295-ie-900-m
                # As I see it, it just uses the defaults, so no params.
                params=dict(),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(500_000, 100, batch_size_factor=_batch_size_factor, peak_lr=1e-2),
            "optimizer.class": rf.build_dict(ScaledAdam)["class"],
            "optimizer.clipping_scale": 2.0,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=[
            "aux_loss_layers",
            # ScaledAdam does not have weight decay (??) (TODO...)
            "optimizer.weight_decay",
            "optimizer.weight_decay_modules_blacklist",
        ],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # ZipFormer large
    ctc_train_exp(
        "zipformer-large-spm512-auxAED-b300k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": rf.build_dict(
                RFZipFormerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (1, 1)],
                ),
                # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#large-scale-model-number-of-model-parameters-174319650-ie-1743-m-1
                params=dict(
                    num_encoder_layers=(2, 2, 4, 5, 4, 2),
                    feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
                    encoder_dim=(192, 256, 512, 768, 512, 256),
                    encoder_unmasked_dim=(192, 192, 256, 320, 256, 192),
                ),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(300_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=["aux_loss_layers"],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # ZipFormer large, spm10k
    ctc_train_exp(
        "zipformer-large-spm10k-auxAED-b300k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": rf.build_dict(
                RFZipFormerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (1, 1)],
                ),
                # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#large-scale-model-number-of-model-parameters-174319650-ie-1743-m-1
                params=dict(
                    num_encoder_layers=(2, 2, 4, 5, 4, 2),
                    feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
                    encoder_dim=(192, 256, 512, 768, 512, 256),
                    encoder_unmasked_dim=(192, 192, 256, 320, 256, 192),
                ),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(300_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=["aux_loss_layers"],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # ZipFormer large blankSep
    ctc_train_exp(
        "zipformer-large-blankSep-spm512-auxAED-b300k",
        config_96gb_bf16_accgrad1,
        model_config={
            "enc_build_dict": rf.build_dict(
                RFZipFormerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (2, 1), (1, 1)],
                ),
                # https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/RESULTS.md#large-scale-model-number-of-model-parameters-174319650-ie-1743-m-1
                params=dict(
                    num_encoder_layers=(2, 2, 4, 5, 4, 2),
                    feedforward_dim=(512, 768, 1536, 2048, 1536, 768),
                    encoder_dim=(192, 256, 512, 768, 512, 256),
                    encoder_unmasked_dim=(192, 192, 256, 320, 256, 192),
                ),
            ),
            "feature_batch_norm": True,
            "out_blank_separated": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(300_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
        },
        config_deletes=["aux_loss_layers"],
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm512",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # TODO ctc without aux
    # TODO ctc with different sampling
    # TODO for smaller vocabs, less downsampling
    # TODO zipformer...

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
