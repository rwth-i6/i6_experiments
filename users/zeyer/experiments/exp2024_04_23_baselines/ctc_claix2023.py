"""
Config for RWTH IPC CLAIX-2023 cluster experiments for CTC
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
from .ctc import train_exp as ctc_train_exp, _raw_sample_rate

from i6_experiments.users.zeyer.experiments.exp2024_10_16_consistency_reg_ctc import cr_ctc_training
from i6_experiments.users.zeyer.train_v4 import train, ModelDefWithCfg

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerPositionwiseFeedForward


def py():
    from returnn.frontend.encoder.conformer import ConformerConvSubsample

    # Consistency regularization (CR) (crLoss).
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

    # TODO Instead of CR, use _interpolate_grad_probs.

    # shuffleBatch100. But not so relevant here? No large laplace, also max 200 seqs in batch.
    # (n12-b250k-shuffleBatch100-spm10k) 6.15

    # Small vocab (spm512), now time downsampling 4.
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

    # Blank separated (blankSep). Baseline (without blankSep): 5.99
    # ctc_train_exp(  # 5.97. so no real improvement.
    #     "time4-n12-spm512-blankSep-auxAED-b150k",
    #     config_96gb_bf16_accgrad1,
    #     model_config={
    #         "enc_input_layer": rf.build_dict(
    #             ConformerConvSubsample,
    #             out_dims=[32, 64, 64],
    #             filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #             pool_sizes=[(1, 2)],
    #             strides=[(1, 1), (2, 1), (2, 1)],
    #         ),
    #         "enc_conformer_layer": rf.build_dict(
    #             ConformerEncoderLayer,
    #             ff=rf.build_dict(
    #                 ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #             ),
    #             num_heads=8,
    #         ),
    #         "feature_batch_norm": True,
    #         "num_enc_layers": 12,
    #         "out_blank_separated": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
    #         "use_fixed_ctc_grad": "v2",
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm512",
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     # avoid OOM
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Diff am/prior scales, with downsampling 4, spm512, blank separation.
    for am_scale, prior_scale, prior_type in [
        # Baseline (1.0, 0.0, None): 5.97
        # (0.7, 0.0, None),  # 6.21
        # (0.5, 0.2, "batch"),  # 7.66
        # (0.5, 0.5, "batch"),  # 99.09
        # (1.0, 1.0, "batch"),  # 94.3
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

    # Diff am/prior scales, with downsampling 4, spm512.
    for am_scale, prior_scale, prior_type in [
        # Baseline (1.0, 0.0, None): 5.99
        # (0.7, 0.0, None),  # 6.34
        # (0.5, 0.2, "batch"),  # 7.13
        # (0.5, 0.5, "batch"),  # 98.48
        # (1.0, 1.0, "batch"),  # 93.79
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

    ctc_train_exp(  # 5.85
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

    # Time downsampling 4, spm10k.
    # Blank separated (blankSep). Baseline (without blankSep): 5.85
    ctc_train_exp(  # 5.77. so some small improvement.
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

    # Time downsampling 6 (standard), spm10k.
    # Blank separated (blankSep).
    # Small improvement on test with blankSep?
    for blank_sep in [
        # best epoch (89): {"dev-clean": 2.49, "dev-other": 5.89, "test-clean": 2.66, "test-other": 6.17}
        # last epoch: {"dev-clean": 2.47, "dev-other": 5.9, "test-clean": 2.63, "test-other": 6.1}
        False,
        True,  # (ep 99) {"dev-clean": 2.49, "dev-other": 5.85, "test-clean": 2.62, "test-other": 6.03}
    ]:
        ctc_train_exp(
            f"n12-spm10k{'-blankSep' if blank_sep else ''}-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            model_config={
                "enc_conformer_layer": rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                "feature_batch_norm": True,
                "num_enc_layers": 12,
                **({"out_blank_separated": True} if blank_sep else {}),
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
                **({"use_fixed_ctc_grad": "v2"} if blank_sep else {}),
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # from .model_ext.ctc_sep_blank import ModelSepBlank, SeparateBlankModel

    # Time downsampling 6 (standard), spm10k.
    # Output blank separated (blankSep) + separated blank model.
    # Currently way worse than baseline (baseline (with blankSep): dev-other: 5.85).
    # Interestingly, relative diff between clean and other is way smaller than in baseline
    # (baseline: "dev-clean": 2.49, "dev-other": 5.85; this: "dev-clean": 8.61, "dev-other": 12.9).
    # ctc_train_exp(  # 12.9
    #     f"n12-spm10k-blankSep-blankSepModel-auxAED-b150k",
    #     config_96gb_bf16_accgrad1,
    #     model_config={
    #         "ctc_model_cls": rf.build_dict(ModelSepBlank)["class"],
    #         "separate_blank_model": rf.build_dict(SeparateBlankModel),
    #         "out_blank_separated": True,
    #         "enc_conformer_layer": rf.build_dict(
    #             ConformerEncoderLayer,
    #             ff=rf.build_dict(
    #                 ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #             ),
    #             num_heads=8,
    #         ),
    #         "feature_batch_norm": True,
    #         "num_enc_layers": 12,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(150_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_attention_decoder": rf.build_dict(TransformerDecoder, num_layers=6),  # purely used for training
    #         "use_fixed_ctc_grad": "v2",
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     # avoid OOM
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    from .model_ext.ctc_sep_net import ModelSepNet, FeedForwardNet, ctc_training_with_sep_net

    # Time downsampling 6 (standard), spm10k.
    # Separate FF net.
    ctc_train_exp(
        f"n12-spm10k-sepFf_alpha05-auxAED-b150k",
        config_96gb_bf16_accgrad1,
        train_def=ctc_training_with_sep_net,
        model_config={
            "ctc_model_cls": rf.build_dict(ModelSepNet)["class"],
            "separate_enc_net": rf.build_dict(FeedForwardNet),
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
            "use_fixed_ctc_grad": "v2",
            "sep_net_grad_interpolate_alpha": 0.5,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        # avoid OOM
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Time downsampling 6 (standard), spm10k.
    # Separate FF net, also with beta (smoothing for both main and sep net).
    for alpha in [0.1, 0.2, 0.5]:
        ctc_train_exp(
            f"n12-spm10k-sepFf_alpha{str(alpha).replace('.', '')}_beta05-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            train_def=ctc_training_with_sep_net,
            model_config={
                "ctc_model_cls": rf.build_dict(ModelSepNet)["class"],
                "separate_enc_net": rf.build_dict(FeedForwardNet),
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
                "use_fixed_ctc_grad": "v2",
                "sep_net_grad_interpolate_alpha": alpha,
                "sep_net_grad_interpolate_beta": 0.5,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    from i6_experiments.common.setups import serialization
    from sisyphus import gs

    # Diff am/prior scales, with downsampling 4, spm10k.
    for am_scale, prior_scale, prior_type, ext_train_opts in [
        # Baseline (1.0, 0.0, None, {}): 5.85
        # (0.7, 0.0, None, {}),  # 6.2
        # (0.5, 0.2, "batch", {}),  # 13.38
        # (0.7, 0.2, "batch", {}),  # 8.47 (note: non-fixed variant)
        # (0.7, 0.2, "running_mean", {"prior_running_mean_momentum": 0.001}),  # 28.87
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
                **ext_train_opts,
                "use_fixed_ctc_grad": "v2",
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Time downsampling 4, spm10k.
    # Log prob normed gradient (lpNormedGrad)
    # Baseline without lpNormedGrad: 5.85
    for name, opts in {
        # "C05_11P1": {  # 6.15
        #     "log_prob_normed_grad": {
        #         "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0}
        #     },
        # },
        # "C05_15P1": {  # 6.66
        #     "log_prob_normed_grad": {
        #         "func": {"clamp_min": 0.5, "clamp_max": 1.5, "scale_type": "inv_num_labels", "prior_exp": 1.0}
        #     }
        # },
        # "C01_11P1": {  # 10.12
        #     "log_prob_normed_grad": {
        #         "func": {"clamp_min": 0.1, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0}
        #     }
        # },
        # "C05_11P1Seq": {  # 5.81
        #     "log_prob_normed_grad": {
        #         "prior": "seq_grad",
        #         "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0},
        #     },
        # },
        # "C05_11P07NSeq": {  # 5.83
        #     "log_prob_normed_grad": {
        #         "prior": "seq_grad",
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     },
        # },
        # "C05_11P07N": {  # 5.85
        #     "log_prob_normed_grad": {
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         }
        #     }
        # },
    }.items():
        ctc_train_exp(
            f"time4-n12-spm10k-auxAED-b150k-lpNormedGrad{name}",
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
                "use_fixed_ctc_grad": "v2",
                # See _maybe_apply_log_probs_normed_grad below.
                # func are opts for NormedGradientFuncInvPrior, other opts are for normed_gradient.
                **opts,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            epilog=[
                serialization.NonhashedCode(f"sys.path.append({gs.BASE_DIR + '/projects/2024-alignment-analysis'!r})\n")
            ],
        )

    # Time downsampling 6, spm10k.
    # Log prob normed gradient (lpNormedGrad)
    # Note on clamping with inv_num_labels:
    # Blank prior has highest prob.
    # inv_num_labels -> grad * (1/prior) * (1/num_labels).
    # The clipping is on (1/prior) * (1/num_labels).
    # Note: (1/num_labels) is like a uniform prior. So this is the ratio between the uniform prior and estimated prior.
    # For blank, this will be quite a bit lower than 1.0, as blank dominates clearly (even when not peaky).
    # Thus, clamp_min is basically only about blank, no other label.
    # All other labels will be less than uniform prior, so clamp_max.
    for name, opts in {
        # best epoch (89):  {"dev-clean": 2.49, "dev-other": 5.89, "test-clean": 2.66, "test-other": 6.17}
        # last epoch (100): {"dev-clean": 2.47, "dev-other": 5.90, "test-clean": 2.63, "test-other": 6.10}
        # (Maybe we should always report last epoch? Currently, we report "best" (which is per dev-other),
        #  which induces quite some noise, as e.g. last epoch is maybe 0.01 worse on dev-other, but overall better,
        #  and more comparable?)
        "": {},
        # "-lpNormedGradC05_11P1Seq": {  # 5.89
        #     "log_prob_normed_grad": {
        #         "prior": "seq_grad",
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 1.0,
        #         },
        #     }
        # },
        # In many cases, using the batch is worse than using the seq? (Also with time4. Also different prior scales.)
        # "-lpNormedGradC05_11P1": {  # 6.13
        #     "log_prob_normed_grad": {
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 1.0,
        #         }
        #     }
        # },
        # For scale 0.7, again using seq is better than batch, but difference is much smaller?
        # Slightly better than prior scale 1.0. (Somewhat more consistent on test-other, last epoch.)
        # "-lpNormedGradC05_11P07NSeq": {  # 5.86. test-other ep100: 6.02
        #     "log_prob_normed_grad": {
        #         "prior": "seq_grad",
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     }
        # },
        # "-lpNormedGradC05_11P07N": {  # 5.91. test-other ep100: 6.11
        #     "log_prob_normed_grad": {
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         }
        #     }
        # },
        # Diffs are small, dev-other is also slightly misleading. Looking at last epoch is more consistent.
        # Then prior scale 0.5 looks to be still slightly better than 0.7 which is also better than 1.0:
        # Last epoch, test-other: 1.0: 6.34, 0.7: 6.11, 0.5: 6.08; baseline: 6.10
        # "-lpNormedGradC05_11P05N": {  # 5.95 (but better on test-other)
        #     "log_prob_normed_grad": {
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.5,
        #             "prior_renorm": True,
        #         }
        #     }
        # },
        "-lpNormedGradC05_11P05NSeq": {
            "log_prob_normed_grad": {
                "prior": "seq_grad",
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.5,
                    "prior_renorm": True,
                },
            }
        },
        "-lpNormedGradC05_11P03N": {
            "log_prob_normed_grad": {
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.3,
                    "prior_renorm": True,
                }
            }
        },
        # The results with running mean are weird. They are all worse.
        # And the higher the momentum (i.e. using more the current batch, less the running mean),
        # the better, which indicates that just using the current batch is overall better.
        # The results above also show this.
        # Note, there is some small inconsistency to above:
        # This is a shared prior mean over the aux layers + final layer.
        # It's unclear whether this is good or bad, just different to the batch/seq prior,
        # which is just for this layer.
        # Such experiments for running mean are below.
        "-lpNormedGradC05_11P07NExp05": {
            "log_prob_normed_grad": {
                "prior_running_mean_momentum": 0.5,
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                },
            }
        },
        # "-lpNormedGradC05_11P07NExp1_1": {  # 6.26. test-other ep100: 6.52
        #     "log_prob_normed_grad": {
        #         "prior_running_mean_momentum": 0.1,
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     }
        # },
        # "-lpNormedGradC05_11P07NExp1_3": {  # 9.06. test-other ep100: 9.29
        #     "log_prob_normed_grad": {
        #         "prior_running_mean_momentum": 0.001,
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     }
        # },
        # Now with prior running mean per layer.
        # Comparison to shared over layers somewhat inconsistent.
        # Again, the less the momentum, the better.
        # With 0.5 momentum, this is even better than the batch/seq prior,
        # also better than the baseline.
        "-lpNormedGradC05_11P07NExpL08": {
            "prior_running_mean_per_layer": True,
            "log_prob_normed_grad": {
                "prior_running_mean_momentum": 0.8,
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                },
            },
        },
        "-lpNormedGradC05_11P07NExpL05": {  # 5.83. test-other ep100: 5.93
            "prior_running_mean_per_layer": True,
            "log_prob_normed_grad": {
                "prior_running_mean_momentum": 0.5,
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                },
            },
        },
        # "-lpNormedGradC05_11P07NExpL1_1": {  # 6.48. test-other ep100: 6.66
        #     "prior_running_mean_per_layer": True,
        #     "log_prob_normed_grad": {
        #         "prior_running_mean_momentum": 0.1,
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     },
        # },
        # "-lpNormedGradC05_11P07NExpL1_3": {  # 8.66. test-other ep100: 9.12
        #     "prior_running_mean_per_layer": True,
        #     "log_prob_normed_grad": {
        #         "prior_running_mean_momentum": 0.001,
        #         "func": {
        #             "clamp_min": 0.5,
        #             "clamp_max": 1.1,
        #             "scale_type": "inv_num_labels",
        #             "prior_exp": 0.7,
        #             "prior_renorm": True,
        #         },
        #     },
        # },
        "-lpNormedGradC05_11P05NExpL05": {
            "prior_running_mean_per_layer": True,
            "log_prob_normed_grad": {
                "prior_running_mean_momentum": 0.5,
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.5,
                    "prior_renorm": True,
                },
            },
        },
        # Trying with prior_min.
        "-lpNormedGradC05_11SMinP07NExpL05": {
            "prior_running_mean_per_layer": True,
            "log_prob_normed_grad": {
                "prior_running_mean_momentum": 0.5,
                "func": {
                    "clamp_min": 0.1,
                    "clamp_max": 1.1,
                    "scale_type": "prior_min",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                },
            },
        },
    }.items():
        ctc_train_exp(
            f"n12-spm10k-auxAED-b150k{name}",
            config_96gb_bf16_accgrad1,
            model_config={
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
                **({"use_fixed_ctc_grad": "v2"} if opts else {}),
                # See _maybe_apply_log_probs_normed_grad below.
                # func are opts for NormedGradientFuncInvPrior, other opts are for normed_gradient.
                **opts,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            epilog=[
                serialization.NonhashedCode(f"sys.path.append({gs.BASE_DIR + '/projects/2024-alignment-analysis'!r})\n")
            ]
            if opts
            else (),
        )

    # Diff am/prior scales, with downsampling 6, spm10k.
    for am_scale, prior_scale, name, prior_type, extra_train_opts in [
        # Baseline (1.0, 0.0, None):
        # "best" epoch (89): {"dev-clean": 2.49, "dev-other": 5.89, "test-clean": 2.66, "test-other": 6.17}
        # last epoch (100):  {"dev-clean": 2.47, "dev-other": 5.90, "test-clean": 2.63, "test-other": 6.10}
        # (0.7, 0.0, "", None, {}),  # 6.52
        # (0.7, 0.2, "-priorBatch", "batch", {}),  # 9.65
        # Note: Running mean here is shared over layers...
        # (0.7, 0.2, "-priorRunningMean1e_3", "running_mean", {"prior_running_mean_momentum": 0.001}),  # 17.12
        # (0.5, 0.2, "-priorBatch", "batch", {}),  # 8.23 (but not fixed)
        # Fixed means, it uses mean instead of sum (non-fixed above).
        # (0.5, 0.2, "-priorBatchFixed", "batch_fixed", {}),  # 7.72
        # Stop grad here is already fixed (mean).
        # It's weird that this is worse. This is inconsistent to earlier experience.
        # (0.5, 0.2, "-priorBatchStopGrad", "batch_stop_grad", {}),  # 8.22
        # Interestingly, just as with lpNormedGrad, seq-based prior is better than batch-based prior.
        # (0.5, 0.2, "-priorSeq", "seq", {}),  # 6.81
        # Interestingly, here stop grad helps.
        # (0.5, 0.2, "-priorSeqStopGrad", "seq_stop_grad", {}),  # 6.48
        # Interestingly, just as with lpNormedGrad, running mean is bad.
        # Running mean here is again shared over layers. (Below is separate per layer.)
        # The smaller the momentum, the worse, just as with lpNormedGrad.
        # Warning: The selected "best" epoch for some of these is very early, epoch 33 or so.
        # (0.5, 0.2, "-priorRunningMean1e_3", "running_mean", {"prior_running_mean_momentum": 0.001}),  # 22.48
        # (0.5, 0.2, "-priorRunningMean1e_1", "running_mean", {"prior_running_mean_momentum": 0.1}),  # 12.02
        (0.5, 0.2, "-priorRunningMean05", "running_mean", {"prior_running_mean_momentum": 0.5}),
        # Sanity check with momentum 1, i.e. not using the running mean at all.
        # It should be like priorBatchStopGrad. For some reason, seems a bit better? (8.22 vs 7.91)
        # (0.5, 0.2, "-priorRunningMean1e_0", "running_mean", {"prior_running_mean_momentum": 1.0}),  # 7.91
        # Weirdly, this is worse than the shared running mean.
        # (  # 21.81 (best ep 33)
        #     0.5,
        #     0.2,
        #     "-priorRunningMeanPerLayer1e_3",
        #     "running_mean",
        #     {"prior_running_mean_momentum": 0.001, "prior_running_mean_per_layer": True},
        # ),
        # (  # 17.31 (best ep 100)
        #     0.5,
        #     0.2,
        #     "-priorRunningMeanPerLayer1e_1",
        #     "running_mean",
        #     {"prior_running_mean_momentum": 0.1, "prior_running_mean_per_layer": True},
        # ),
        # (0.3, 0.15, "-priorRunningMean1e_3", "running_mean", {"prior_running_mean_momentum": 0.001}),  # 20.37
        (0.3, 0.15, "-priorSeqStopGrad", "seq_stop_grad", {}),
    ]:
        ctc_train_exp(
            f"n12-spm10k-am{am_scale}-prior{prior_scale}{name}-auxAED-b150k",
            config_96gb_bf16_accgrad1,
            model_config={
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
                **extra_train_opts,
                "use_fixed_ctc_grad": "v2",
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Time downsampling 6.
    # Comparing different vocabs, samplings (using max_seq_length_default_input).
    for vocab, sample, alpha in [
        ("spm10k", "bpe", 0.01),  # 6.01 (5.93 without max seq len on audio)
        # ("spm512", None, None),  # 6.08 (11gb)
        # ("spm512", "bpe", 0.001),  # 6.05 (11gb)
        # ("spm512", "bpe", 0.005),  # 6.01 (11gb)
        # ("spm512", "bpe", 0.01),  # 6.08 (but test-* is better than spm512 without sampling) (11gb)
        ("spm512", None, None),  # 6.04
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

    # Another baseline. Time downsampling 6, spm512.
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
