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

    # Blank separated (blankSep). Baseline (without blankSep): 5.99
    ctc_train_exp(  # 5.97. so no real improvement.
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
        # Baseline (1.0, 0.0, None): 5.97
        (0.7, 0.0, None),  # 6.21
        (0.5, 0.2, "batch"),  # 7.66
        (0.5, 0.5, "batch"),  # 99.09
        (1.0, 1.0, "batch"),  # 94.3
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
        # Baseline (1.0, 0.0, None): 5.99
        (0.7, 0.0, None),  # 6.34
        (0.5, 0.2, "batch"),  # 7.13
        (0.5, 0.5, "batch"),  # 98.48
        (1.0, 1.0, "batch"),  # 93.79
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

    # Blank separated (blankSep).
    for blank_sep in [False, True]:
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

    from i6_experiments.common.setups import serialization
    from sisyphus import gs

    for am_scale, prior_scale, prior_type in [
        # Baseline (1.0, 0.0, None): 5.85
        # (0.7, 0.0, None),  # 6.2
        # (0.5, 0.2, "batch"),  # 13.38
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
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            # avoid OOM
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Log prob normed gradient (lpNormedGrad)
    # Baseline without lpNormedGrad: 5.85
    for name, opts in {
        "C05_11P1": {  # 6.15
            "log_prob_normed_grad": {
                "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0}
            },
        },
        "C05_15P1": {  # 6.66
            "log_prob_normed_grad": {
                "func": {"clamp_min": 0.5, "clamp_max": 1.5, "scale_type": "inv_num_labels", "prior_exp": 1.0}
            }
        },
        "C01_11P1": {  # 10.12
            "log_prob_normed_grad": {
                "func": {"clamp_min": 0.1, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0}
            }
        },
        "C05_11P1Seq": {  # 5.81
            "log_prob_normed_grad": {
                "prior": "seq_grad",
                "func": {"clamp_min": 0.5, "clamp_max": 1.1, "scale_type": "inv_num_labels", "prior_exp": 1.0},
            },
        },
        "C05_11P07NSeq": {
            "log_prob_normed_grad": {
                "prior": "seq_grad",
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                },
            },
        },
        "C05_11P07N": {  # 5.85
            "log_prob_normed_grad": {
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                }
            }
        },
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

    # Log prob normed gradient (lpNormedGrad)
    for name, opts in {
        "": {},
        "-lpNormedGradC05_11P07N": {
            "log_prob_normed_grad": {
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.7,
                    "prior_renorm": True,
                }
            }
        },
        "-lpNormedGradC05_11P05N": {
            "log_prob_normed_grad": {
                "func": {
                    "clamp_min": 0.5,
                    "clamp_max": 1.1,
                    "scale_type": "inv_num_labels",
                    "prior_exp": 0.5,
                    "prior_renorm": True,
                }
            }
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
