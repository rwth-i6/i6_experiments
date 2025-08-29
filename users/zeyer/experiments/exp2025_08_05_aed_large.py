"""
New AED baseline tuning for larger AED model.
"""

from __future__ import annotations

from typing import Union, Any, Sequence, Dict, Tuple
import functools
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v3,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)

from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)


__setup_root_prefix__ = "exp2025_08_05_aed_large"


def py():
    prefix = get_setup_prefix_for_module(__name__)

    # Warning: this keeps aux_loss_layers=[4, 8], not sure if this is optimal...
    # {"dev-clean": 9.78, "dev-other": 11.25, "test-clean": 18.63, "test-other": 13.44}
    # Bad. See below for better.
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-spm10k-spmSample07-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             pos_enc=None,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Try with BPE sampling (SamplingBytePairEncoding) (bpeSample). (More consistent to our CTC setup.)
    #  spmSample07: {"dev-clean": 9.78, "dev-other": 11.25, "test-clean": 18.63, "test-other": 13.44}
    # bpeSample001: {"dev-clean": 3.51, "dev-other":  5.66, "test-clean":  3.54, "test-other":  5.85}
    # (Commented out because we have better baselines below.)
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-spm10k-bpeSample001-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             pos_enc=None,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # for lr in [0.1, 0.5, 1.0]:
    #     # lr 0.1: 5.39
    #     # lr 0.5: 5.16
    #     # lr 1.0: 5.66
    #     # lr 2.0: 14.36
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-spm10k-bpeSample001-baseLr{lr}-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 pos_enc=None,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=lr),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # DecCrossAttNoBias: 10.33
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecCrossAttNoBias-spm10k-bpeSample001-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             pos_enc=None,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(
    #                 self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False),
    #                 cross_att=rf.build_dict(rf.CrossAttention, with_bias=False),
    #             ),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Try aux_loss_layers=[6, 12].
    #  [4, 8]: {"dev-clean": 9.78, "dev-other": 11.25, "test-clean": 18.63, "test-other": 13.44}
    # [6, 12]: {"dev-clean": 3.61, "dev-other":  5.72, "test-clean":  4.18, "test-other":  5.84}
    # (But better baselines below.)
    # (More aux_loss_layers experiments below.)
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-aux6_12-spm10k-spmSample07-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             pos_enc=None,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [6, 12],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # TrafoDec with pos enc: 5.18 (vs 5.72)
    # (Better baselines below.)
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-aux6_12-spm10k-spmSample07-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [6, 12],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Taking best we have so far: DecPosEncAbs, aux6_12, baseLr0.5. (But still spmSample07.)
    # {"dev-clean": 2.82, "dev-other": 4.97, "test-clean": 3.41, "test-other": 5.3}
    # (Better baselines below.)
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-aux6_12-spm10k-spmSample07-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [6, 12],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Some other better baseline: DecPosEncAbs + bpeSample001 + baseLr0.5.
    # 4.87, {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # (But better baselines below, with featBN.)
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Again other aux_loss_layers with better baseline.
    # (Better baselines below. Also below again some more on aux_loss_layers.)
    # for aux_loss_layers in [
    #     [4, 8],  # {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35} (bad baseline)
    #     [6, 12],  # {"dev-clean": 3.45, "dev-other": 5.51, "test-clean": 3.95, "test-other": 6.03}
    #     [8, 16],  # {"dev-clean": 2.57, "dev-other": 5.5, "test-clean": 4.27, "test-other": 6.62}
    #     [4, 10, 16],  # {"dev-clean": 2.84, "dev-other": 4.85, "test-clean": 3.02, "test-other": 5.2}
    #     [16],  # {"dev-clean": 3.08, "dev-other": 5.52, "test-clean": 4.0, "test-other": 5.83}
    # ]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-aux{'_'.join(map(str, aux_loss_layers))}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "aux_loss_layers": aux_loss_layers,
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # Try different decoder dropout. (Default is dropout 0.1, att_dropout 0.1.)
    # dec_drop,dec_att_drop.
    # 0.1,0.1: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # 0.1,0.2: {"dev-clean": 2.42, "dev-other": 5.02, "test-clean": 2.83, "test-other": 5.20}
    # 0.2,0.1: {"dev-clean": 11.17, "dev-other": 11.08, "test-clean": 16.7, "test-other": 12.04}
    # 0.2,0.2: {"dev-clean": 8.84, "dev-other": 9.53, "test-clean": 12.69, "test-other": 10.47}
    # for dec_drop, dec_att_drop in [(0.2, 0.2), (0.1, 0.2), (0.2, 0.1)]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-DecDrop{dec_drop}-DecAttDrop{dec_att_drop}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 dropout=dec_drop,
    #                 att_dropout=dec_att_drop,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # Try different dec model dim.
    # 512: {"dev-clean": 2.9, "dev-other": 5.2, "test-clean": 3.09, "test-other": 5.54}
    # 640: {"dev-clean": 2.34, "dev-other": 5.04, "test-clean": 2.46, "test-other": 5.49}
    # 768: {"dev-clean": 7.02, "dev-other": 9.33, "test-clean": 10.72, "test-other": 9.85}
    # 1024: {"dev-clean": 2.92, "dev-other": 5.04, "test-clean": 3.12, "test-other": 5.58}
    # for dec_model_dim in [512, 640, 768, 1024]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecD{dec_model_dim}-DecPosEncAbs-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=dec_model_dim,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 dropout=0.1,
    #                 att_dropout=0.1,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # label smoothing
    # 0.1: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # 0.2: {"dev-clean": 3.43, "dev-other": 5.82, "test-clean": 3.55, "test-other": 6.06}
    # for label_smoothing in [0.1, 0.2]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-ls{label_smoothing}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             **({"label_smoothing": label_smoothing} if label_smoothing != 0.1 else {}),
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # wdblackNorm: add LayerNorm/RMSNorm to weight decay blacklist
    # no wdblackNorm: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    #    wdblackNorm: {"dev-clean": 2.54, "dev-other": 4.97, "test-clean": 3.14, "test-other": 5.51}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-wdblackNorm-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=dec_model_dim,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             dropout=0.1,
    #             att_dropout=0.1,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "optimizer.weight_decay_modules_blacklist": [
    #             "rf.Embedding",
    #             "rf.LearnedRelativePositionalEncoding",
    #             "rf.LayerNorm",
    #             "rf.RMSNorm",
    #         ],
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # RMSNormGemma: ..*(1+scale) instead of ..*scale as normally done in RMSNorm
    # no RMSNormGemma: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    #    RMSNormGemma: {"dev-clean": 2.44, "dev-other": 5.21, "test-clean": 2.72, "test-other": 5.47}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-RMSNormGemma-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=dec_model_dim,
    #             norm=rf.build_dict(RMSNormGemma),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             dropout=0.1,
    #             att_dropout=0.1,
    #         ),
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Try batch norm on features (feature_batch_norm) (featBN) (as we have it in the CTC setup).
    # without featBN: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    #    with featBN: {"dev-clean": 3.20, "dev-other": 5.74, "test-clean": 5.56, "test-other": 6.94}
    # But this seems to be an outlier? With better aux CTC loss, featBN seems better (see below).
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=dec_model_dim,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             dropout=0.1,
    #             att_dropout=0.1,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # featBN with better baseline (DecPosEncAbs and aux4_10_16 etc).
    # (baseline without featBN:
    #         {"dev-clean": 2.84, "dev-other": 4.85, "test-clean": 3.02, "test-other": 5.20})
    # featBN: {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08} !!
    # (But see right below for a better setup as starting point, using the same settings.)
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Same exp as above, but using new serialization. This should not change anything in behavior.
    # This is:
    # 1. To have the settings better for future experiments.
    # 2. To see how much it changes by a second run (hopefully not much).
    # (Note, the "-s2" at the end is to have a different name, to not use the same alias.
    #  but for other experiments, we would not add the "-s2".)
    # baseline (s1): {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    #           s2:  {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # This is a very unfortunate result...
    # It is also not exactly the same... e.g. behavior_version 21 -> 24, num_workers 25 -> 4
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k-s2",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # For comparison, behavior version 21 again, also 23. Does that make the difference?
    for bhv in [21, 23, 24]:
        aed_train_exp(
            f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k-s2-bhv{bhv}",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                "behavior_version": bhv,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Disable some of the masking that is done with behavior version 24 (in training) (noConvMasks).
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k-s2-noConvMasks",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "rf_use_mask_conv": False,
            "rf_use_mask_pool": False,
            "rf_use_mask_stft": False,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # TODO likely bhv24 is worse, probably due to less regularization that we have with bhv21.
    #   how can we replicate that more explicitly?
    #   maybe in RF conv.py, add sth like:
    #   def setup_hook_on_conv_padding(...):
    #     ...
    #   but in a way that would keep the out_spatial_dims. similar to _consistent_same_padding.
    #   could also cover randomness in striding offsets, like _consistent_same_padding but random instead of consistent.
    #   or: just pad some random noise left/right to the audio raw samples. or silence.
    #   that might have a similar effect (but then also do in recog? or do this in training only sometimes?)

    # Also try abs pos enc in encoder (EncPosEncAbs) (compare this to the ...-s2 above)
    # baseline (s1): {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # EncPosEncAbs: ...
    aed_train_exp(
        "EncL16-DecL6-D1024-EncPosEncAbs-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                pos_enc=rf.build_dict(rf.sinusoidal_positional_encoding),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # EncAddEos
    enc_dim = Dim(1024, name="enc_feat")
    aed_train_exp(
        "EncL16-DecL6-D1024-EncAddEos-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerInputLayerExt,
                    out_dim=enc_dim,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_postfix_frames=1,
                ),
                num_layers=16,
                out_dim=enc_dim,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    del enc_dim

    # Pad audio (AudioPad) to somehow have it similar as wrong conv
    # (as in behavior version 21, but here using behavior version 24).
    # Baseline (bhv21):
    # Baseline (bhv24) (directly comparable): ...
    for name, opts in {"0": None, "1k": 1000}.items():
        aed_train_exp(
            f"EncL16-DecL6-D1024-AudioPad{name}-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
                **({"pad_audio": opts} if opts else {}),
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Note: Gemma3 changes: pre+post norm, qknorm, groupatt, sliding+full att, RMSNormGemma
    # RMSNormGemma: rmsnorm with 1+scale (better for weight decay)
    # TODO try Gemma3 changes: pre+post norm, qknorm, groupatt, sliding+full att

    # Aux CTC loss with label smoothing (auxCtcLs)
    # (Note, suboptimal: featBN missing here.)
    # for aux_loss_layers, aux_ctc_ls in [
    #     # ([4, 8], 0.0),  # {"dev-clean": 2.92, "dev-other": 5.04, "test-clean": 3.12, "test-other": 5.58} (baseline)
    #     # ([4, 8], 0.1),  # {"dev-clean": 2.45, "dev-other": 5.14, "test-clean": 2.84, "test-other": 5.47}
    #     # ([4, 8], 0.5),  # {"dev-clean": 2.81, "dev-other": 5.29, "test-clean": 3.24, "test-other": 5.76}
    #     # ([4, 10, 16], 0.1),  # {"dev-clean": 4.24, "dev-other": 5.93, "test-clean": 4.62, "test-other": 6.34}
    #     # ([4, 10, 16], 0.2),  # {"dev-clean": 3.69, "dev-other": 5.24, "test-clean": 3.82, "test-other": 5.81}
    #     # ([4, 10, 16], 0.5),  # {"dev-clean": 4.13, "dev-other": 5.93, "test-clean": 4.47, "test-other": 6.09}
    #     # ([16], 0.1),  # {"dev-clean": 3.56, "dev-other": 5.87, "test-clean": 3.75, "test-other": 6.06}
    #     # ([16], 0.5),  # {"dev-clean": 3.34, "dev-other": 6.02, "test-clean": 3.77, "test-other": 6.13}
    # ]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-aux{'_'.join(map(str, aux_loss_layers))}-auxCtcLs{aux_ctc_ls}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=dec_model_dim,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 dropout=0.1,
    #                 att_dropout=0.1,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "aux_loss_layers": aux_loss_layers,
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #             **({"aux_ctc_label_smoothing": aux_ctc_ls} if aux_ctc_ls else {}),
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # Try aux_loss_layers=[4, 10, 16]
    # baseline [4, 8]: {"dev-clean": 9.78, "dev-other": 11.25, "test-clean": 18.63, "test-other": 13.44}
    #     [4, 10, 16]: {"dev-clean": 2.82, "dev-other":  4.80, "test-clean":  3.66, "test-other":  5.22}
    aed_train_exp(
        "EncL16-DecL6-D1024-aux4_10_16-spm10k-spmSample07-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                pos_enc=None,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Better baseline (DecPosEncAbs, featBN, bpeSample001, baseLr0.5) with CTC label smoothing (auxCtcLs0.1).
    # Baseline without aux CTC loss: {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    #                   auxCtcLs0.1: {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    # -> Aux CTC loss label smoothing unclear...
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "aux_ctc_label_smoothing": 0.1,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Try with text augment.
    from i6_experiments.users.zeyer.nn_rf.text_augment import text_augment

    def _ta_vA_err_prob(q):
        # q=(1-p)^2 will be the approx error rate.
        p = round(1 - (1 - q) ** 0.5, 5)
        return {
            "ins_probs": [1 - p, p * 0.8, p * 0.2],
            "keep_del_sub_probs": [1 - p, p * 0.6, p * 0.4],
        }

    def _ta_sub_err_prob(p):
        return {"keep_del_sub_probs": [1 - p, 0.0, p]}

    # auxCtcLs0.1 was maybe a bad choice here... Redoing this below.
    # for name, opts in [
    #     # baseline (0): {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    #     ("0", None),
    #     #         A0.1: {"dev-clean": 3.66, "dev-other": 5.50, "test-clean": 3.89, "test-other": 5.63}
    #     ("A0.1", _ta_vA_err_prob(0.1)),
    #     #         A0.2: {"dev-clean": 3.46, "dev-other": 5.41, "test-clean": 3.74, "test-other": 5.56}
    #     ("A0.2", _ta_vA_err_prob(0.2)),
    #     #       Sub0.1: {"dev-clean": 3.47, "dev-other": 5.30, "test-clean": 3.81, "test-other": 5.55}
    #     ("Sub0.1", _ta_sub_err_prob(0.1)),
    # ]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-textAug{name}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             # More futureproof, but also required for some funcs / setups.
    #             "behavior_version": 24,
    #             "__serialization_version": 2,
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #             "feature_batch_norm": True,
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "aux_loss_layers": [4, 10, 16],
    #             "aux_ctc_label_smoothing": 0.1,
    #             **({"text_augment": functools.partial(text_augment, **opts)} if opts else {}),
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # Again but without aux CTC loss LS (which seems to be suboptimal).
    # Unclear... Too much?
    # TODO more... less errs. maybe no ins?
    for name, opts in [
        # {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
        ("0", None),
        # {"dev-clean": 3.29, "dev-other": 5.14, "test-clean": 3.82, "test-other": 5.62}
        ("A0.1", _ta_vA_err_prob(0.1)),
        # {"dev-clean": 3.67, "dev-other": 5.25, "test-clean": 4.07, "test-other": 5.66}
        ("A0.2", _ta_vA_err_prob(0.2)),
        # {"dev-clean": 3.95, "dev-other": 5.49, "test-clean": 4.41, "test-other": 6.15}
        ("Sub0.1", _ta_sub_err_prob(0.1)),
    ]:
        aed_train_exp(
            f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-textAug{name}-spm10k-bpeSample001-baseLr0.5-b100k",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                # More futureproof, but also required for some funcs / setups.
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                **({"text_augment": functools.partial(text_augment, **opts)} if opts else {}),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Try mult EOS labels (multEOS).
    # baseline (like multEOS1): {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    #                 multEOS3: {"dev-clean": 4.39, "dev-other": 5.99, "test-clean": 4.81, "test-other": 6.71}
    # Bad. But do another variant below.
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-multEOS3-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "aux_ctc_label_smoothing": 0.1,
    #         "text_augment": functools.partial(
    #             text_augment,
    #             ins_probs=[1.0],  # no insertions
    #             ins_probs_last_frame=[0.0, 0.0, 1.0],  # always insert 2 labels at end, i.e. we have 3 EOS target labels
    #             keep_del_sub_probs=[1.0, 0.0, 0.0],  # always keep, no deletions, no substitutions
    #         ),
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Again mult EOS labels (multEOS), but only 1 extra EOS, and better setting (no aux CTC loss LS).
    # baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #      multEOS2: {"dev-clean": 3.53, "dev-other": 5.18, "test-clean": 4.19, "test-other": 5.72}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-multEOS2-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "text_augment": functools.partial(
    #             text_augment,
    #             ins_probs=[1.0],  # no insertions
    #             ins_probs_last_frame=[0.0, 1.0],  # always insert 1 label at end, i.e. we have 2 EOS target labels
    #             keep_del_sub_probs=[1.0, 0.0, 0.0],  # always keep, no deletions, no substitutions
    #         ),
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Try custom LR multipliers.
    from i6_experiments.users.zeyer.returnn.updater.lr_multiplier import optimizer_param_groups_custom_lr_multiplier

    # Note, baseline here is a bit suboptimal (auxCtcLs0.1, also based on s2).
    # for name, (lrs, lr_mult_by_patterns) in {
    #     # Baseline (None) lr 0.5: {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    #     "None": ([0.5], None),
    #     #          Dec0.1 lr 0.5: {"dev-clean": 5.39, "dev-other": 7.18, "test-clean": 7.51, "test-other": 7.63}
    #     "Dec0.1": ([0.5], {"decoder.*": 0.1}),
    #     #          Dec0.5 lr 0.5: {"dev-clean": 4.30, "dev-other": 6.08, "test-clean": 4.96, "test-other": 5.87}
    #     #          Dec0.5 lr 1.0: {"dev-clean": 4.67, "dev-other": 6.32, "test-clean": 5.23, "test-other": 6.41}
    #     "Dec0.5": ([0.5, 1.0], {"decoder.*": 0.5}),
    #     #  Enc2IncrDec0.5 lr 0.5: {"dev-clean": 4.17, "dev-other": 5.82, "test-clean": 4.70, "test-other": 6.13}
    #     "Enc2IncrDec0.5": (
    #         [0.5],
    #         {
    #             "feature_batch_norm.*": 2.0,
    #             "encoder.input_layer.*": 2.0,
    #             "encoder.input_projection.*": 2.0,
    #             **{f"encoder.layers.{i}.*": 2.0 - i / 15 for i in range(16)},
    #             "enc_aux_logits_*": 1.0,
    #             "decoder.*": 0.5,
    #         },
    #     ),
    #     # Enc2IncrDec1 lr 0.5: {"dev-clean": 4.03, "dev-other": 5.86, "test-clean": 4.52, "test-other": 6.24}
    #     "Enc2IncrDec1": (
    #         [0.5],
    #         {
    #             "feature_batch_norm.*": 2.0,
    #             "encoder.input_layer.*": 2.0,
    #             "encoder.input_projection.*": 2.0,
    #             **{f"encoder.layers.{i}.*": 2.0 - i / 15 for i in range(16)},
    #             "enc_aux_logits_*": 1.0,
    #             "decoder.*": 1.0,
    #         },
    #     ),
    #     # Enc2IncrDec1Incr lr 0.5: {"dev-clean": 4.06, "dev-other": 5.78, "test-clean": 4.49, "test-other": 6.18}
    #     "Enc2IncrDec1Incr": (
    #         [0.5],
    #         {
    #             "feature_batch_norm.*": 2.0,
    #             "encoder.input_layer.*": 2.0,
    #             "encoder.input_projection.*": 2.0,
    #             **{f"encoder.layers.{i}.*": 2.0 - i / 15 for i in range(16)},
    #             "enc_aux_logits_*": 1.0,
    #             "decoder.input_embedding.*": 1.0,
    #             **{f"decoder.layers.{i}.*": 1.0 - 0.5 * i / 5 for i in range(6)},
    #             "decoder.final_layer_norm.*": 0.5,
    #             "decoder.logits.*": 0.5,
    #         },
    #     ),
    # }.items():
    #     for lr in lrs:
    #         aed_train_exp(
    #             f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-customLr{name}-spm10k-bpeSample001-baseLr{lr}-b100k",
    #             config_96gb_bf16_accgrad1,
    #             prefix=prefix + "/aed/",
    #             model_config={
    #                 # More futureproof, but also required for some funcs / setups.
    #                 "behavior_version": 24,
    #                 "__serialization_version": 2,
    #                 "enc_build_dict": rf.build_dict(
    #                     ConformerEncoder,
    #                     input_layer=rf.build_dict(
    #                         ConformerConvSubsample,
    #                         out_dims=[32, 64, 64],
    #                         filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                         pool_sizes=[(1, 2)],
    #                         strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                     ),
    #                     num_layers=16,
    #                     out_dim=1024,
    #                     encoder_layer=rf.build_dict(
    #                         ConformerEncoderLayer,
    #                         ff=rf.build_dict(
    #                             ConformerPositionwiseFeedForward,
    #                             activation=rf.build_dict(rf.relu_square),
    #                             with_bias=False,
    #                         ),
    #                         num_heads=8,
    #                     ),
    #                 ),
    #                 # Default AED decoder size: 6 layers, 512 dim
    #                 "dec_build_dict": rf.build_dict(
    #                     TransformerDecoder,
    #                     num_layers=6,
    #                     model_dim=1024,
    #                     norm=rf.build_dict(rf.RMSNorm),
    #                     ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                     layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                     # When only trained on LS ASR data, keep the default dropout?
    #                     # dropout=0.0,
    #                     # att_dropout=0.0,
    #                 ),
    #                 "feature_batch_norm": True,
    #             },
    #             config_updates={
    #                 **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=lr),
    #                 "batch_size": 100_000 * _batch_size_factor,
    #                 "optimizer.weight_decay": 1e-2,
    #                 **(
    #                     {
    #                         "optimizer.param_groups_custom": optimizer_param_groups_custom_lr_multiplier,
    #                         "optimizer.learning_rate_multipliers_by_patterns": lr_mult_by_patterns,
    #                     }
    #                     if lr_mult_by_patterns
    #                     else {}
    #                 ),
    #                 "accum_grad_multiple_step": 1,
    #                 "__train_audio_preprocess": speed_pert_librosa_config,
    #                 "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #                 "aux_loss_layers": [4, 10, 16],
    #                 "aux_ctc_label_smoothing": 0.1,
    #                 "max_seq_length_default_target": None,
    #                 # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #                 # out of 281241 seqs in train, we removed only 71 seqs.
    #                 # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #                 "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #             },
    #             post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #             vocab="spm10k",
    #             # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #             train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #             dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #             env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #         )

    # Again without aux CTC LS (but still based on s2). All using baseLr0.5.
    for name, lr_mult_by_patterns in {
        # {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.4}
        "None": None,
        # {"dev-clean": 3.02, "dev-other": 5.01, "test-clean": 3.38, "test-other": 5.36}
        "Dec0.5": {"decoder.*": 0.5},
        # {"dev-clean": 3.39, "dev-other": 5.2, "test-clean": 3.98, "test-other": 5.75}
        "Dec2.0": {"decoder.*": 2.0},
        # {"dev-clean": 3.24, "dev-other": 5.09, "test-clean": 4.37, "test-other": 5.71}
        "Enc1IncrA": {f"encoder.layers.{i}.*": 1.0 - 0.5 * i / 15 for i in range(16)},
        # {"dev-clean": 3.67, "dev-other": 5.37, "test-clean": 3.92, "test-other": 5.89}
        "Enc2IncrA": {
            "feature_batch_norm.*": 2.0,
            "encoder.input_layer.*": 2.0,
            "encoder.input_projection.*": 2.0,
            **{f"encoder.layers.{i}.*": 2.0 - i / 15 for i in range(16)},
            "enc_aux_logits_*": 2.0,
        },
    }.items():
        aed_train_exp(
            f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-customLr{name}-spm10k-bpeSample001-baseLr0.5-b100k",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                # More futureproof, but also required for some funcs / setups.
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward,
                            activation=rf.build_dict(rf.relu_square),
                            with_bias=False,
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                **(
                    {
                        "optimizer.param_groups_custom": optimizer_param_groups_custom_lr_multiplier,
                        "optimizer.learning_rate_multipliers_by_patterns": lr_mult_by_patterns,
                    }
                    if lr_mult_by_patterns
                    else {}
                ),
                "accum_grad_multiple_step": 1,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # CTC aux logits without bias (auxNoBias).
    # Baseline:  {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    # auxNoBias: {"dev-clean": 4.31, "dev-other": 5.77, "test-clean": 4.56, "test-other": 6.05}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-auxNoBias-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "enc_aux_logits_with_bias": False,
    #         "aux_ctc_label_smoothing": 0.1,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # CTC aux logits without bias (auxNoBias) again but better baseline (no aux CTC LS).
    # Baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # auxNoBias: ...
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxNoBias-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "enc_aux_logits_with_bias": False,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # CTC aux logits shared (auxShared) (enc_aux_logits_share_weights=True).
    # Baseline:  {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    # auxShared: {"dev-clean": 3.63, "dev-other": 5.62, "test-clean": 3.84, "test-other": 5.88}
    # But auxCtcLs0.1 bad... Do again without.
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-auxShared-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "enc_aux_logits_share_weights": True,
    #         "aux_ctc_label_smoothing": 0.1,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Again auxShared but without aux CTC loss label smoothing.
    # Baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #     auxShared: {"dev-clean": 3.00, "dev-other": 5.14, "test-clean": 3.25, "test-other": 5.29}
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxShared-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "enc_aux_logits_share_weights": True,
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Aux decoder layer (auxDec).
    # Baseline:  {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    #   auxDec3: {"dev-clean": 3.90, "dev-other": 5.54, "test-clean": 4.31, "test-other": 5.79}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k",
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerConvSubsample,
    #                 out_dims=[32, 64, 64],
    #                 filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                 pool_sizes=[(1, 2)],
    #                 strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #             ),
    #             num_layers=16,
    #             out_dim=1024,
    #             encoder_layer=rf.build_dict(
    #                 ConformerEncoderLayer,
    #                 ff=rf.build_dict(
    #                     ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                 ),
    #                 num_heads=8,
    #             ),
    #         ),
    #         # Default AED decoder size: 6 layers, 512 dim
    #         "dec_build_dict": rf.build_dict(
    #             TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             # When only trained on LS ASR data, keep the default dropout?
    #             # dropout=0.0,
    #             # att_dropout=0.0,
    #         ),
    #         "feature_batch_norm": True,
    #     },
    #     config_updates={
    #         **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #         "batch_size": 100_000 * _batch_size_factor,
    #         "optimizer.weight_decay": 1e-2,
    #         "accum_grad_multiple_step": 1,
    #         "__train_audio_preprocess": speed_pert_librosa_config,
    #         "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #         "aux_loss_layers": [4, 10, 16],
    #         "dec_aux_loss_layers": [3],
    #         "aux_ctc_label_smoothing": 0.1,
    #         "max_seq_length_default_target": None,
    #         # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #         # out of 281241 seqs in train, we removed only 71 seqs.
    #         # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #         "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #     },
    #     post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #     vocab="spm10k",
    #     # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #     train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #     dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #     env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    # )

    # Again aux decoder layer (auxDec) but without aux CTC loss label smoothing.
    # Baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #       auxDec3: {"dev-clean": 2.87, "dev-other": 4.71, "test-clean": 2.79, "test-other": 5.03}
    aed_train_exp(
        "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            # More futureproof, but also required for some funcs / setups.
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # variational noise (variational_noise_by_pattern)
    for name, opts in {
        # baseline (0): {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
        "0": None,
        "Dec0.0005": {"decoder.*": 0.0005},
        "Dec0.0025": {"decoder.*": 0.0025},
        "Dec0.01": {"decoder.*": 0.01},
    }.items():
        aed_train_exp(
            f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-vn{name}-spm10k-bpeSample001-baseLr0.5-b100k",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                # More futureproof, but also required for some funcs / setups.
                "behavior_version": 24,
                "__serialization_version": 2,
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
                "feature_batch_norm": True,
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
                "batch_size": 100_000 * _batch_size_factor,
                "optimizer.weight_decay": 1e-2,
                "accum_grad_multiple_step": 1,
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "aux_ctc_label_smoothing": 0.1,
                **({"variational_noise_by_pattern": opts} if opts else {}),
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Try with standard Transformer decoder.
    aed_train_exp(
        "EncL16-DecL6-DecStd-D1024-aux4_10_16-spm10k-spmSample07-b100k",
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            "enc_build_dict": rf.build_dict(
                ConformerEncoder,
                input_layer=rf.build_dict(
                    ConformerConvSubsample,
                    out_dims=[32, 64, 64],
                    filter_sizes=[(3, 3), (3, 3), (3, 3)],
                    pool_sizes=[(1, 2)],
                    strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                ),
                num_layers=16,
                out_dim=1024,
                encoder_layer=rf.build_dict(
                    ConformerEncoderLayer,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
            ),
            # Default AED decoder size: 6 layers, 512 dim
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                # When only trained on LS ASR data, keep the default dropout?
                # dropout=0.0,
                # att_dropout=0.0,
            ),
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
            "optimizer.weight_decay": 1e-2,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )

    # Standard Transformer decoder more label smoothing.
    for label_smoothing in [0.0, 0.1, 0.2]:
        aed_train_exp(
            f"EncL16-DecL6-DecStd-D1024-aux4_10_16-ls{label_smoothing}-spm10k-spmSample07-b100k",
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                "enc_build_dict": rf.build_dict(
                    ConformerEncoder,
                    input_layer=rf.build_dict(
                        ConformerConvSubsample,
                        out_dims=[32, 64, 64],
                        filter_sizes=[(3, 3), (3, 3), (3, 3)],
                        pool_sizes=[(1, 2)],
                        strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
                    ),
                    num_layers=16,
                    out_dim=1024,
                    encoder_layer=rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                ),
                # Default AED decoder size: 6 layers, 512 dim
                "dec_build_dict": rf.build_dict(
                    TransformerDecoder,
                    num_layers=6,
                    model_dim=1024,
                    # When only trained on LS ASR data, keep the default dropout?
                    # dropout=0.0,
                    # att_dropout=0.0,
                ),
            },
            config_updates={
                **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
                "optimizer.weight_decay": 1e-2,
                **({"label_smoothing": label_smoothing} if label_smoothing != 0.1 else {}),
                "__train_audio_preprocess": speed_pert_librosa_config,
                "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                "aux_loss_layers": [4, 10, 16],
                "max_seq_length_default_target": None,
                # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
                # out of 281241 seqs in train, we removed only 71 seqs.
                # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
                "max_seq_length_default_input": 19.5 * _raw_sample_rate,
            },
            post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
            vocab="spm10k",
            train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )

    # Try stochastic depth in decoder.
    # 0.0: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # 0.1: {"dev-clean": 2.87, "dev-other": 5.25, "test-clean": 2.96, "test-other": 5.53}
    # 0.2: {"dev-clean": 3.19, "dev-other": 5.48, "test-clean": 3.22, "test-other": 5.78}
    # 0.3: {"dev-clean": 2.60, "dev-other": 5.07, "test-clean": 2.80, "test-other": 5.53}
    # 0.4: {"dev-clean": 2.82, "dev-other": 5.30, "test-clean": 3.53, "test-other": 5.49}
    # for layer_drop in [0.0, 0.1, 0.2, 0.3, 0.4]:
    #     aed_train_exp(
    #         f"EncL16-DecL6-D1024-DecPosEncAbs-DecLayerDrop{layer_drop}-spm10k-bpeSample001-baseLr0.5-b100k",
    #         config_96gb_bf16_accgrad1,
    #         prefix=prefix + "/aed/",
    #         model_config={
    #             "enc_build_dict": rf.build_dict(
    #                 ConformerEncoder,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_layers=16,
    #                 out_dim=1024,
    #                 encoder_layer=rf.build_dict(
    #                     ConformerEncoderLayer,
    #                     ff=rf.build_dict(
    #                         ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
    #                     ),
    #                     num_heads=8,
    #                 ),
    #             ),
    #             # Default AED decoder size: 6 layers, 512 dim
    #             "dec_build_dict": rf.build_dict(
    #                 TransformerDecoder,
    #                 num_layers=6,
    #                 model_dim=1024,
    #                 norm=rf.build_dict(rf.RMSNorm),
    #                 ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #                 layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #                 **(
    #                     dict(sequential=functools.partial(SequentialLayerDrop, layer_drop=layer_drop))
    #                     if layer_drop
    #                     else {}
    #                 ),
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #             **({"__serialization_version": 2} if layer_drop else {}),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
    #             "batch_size": 100_000 * _batch_size_factor,
    #             "optimizer.weight_decay": 1e-2,
    #             "accum_grad_multiple_step": 1,
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )

    # TODO lstm decoder
    # TODO joint trafo+lstm decoder

    # TODO prior/ILM?
    # TODO recog, also with LM, maybe ILM

    # TODO recog by default with joint CTC+AED. but do timesync with CTC. also recomb with CTC. maybe collapse CTC.


class RMSNormGemma(rf.Module):
    """
    RMSNorm with ...*(1+scale) as in Gemma.
    Should behave better with weight decay.
    (Note that LayerNorm/RMSNorm is usually excluded from weight decay.
     However, RMSNormGemma could be used with weight decay.
     RMSNormGemma without weight decay should behave just exactly the same as RMSNorm.)
    """

    def __init__(self, in_dim: Union[rf.Dim, Sequence[rf.Dim]], *, eps: float = 1e-6, with_bias: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.eps = eps
        self.scale = rf.Parameter([self.in_dim] if isinstance(self.in_dim, rf.Dim) else self.in_dim)
        self.scale.initial = 0.0
        self.bias = None
        if with_bias:
            self.bias = rf.Parameter(self.scale.dims)
            self.bias.initial = 0.0

    def __call__(self, x: Tensor) -> Tensor:
        variance = rf.reduce_mean(rf.square(x), axis=self.in_dim)
        norm_x = x * rf.rsqrt(variance + self.eps)
        out = norm_x * (self.scale + 1.0)
        if self.bias is not None:
            out += self.bias
        return out


class ConformerInputLayerExt(rf.Module):
    """
    Has the input_projection in here.
    Also does further processing.
    """

    def __init__(
        self,
        in_dim: Dim,
        out_dim: Dim,
        *,
        input_layer: Dict[str, Any],
        num_prefix_frames: int = 0,
        num_postfix_frames: int = 0,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # Follow ConformerEncoder code.
        if isinstance(input_layer, dict):
            input_layer = rf.build_from_dict(input_layer, in_dim)
            input_layer: ConformerConvSubsample  # maybe not true, but assume for some attribs
        else:
            raise TypeError(f"unexpected input_layer {input_layer!r}")
        self.input_layer = input_layer
        self.input_projection = rf.Linear(self.input_layer.out_dim, self.out_dim, with_bias=False)

        self.num_prefix_frames = Dim(num_prefix_frames, name="prefix_frames")
        self.num_postfix_frames = Dim(num_postfix_frames, name="postfix_frames")
        self.prefix_frames_embeds = None
        self.postfix_frames_embeds = None
        if num_prefix_frames > 0:
            self.prefix_frames_embeds = rf.Parameter([self.num_prefix_frames, self.out_dim])
        if num_postfix_frames > 0:
            self.postfix_frames_embeds = rf.Parameter([self.num_postfix_frames, self.out_dim])

    def __call__(self, x: Tensor, *, in_spatial_dim: Dim) -> Tuple[Tensor, Dim]:
        """forward"""
        x, spatial_dim = self.input_layer(x, in_spatial_dim=in_spatial_dim)
        x = self.input_projection(x)
        if self.num_prefix_frames.dimension > 0 or self.num_postfix_frames.dimension > 0:
            x, spatial_dim = rf.concat(
                *(
                    (
                        [(rf.cast(self.prefix_frames_embeds, x.dtype), self.num_prefix_frames)]
                        if self.prefix_frames_embeds is not None
                        else []
                    )
                    + [(x, spatial_dim)]
                    + (
                        [(rf.cast(self.postfix_frames_embeds, x.dtype), self.num_postfix_frames)]
                        if self.postfix_frames_embeds is not None
                        else []
                    )
                ),
                allow_broadcast=True,
                handle_dynamic_dims=True,
            )
        return x, spatial_dim
