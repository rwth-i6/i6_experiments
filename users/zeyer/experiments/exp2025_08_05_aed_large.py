"""
New AED baseline tuning for larger AED model.
"""

from __future__ import annotations

from typing import Union, Any, Sequence, Dict, Tuple
import functools
from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module

from i6_experiments.users.zeyer.datasets.librispeech import get_librispeech_task_raw_v2
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
    config_96gb_bf16_accgrad1,
    _get_cfg_lrlin_oclr_by_bs_nep_v4,
    _batch_size_factor,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
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
    task_spm10k = get_librispeech_task_raw_v2(vocab="spm10k")

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

    # varying learning rate
    # for lr in [0.1, 0.5, 1.0]:
    #     # lr 0.1: 5.39, {"dev-clean": 2.5, "dev-other": 5.39, "test-clean": 2.88, "test-other": 5.82}
    #     # +CTC: {"dev-clean": 2.06, "dev-other": 5.12, "test-clean": 2.25, "test-other": 5.22}
    #     # lr 0.5: 5.16, {"dev-clean": 2.73, "dev-other": 5.16, "test-clean": 2.95, "test-other": 5.66}
    #     # +CTC: {"dev-clean": 2.03, "dev-other": 4.59, "test-clean": 2.1, "test-other": 4.8}
    #     # lr 1.0: 5.66, {"dev-clean": 3.51, "dev-other": 5.66, "test-clean": 3.54, "test-other": 5.85}
    #     # +CTC: {"dev-clean": 1.91, "dev-other": 4.64, "test-clean": 2.11, "test-other": 4.92}
    #     # lr 2.0: 14.36
    #     name = f"EncL16-DecL6-D1024-spm10k-bpeSample001-baseLr{lr}-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=8,
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
    #     Baseline: {"dev-clean": 3.61, "dev-other": 5.72, "test-clean": 4.18, "test-other": 5.84}
    # DecPosEncAbs: {"dev-clean": 3.03, "dev-other": 5.18, "test-clean": 3.21, "test-other": 5.64}
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
    # +CTC: {"dev-clean": 1.97, "dev-other": 4.59, "test-clean": 2.15, "test-other": 4.8}
    # (But better baselines below, with featBN.)
    name = "EncL16-DecL6-D1024-DecPosEncAbs-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
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
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=8,
    )

    # Again other aux_loss_layers with better baseline.
    # (Better baselines below. Also below again some more on aux_loss_layers.)
    # for aux_loss_layers in [
    #     # {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35} (bad baseline)
    #     # +CTC: {"dev-clean": 1.97, "dev-other": 4.59, "test-clean": 2.15, "test-other": 4.8}
    #     [4, 8],
    #     # {"dev-clean": 3.45, "dev-other": 5.51, "test-clean": 3.95, "test-other": 6.03}
    #     # +CTC: {"dev-clean": 2.0, "dev-other": 4.44, "test-clean": 2.1, "test-other": 4.65}
    #     [6, 12],
    #     # {"dev-clean": 2.57, "dev-other": 5.5, "test-clean": 4.27, "test-other": 6.62}
    #     # +CTC: {"dev-clean": 1.9, "dev-other": 4.28, "test-clean": 2.12, "test-other": 4.64}
    #     [8, 16],
    #     # {"dev-clean": 2.84, "dev-other": 4.85, "test-clean": 3.02, "test-other": 5.2}
    #     # +CTC: {"dev-clean": 1.91, "dev-other": 4.2, "test-clean": 2.08, "test-other": 4.61}
    #     [4, 10, 16],
    #     # {"dev-clean": 3.08, "dev-other": 5.52, "test-clean": 4.0, "test-other": 5.83}
    #     # +CTC: {"dev-clean": 2.01, "dev-other": 4.92, "test-clean": 2.24, "test-other": 5.12}
    #     [16],
    # ]:
    #     name = f"EncL16-DecL6-D1024-DecPosEncAbs-aux{'_'.join(map(str, aux_loss_layers))}-spm10k-bpeSample001-baseLr0.5-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=max(aux_loss_layers),
    #     )

    # Try different decoder dropout. (Default is dropout 0.1, att_dropout 0.1.)
    # dec_drop,dec_att_drop.
    # 0.1,0.1: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # +CTC: {"dev-clean": 1.97, "dev-other": 4.59, "test-clean": 2.15, "test-other": 4.8}
    # 0.1,0.2: {"dev-clean": 2.42, "dev-other": 5.02, "test-clean": 2.83, "test-other": 5.20}
    # +CTC: {"dev-clean": 2.01, "dev-other": 4.66, "test-clean": 2.14, "test-other": 4.78}
    # 0.2,0.1: {"dev-clean": 11.17, "dev-other": 11.08, "test-clean": 16.7, "test-other": 12.04}
    # +CTC: {"dev-clean": 2.66, "dev-other": 5.12, "test-clean": 3.04, "test-other": 5.36}
    # 0.2,0.2: {"dev-clean": 8.84, "dev-other": 9.53, "test-clean": 12.69, "test-other": 10.47}
    # +CTC: {"dev-clean": 2.17, "dev-other": 4.81, "test-clean": 2.43, "test-other": 5.0}
    # for dec_drop, dec_att_drop in [(0.2, 0.2), (0.1, 0.2), (0.2, 0.1)]:
    #     name = f"EncL16-DecL6-D1024-DecPosEncAbs-DecDrop{dec_drop}-DecAttDrop{dec_att_drop}-spm10k-bpeSample001-baseLr0.5-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=8,
    #     )

    # Try different dec model dim.
    # 512: {"dev-clean": 2.9, "dev-other": 5.2, "test-clean": 3.09, "test-other": 5.54}
    # +CTC: {"dev-clean": 1.97, "dev-other": 4.63, "test-clean": 2.15, "test-other": 4.76}
    # 640: {"dev-clean": 2.34, "dev-other": 5.04, "test-clean": 2.46, "test-other": 5.49}
    # +CTC: {"dev-clean": 2.0, "dev-other": 4.85, "test-clean": 2.18, "test-other": 4.9}
    # 768: {"dev-clean": 7.02, "dev-other": 9.33, "test-clean": 10.72, "test-other": 9.85}
    # +CTC: {"dev-clean": 2.31, "dev-other": 4.97, "test-clean": 2.6, "test-other": 5.13}
    # 1024: {"dev-clean": 2.92, "dev-other": 5.04, "test-clean": 3.12, "test-other": 5.58}
    # +CTC: {"dev-clean": 1.98, "dev-other": 4.59, "test-clean": 2.18, "test-other": 4.75}
    # for dec_model_dim in [512, 640, 768, 1024]:
    #     name = f"EncL16-DecL6-D1024-DecD{dec_model_dim}-DecPosEncAbs-spm10k-bpeSample001-baseLr0.5-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=8,
    #     )

    # label smoothing
    # 0.1: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35}
    # +CTC: {"dev-clean": 1.97, "dev-other": 4.59, "test-clean": 2.15, "test-other": 4.8}
    # 0.2: {"dev-clean": 3.43, "dev-other": 5.82, "test-clean": 3.55, "test-other": 6.06}
    # +CTC: {"dev-clean": 1.96, "dev-other": 4.56, "test-clean": 2.14, "test-other": 4.82}
    # for label_smoothing in [0.1, 0.2]:
    #     name = f"EncL16-DecL6-D1024-DecPosEncAbs-ls{label_smoothing}-spm10k-bpeSample001-baseLr0.5-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=8,
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
    # Joint AED+CTC:
    # baseline: {"dev-clean": 1.91, "dev-other": 4.20, "test-clean": 2.08, "test-other": 4.61}
    #   featBN: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    # (But see right below for a better setup (bhv21-s2) as starting point, using the same settings.)
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
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
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k-s2"
    exp = aed_train_exp(
        name,
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
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    # Try again old-vs-new serialization, also old-vs-new behavior version.
    # Try also running it again multiple times (__trigger_new_hash, "hN").
    # Note: fixMp (fix_mp) will fix MultiProcDataset opts for serialization version 2.
    # This will only change num_workers from 4 -> 25. And 4 workers gives much worse results?
    # But what does this change? The only thing that I can see now is the RNG for audio/targets:
    # self._audio_random.seed(random_seed), self.targets.set_random_seed(random_seed).
    # The RNG should then be the same across all 4 or 25 workers.
    # In case of 25 workers, every 25 seqs will have exactly the same kind of speed perturbation.
    # This was not really intended, but this is helping?
    # bhv21-s1-h0: {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # bhv21-s1-h1: {"dev-clean": 2.66, "dev-other": 4.74, "test-clean": 2.85, "test-other": 5.00}
    # bhv21-s2-h0: {"dev-clean": 3.41, "dev-other": 5.56, "test-clean": 3.90, "test-other": 5.77}
    # bhv21-s2-h1: {"dev-clean": 3.52, "dev-other": 5.26, "test-clean": 3.73, "test-other": 5.61}
    # bhv21-s2-fixMp-h0: {"dev-clean": 2.81, "dev-other": 4.89, "test-clean": 3.07, "test-other": 5.04}
    # bhv23-s2-h0: {"dev-clean": 3.58, "dev-other": 5.09, "test-clean": 3.84, "test-other": 5.66}
    # bhv24-s1-h0: {"dev-clean": 3.54, "dev-other": 5.65, "test-clean": 4.05, "test-other": 6.28}
    # bhv24-s2-h0: {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # bhv24-s2-fixMp-h0: {"dev-clean": 3.69, "dev-other": 5.36, "test-clean": 3.57, "test-other": 5.53}
    # All with AED+CTC:
    # bhv21-s1-h0: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    # bhv21-s1-h1: {"dev-clean": 1.87, "dev-other": 4.30, "test-clean": 2.14, "test-other": 4.56}
    # bhv21-s2-h0: {"dev-clean": 1.90, "dev-other": 4.19, "test-clean": 2.12, "test-other": 4.48}
    # bhv21-s2-h1: {"dev-clean": 1.86, "dev-other": 4.17, "test-clean": 2.06, "test-other": 4.53}
    # bhv21-s2-fixMp-h0: {"dev-clean": 1.89, "dev-other": 4.24, "test-clean": 2.04, "test-other": 4.46}
    # bhv23-s2-h0: {"dev-clean": 1.9, "dev-other": 4.24, "test-clean": 2.07, "test-other": 4.47}
    # bhv24-s1-h0: {"dev-clean": 1.87, "dev-other": 4.31, "test-clean": 2.13, "test-other": 4.49}
    # bhv24-s2-h0: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
    # bhv24-s2-fixMp-h0: {"dev-clean": 1.92, "dev-other": 4.32, "test-clean": 2.07, "test-other": 4.49}
    # -> It seems like there is not really much difference with CTC, so maybe it's all noise/variance...
    # fixMp seems to help a little bit though...
    for bhv, sv, hv, fix_mp in [
        (21, 1, 0, False),
        # (21, 1, 1, False),
        # (21, 2, 0, False),
        # (21, 2, 1, False),
        # (21, 2, 0, True),
        # (23, 2, 0, False),
        # (24, 1, 0, False),
        (24, 2, 0, False),
        (24, 2, 0, True),
    ]:
        name = (
            f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
            f"-bhv{bhv}-s{sv}-{'fixMp-' if fix_mp else ''}h{hv}"
        )
        exp = aed_train_exp(
            name,
            config_96gb_bf16_accgrad1,
            prefix=prefix + "/aed/",
            model_config={
                "behavior_version": bhv,
                **({"__serialization_version": sv} if sv != 1 else {}),
                **({"__trigger_new_hash": hv} if hv != 0 else {}),
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
            post_config_updates={
                "log_grad_norm": True,
                "__multi_proc_dataset" if fix_mp else "__multi_proc_dataset_opts": {"num_workers": 25},
            },
            vocab="spm10k",
            # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
            train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
            dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
            env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        )
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
        )

    # Disable some of the masking that is done with behavior version 24 (in training) (noConvMasks).
    # Baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #   noConvMasks: {"dev-clean": 3.56, "dev-other": 5.26, "test-clean": 3.64, "test-other": 5.73}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k-s2-noConvMasks",
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
    #         "rf_use_mask_conv": False,
    #         "rf_use_mask_pool": False,
    #         "rf_use_mask_stft": False,
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

    # Also try abs pos enc in encoder (EncPosEncAbs) (compare this to the ...-s2 above)
    # baseline (s1): {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    # EncPosEncAbs: {"dev-clean": 3.55, "dev-other": 5.25, "test-clean": 4.17, "test-other": 5.64}
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-EncPosEncAbs-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k",
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
    #             pos_enc=rf.build_dict(rf.sinusoidal_positional_encoding),
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

    # EncAddEos: add an EOS token to the encoder output (before passing to the decoder).
    # baseline (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #          +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
    #     EncAddEos: {"dev-clean": 3.19, "dev-other": 5.17, "test-clean": 3.50, "test-other": 5.61}
    #          +CTC: {"dev-clean": 1.97, "dev-other": 4.32, "test-clean": 2.11, "test-other": 4.62}
    # name = "EncL16-DecL6-D1024-EncAddEos-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    # enc_dim = Dim(1024, name="enc_feat")
    # exp = aed_train_exp(
    #     name,
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         # More futureproof, but also required for some funcs / setups.
    #         "behavior_version": 24,
    #         "__serialization_version": 2,
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 ConformerInputLayerExt,
    #                 out_dim=enc_dim,
    #                 input_layer=rf.build_dict(
    #                     ConformerConvSubsample,
    #                     out_dims=[32, 64, 64],
    #                     filter_sizes=[(3, 3), (3, 3), (3, 3)],
    #                     pool_sizes=[(1, 2)],
    #                     strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
    #                 ),
    #                 num_postfix_frames=1,
    #             ),
    #             num_layers=16,
    #             out_dim=enc_dim,
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
    # del enc_dim
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    # )

    # Pad audio (AudioPad) to somehow have it similar as wrong conv
    # (as in behavior version 21, but here using behavior version 24).
    # Note: the number (eg 1k) is on sample level. 1k means 1000 samples, i.e. ~0.06 sec.
    for name, opts in {
        # Baseline (bhv24-s2):
        # 0: {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.4}
        # +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
        "0": None,
        # 1k: {"dev-clean": 3.51, "dev-other": 5.02, "test-clean": 3.49, "test-other": 5.64}
        # +CTC: {"dev-clean": 1.85, "dev-other": 4.21, "test-clean": 2.13, "test-other": 4.59}
        # "1k": 1000,
        # Rnd2k: {"dev-clean": 3.95, "dev-other": 5.43, "test-clean": 4.44, "test-other": 5.92}
        # +CTC: {"dev-clean": 1.91, "dev-other": 4.19, "test-clean": 2.12, "test-other": 4.57}
        # "Rnd2k": {"train": ((0, 2000), (0, 2000))},
        # Rnd100: {"dev-clean": 3.07, "dev-other": 5.27, "test-clean": 3.83, "test-other": 5.74}
        # +CTC: {"dev-clean": 1.85, "dev-other": 4.19, "test-clean": 2.07, "test-other": 4.33}
        "Rnd100": {"train": ((0, 100), (0, 100))},
    }.items():
        name = f"EncL16-DecL6-D1024-AudioPad{name}-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
        exp = aed_train_exp(
            name,
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
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
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
    # aed_train_exp(
    #     "EncL16-DecL6-D1024-aux4_10_16-spm10k-spmSample07-b100k",
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
    #         "aux_loss_layers": [4, 10, 16],
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

    # Better baseline (DecPosEncAbs, featBN, bpeSample001, baseLr0.5) with CTC label smoothing (auxCtcLs0.1).
    # Baseline without aux CTC loss (s2): {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
    #                        auxCtcLs0.1: {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
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

    def _ta_vA_err_prob(q: float, *, rel_sub: float = 0.4):
        # q=(1-p)^2 will be the approx error rate.
        p = round(1 - (1 - q) ** 0.5, 5)
        return {
            "ins_probs": [1 - p, p * 0.8, p * 0.2],
            "keep_del_sub_probs": [1 - p, p * (1 - rel_sub), p * rel_sub],
        }

    # def _ta_del_sub_err_prob(del_prob, sub_prob):
    #     return {"keep_del_sub_probs": [1 - del_prob - sub_prob, del_prob, sub_prob]}

    # def _ta_sub_err_prob(p):
    #     return {"keep_del_sub_probs": [1 - p, 0.0, p]}

    # auxCtcLs0.1 was maybe a bad choice here... / also s2. Redoing this below.
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
    # Unclear... Too much? Too little? No effect? Only bad effect?
    # TODO what now?
    for name, opts in [
        # 0: {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
        # +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
        ("0", None),
        # A0.01: {"dev-clean": 3.74, "dev-other": 5.39, "test-clean": 4.23, "test-other": 5.71}
        # +CTC: {"dev-clean": 1.92, "dev-other": 4.24, "test-clean": 2.05, "test-other": 4.52}
        # ("A0.01", _ta_vA_err_prob(0.01)),
        # A0.05: {"dev-clean": 3.47, "dev-other": 5.28, "test-clean": 3.73, "test-other": 5.89}
        # +CTC: {"dev-clean": 1.86, "dev-other": 4.19, "test-clean": 2.09, "test-other": 4.49}
        # ("A0.05", _ta_vA_err_prob(0.05)),
        # A0.1: {"dev-clean": 3.29, "dev-other": 5.14, "test-clean": 3.82, "test-other": 5.62}
        # +CTC: {"dev-clean": 1.93, "dev-other": 4.21, "test-clean": 2.06, "test-other": 4.53}
        ("A0.1", _ta_vA_err_prob(0.1)),
        # A0.2: {"dev-clean": 3.67, "dev-other": 5.25, "test-clean": 4.07, "test-other": 5.66}
        # +CTC: {"dev-clean": 1.9, "dev-other": 4.27, "test-clean": 2.15, "test-other": 4.52}
        ("A0.2", _ta_vA_err_prob(0.2)),
        ("A0.3", _ta_vA_err_prob(0.3)),
        ("A0.1S0", _ta_vA_err_prob(0.1, rel_sub=0.0)),
        # Sub0.1: {"dev-clean": 3.95, "dev-other": 5.49, "test-clean": 4.41, "test-other": 6.15}
        # +CTC: {"dev-clean": 2.42, "dev-other": 4.28, "test-clean": 2.48, "test-other": 4.83}
        # ("Sub0.1", _ta_sub_err_prob(0.1)),
        # Del0.05Sub0.05: {"dev-clean": 4.19, "dev-other": 5.57, "test-clean": 4.59, "test-other": 5.82}
        # +CTC: {"dev-clean": 1.96, "dev-other": 4.32, "test-clean": 2.23, "test-other": 4.55}
        # ("Del0.05Sub0.05", _ta_del_sub_err_prob(0.05, 0.05)),
    ]:
        name = f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-textAug{name}-spm10k-bpeSample001-baseLr0.5-b100k"
        exp = aed_train_exp(
            name,
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
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
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

    # multEOS2 again but now with ctcEos (use_eos_postfix) (also using bhv24, s2, fixMp).
    # Baseline
    # (bhv24-s2-fixMp): {"dev-clean": 3.69, "dev-other": 5.36, "test-clean": 3.57, "test-other": 5.53}
    #             +CTC: {"dev-clean": 1.92, "dev-other": 4.32, "test-clean": 2.07, "test-other": 4.49}
    # multEOS2: TODO...
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-multEOS2-ctcEos-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
            "use_eos_postfix": True,  # in model because we need to know about this in joint AED+CTC recog
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "text_augment": functools.partial(
                text_augment,
                ins_probs=[1.0],  # no insertions
                ins_probs_last_frame=[0.0, 1.0],  # always insert 1 label at end, i.e. we have 2 EOS target labels
                keep_del_sub_probs=[1.0, 0.0, 0.0],  # always keep, no deletions, no substitutions
            ),
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    # in joint AED+CTC model_recog_with_recomb, we mask ctc log probs for EOS to -inf
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

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

    # Again customLr without aux CTC LS (but still based on s2). All using baseLr0.5.
    # TODO what now?
    for name, lr_mult_by_patterns in {
        # None: {"dev-clean": 3.09, "dev-other": 4.97, "test-clean": 3.49, "test-other": 5.40}
        # +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
        "None": None,
        # Dec0.1: TODO...
        #   +CTC: TODO...
        "Dec0.1": {"decoder.*": 0.1},
        # Dec0.5: {"dev-clean": 3.02, "dev-other": 5.01, "test-clean": 3.38, "test-other": 5.36}
        #   +CTC: {"dev-clean": 1.82, "dev-other": 4.21, "test-clean": 2.07, "test-other": 4.43}
        "Dec0.5": {"decoder.*": 0.5},
        # Dec2.0: {"dev-clean": 3.39, "dev-other": 5.20, "test-clean": 3.98, "test-other": 5.75}
        #   +CTC: {"dev-clean": 1.89, "dev-other": 4.23, "test-clean": 2.06, "test-other": 4.48}
        "Dec2.0": {"decoder.*": 2.0},
        # Enc1IncrA: {"dev-clean": 3.24, "dev-other": 5.09, "test-clean": 4.37, "test-other": 5.71}
        #      +CTC: {"dev-clean": 1.87, "dev-other": 4.34, "test-clean": 2.11, "test-other": 4.58}
        # "Enc1IncrA": {f"encoder.layers.{i}.*": 1.0 - 0.5 * i / 15 for i in range(16)},
        # Enc2IncrA: {"dev-clean": 3.67, "dev-other": 5.37, "test-clean": 3.92, "test-other": 5.89}
        #      +CTC: {"dev-clean": 1.89, "dev-other": 4.32, "test-clean": 2.09, "test-other": 4.61}
        # "Enc2IncrA": {
        #     "feature_batch_norm.*": 2.0,
        #     "encoder.input_layer.*": 2.0,
        #     "encoder.input_projection.*": 2.0,
        #     **{f"encoder.layers.{i}.*": 2.0 - i / 15 for i in range(16)},
        #     "enc_aux_logits_*": 2.0,
        # },
    }.items():
        name = f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-customLr{name}-spm10k-bpeSample001-baseLr0.5-b100k"
        exp = aed_train_exp(
            name,
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
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
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
    #          +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
    #     auxNoBias: {"dev-clean": 3.42, "dev-other": 5.16, "test-clean": 3.69, "test-other": 5.29}
    #          +CTC: {"dev-clean": 1.84, "dev-other": 4.24, "test-clean": 2.09, "test-other": 4.46}
    # name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxNoBias-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    #     extra_config={"enc_aux_logits_with_bias": False},
    # )

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
    #          +CTC: {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
    #     auxShared: {"dev-clean": 3.00, "dev-other": 5.14, "test-clean": 3.25, "test-other": 5.29}
    #          +CTC: {"dev-clean": 1.94, "dev-other": 4.33, "test-clean": 2.08, "test-other": 4.43}
    # name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxShared-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    #     extra_config={"aux_loss_layers": [4, 10, 16], "enc_aux_logits_share_weights": True},
    # )

    # Aux decoder layer (auxDec).
    # Baseline:  {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
    #   auxDec3: {"dev-clean": 3.90, "dev-other": 5.54, "test-clean": 4.31, "test-other": 5.79}
    # (Do again below with better baseline.)
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
    # +CTC:
    # Baseline (s2): {"dev-clean": 1.88, "dev-other": 4.27, "test-clean": 2.12, "test-other": 4.51}
    #       auxDec3: {"dev-clean": 1.87, "dev-other": 4.06, "test-clean": 2.06, "test-other": 4.38}
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    # variational noise (variational_noise_by_pattern)
    for name, opts in {
        # baseline (0): {"dev-clean": 4.27, "dev-other": 5.67, "test-clean": 4.41, "test-other": 5.93}
        # +CTC: {"dev-clean": 2.42, "dev-other": 4.43, "test-clean": 2.35, "test-other": 4.69}
        "0": None,
        # Dec0.0005: {"dev-clean": 3.97, "dev-other": 5.78, "test-clean": 4.83, "test-other": 6.19}
        # +CTC: {"dev-clean": 2.1, "dev-other": 4.59, "test-clean": 2.63, "test-other": 4.82}
        "Dec0.0005": {"decoder.*": 0.0005},
        # Dec0.0025: {"dev-clean": 4.15, "dev-other": 5.77, "test-clean": 4.83, "test-other": 6.38}
        # +CTC: {"dev-clean": 1.88, "dev-other": 4.18, "test-clean": 2.08, "test-other": 4.6}
        "Dec0.0025": {"decoder.*": 0.0025},
        # Dec0.01: {"dev-clean": 4.24, "dev-other": 5.81, "test-clean": 7.34, "test-other": 6.91}
        # +CTC: {"dev-clean": 1.9, "dev-other": 4.23, "test-clean": 2.07, "test-other": 4.47}
        "Dec0.01": {"decoder.*": 0.01},
        # 0.01: TODO... +CTC...
        "0.01": {"*": 0.01},
    }.items():
        name = f"EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxCtcLs0.1-vn{name}-spm10k-bpeSample001-baseLr0.5-b100k"
        exp = aed_train_exp(
            name,
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
        aed_ctc_timesync_recog_recomb_auto_scale(
            prefix=prefix + "/aed/" + name + "/aed+ctc",
            task=task_spm10k,
            aed_ctc_model=exp.get_last_fixed_epoch(),
            aux_ctc_layer=16,
        )

    # New baseline: bhv24 + s2 + fixMp + auxShared + auxNoBias + auxDec3 + AudioPadRnd100
    name = "EncL16-DecL6-D1024-AudioPadRnd100-DecPosEncAbs-featBN-aux4_10_16-auxShared-auxNoBias-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
            "pad_audio": {"train": ((0, 100), (0, 100))},
            "feature_batch_norm": True,
            "aux_loss_layers": [4, 10, 16],
            "enc_aux_logits_share_weights": True,
            "enc_aux_logits_with_bias": False,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": 100_000 * _batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset": {"num_workers": 25}},
        vocab="spm10k",
        # train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    # TODO new baseline:
    #   based on: EncL16-DecL6-D1024-AudioPadRnd100-DecPosEncAbs-featBN-aux4_10_16-auxShared-auxNoBias-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k
    #   bhv24 + s2 + fixMp + auxShared + auxNoBias + auxDec3 + AudioPadRnd100 + DecLayerDrop0.3 + var noise (?) + ...?

    # Try with standard Transformer decoder (DecStd).
    # baseline: {"dev-clean": 2.82, "dev-other": 4.8, "test-clean": 3.66, "test-other": 5.22}
    #   DecStd: {"dev-clean": 3.4, "dev-other": 5.23, "test-clean": 3.58, "test-other": 5.44}
    # aed_train_exp(
    #     "EncL16-DecL6-DecStd-D1024-aux4_10_16-spm10k-spmSample07-b100k",
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
    #         "aux_loss_layers": [4, 10, 16],
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

    # Standard Transformer decoder more label smoothing.
    # (Note: somewhat outdated baseline. still spmSample007, no featBN)
    # (Also see other label smoothing experiments before,
    #  EncL16-DecL6-D1024-DecPosEncAbs-ls{label_smoothing}-spm10k-bpeSample001-baseLr0.5-b100k)
    # (ESPnet uses LS 0.1 usually...)
    # 0.0: {"dev-clean": 3.40, "dev-other": 5.40, "test-clean": 3.63, "test-other": 5.66}
    # +CTC: {"dev-clean": 1.94, "dev-other": 4.41, "test-clean": 2.16, "test-other": 4.71}
    # 0.1: {"dev-clean": 3.40, "dev-other": 5.23, "test-clean": 3.58, "test-other": 5.44}
    # +CTC: {"dev-clean": 1.94, "dev-other": 4.38, "test-clean": 2.13, "test-other": 4.62}
    # 0.2: {"dev-clean": 3.39, "dev-other": 5.18, "test-clean": 3.59, "test-other": 5.44}
    # +CTC: {"dev-clean": 1.92, "dev-other": 4.28, "test-clean": 2.06, "test-other": 4.57}
    # for label_smoothing in [0.0, 0.1, 0.2]:
    #     name = f"EncL16-DecL6-DecStd-D1024-aux4_10_16-ls{label_smoothing}-spm10k-spmSample07-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #                 # When only trained on LS ASR data, keep the default dropout?
    #                 # dropout=0.0,
    #                 # att_dropout=0.0,
    #             ),
    #         },
    #         config_updates={
    #             **_get_cfg_lrlin_oclr_by_bs_nep_v3(100_000, 100, batch_size_factor=_batch_size_factor),
    #             "optimizer.weight_decay": 1e-2,
    #             **({"label_smoothing": label_smoothing} if label_smoothing != 0.1 else {}),
    #             "__train_audio_preprocess": speed_pert_librosa_config,
    #             "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
    #             "aux_loss_layers": [4, 10, 16],
    #             "max_seq_length_default_target": None,
    #             # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
    #             # out of 281241 seqs in train, we removed only 71 seqs.
    #             # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
    #             "max_seq_length_default_input": 19.5 * _raw_sample_rate,
    #         },
    #         post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
    #         vocab="spm10k",
    #         train_vocab_opts={"other_opts": {"enable_sampling": True, "alpha": 0.7}},
    #         # train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    #         dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
    #         env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    #     )
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=16,
    #     )

    # from i6_experiments.users.zeyer.nn_rf.layerdrop import SequentialLayerDrop

    # Try stochastic depth in decoder. (Suboptimal aux_loss_layers here, implicit aux4_8)
    # (Btw, ESPnet EBranchformer
    #  (https://github.com/espnet/espnet/blob/master/egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml)
    #  uses layer drop 0.1 in encoder, 0.2 in decoder.
    #  But those are smaller models. So maybe even larger layer drop might be optimal here?)
    # 0.0: {"dev-clean": 2.38, "dev-other": 4.87, "test-clean": 2.89, "test-other": 5.35} (compare warning: s1)
    # +CTC: {"dev-clean": 1.97, "dev-other": 4.59, "test-clean": 2.15, "test-other": 4.8}
    # 0.1: {"dev-clean": 2.87, "dev-other": 5.25, "test-clean": 2.96, "test-other": 5.53}
    # +CTC: {"dev-clean": 1.94, "dev-other": 4.69, "test-clean": 2.11, "test-other": 4.84}
    # 0.2: {"dev-clean": 3.19, "dev-other": 5.48, "test-clean": 3.22, "test-other": 5.78}
    # +CTC: {"dev-clean": 1.98, "dev-other": 4.58, "test-clean": 2.2, "test-other": 4.82}
    # 0.3: {"dev-clean": 2.60, "dev-other": 5.07, "test-clean": 2.80, "test-other": 5.53}
    # +CTC: {"dev-clean": 1.87, "dev-other": 4.56, "test-clean": 2.1, "test-other": 4.72}
    # 0.4: {"dev-clean": 2.82, "dev-other": 5.30, "test-clean": 3.53, "test-other": 5.49}
    # +CTC: {"dev-clean": 1.94, "dev-other": 4.73, "test-clean": 2.19, "test-other": 4.72}
    # for layer_drop in [0.0, 0.1, 0.2, 0.3, 0.4]:
    #     name = f"EncL16-DecL6-D1024-DecPosEncAbs-DecLayerDrop{layer_drop}-spm10k-bpeSample001-baseLr0.5-b100k"
    #     exp = aed_train_exp(
    #         name,
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
    #     aed_ctc_timesync_recog_recomb_auto_scale(
    #         prefix=prefix + "/aed/" + name + "/aed+ctc",
    #         task=task_spm10k,
    #         aed_ctc_model=exp.get_last_fixed_epoch(),
    #         aux_ctc_layer=8,
    #     )

    # See i6_experiments/users/zeyer/experiments/exp2024_04_23_baselines/ctc.py for some earlier usage.
    # from returnn.frontend.encoder.e_branchformer import EBranchformerLayer

    # EBranchformer.
    # The EBranchformer has more components which makes it bigger for the same num layers and dim,
    # thus we reduce the dim (896 instead of 1024) to make it comparable in total num params to the Conformer.
    # (But actually, you should look more at absolute speed, not so much at num params...)
    # (But memory consumption with D1024-EBranchformer is still higher, and runs OOM with this batch size...)
    # D1024 (Conformer):  {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # D896-EBranchformer: {"dev-clean": 2.74, "dev-other": 4.84, "test-clean": 2.87, "test-other": 5.22}
    # +CTC:
    # D1024 (Conformer):  {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    # D896-EBranchformer: {"dev-clean": 1.92, "dev-other": 4.31, "test-clean": 2.09, "test-other": 4.53}
    # name = "EncL16-DecL6-D896-EBranchformer-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
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
    #             out_dim=896,
    #             encoder_layer=rf.build_dict(
    #                 EBranchformerLayer,
    #                 ff=rf.build_dict(
    #                     rf.encoder.conformer.ConformerPositionwiseFeedForward,
    #                     # Note: the ffdim in the original EBranchformer is only 1024, but here we use 2048,
    #                     # as this is also what we use for Conformer.
    #                     # (But this results in more parameters for the EBranchformer, due to more params in cgMLP.)
    #                     activation=rf.build_dict(rf.relu_square),
    #                     with_bias=False,
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    # )

    from i6_experiments.users.zeyer.nn_rf.decoder.lstm import LstmDecoder

    # Using LSTM decoder.
    # (Note: Using some older baseline (s1), behavior version 21:
    #  EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k)
    # baseline (trafo dec, s1): {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    #                  DecLstm: {"dev-clean": 2.12, "dev-other": 4.63, "test-clean": 2.31, "test-other": 4.65}
    # Note: See also joint AED+CTC results (recog_ext/aed_ctc.py).
    # +CTC:
    # baseline: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    #  DecLstm: {"dev-clean": 1.95, "dev-other": 4.39, "test-clean": 2.08, "test-other": 4.59}
    name = "EncL16-DecLstm-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    exp = aed_train_exp(
        name,
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
            "dec_build_dict": rf.build_dict(LstmDecoder),
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
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )

    # from i6_experiments.users.zeyer.nn_rf.decoder.lstm_trafo import LstmTransformerDecoderV2

    # Using LSTM Transformer decoder (DecLstmTrafoV2).
    # But we also need auxDec0 here, otherwise it does not converge.
    # baseline (trafo dec, s1): {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    #   DecLstmTrafoV2+auxDec0: {"dev-clean": 2.56, "dev-other": 4.61, "test-clean": 2.60, "test-other": 4.90}
    # +CTC:
    #               baseline: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    # DecLstmTrafoV2+auxDec0: {"dev-clean": 1.88, "dev-other": 4.38, "test-clean": 2.15, "test-other": 4.56}
    # name = "EncL16-DecLstmTrafoV2-featBN-aux4_10_16-auxDec0-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
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
    #         "dec_build_dict": rf.build_dict(
    #             LstmTransformerDecoderV2,
    #             lstm_dim=128,
    #             transformer=rf.build_dict(
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
    #         "dec_aux_loss_layers": [0],
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    # )

    # from i6_experiments.users.zeyer.nn_rf.decoder import trafo_custom_readout

    # Testing custom readout at the top of the Transformer decoder (DecReadoutMax)
    #      Baseline: {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # DecReadoutMax: {"dev-clean": 2.60, "dev-other": 4.96, "test-clean": 2.67, "test-other": 5.27}
    # +CTC:
    #      baseline: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.10, "test-other": 4.50}
    # DecReadoutMax: {"dev-clean": 1.88, "dev-other": 4.33, "test-clean": 2.08, "test-other": 4.56}
    # name = "EncL16-DecL6-D1024-DecPosEncAbs-DecReadoutMax-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
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
    #             trafo_custom_readout.TransformerDecoder,
    #             num_layers=6,
    #             model_dim=1024,
    #             norm=rf.build_dict(rf.RMSNorm),
    #             ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
    #             layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
    #             readout=rf.build_dict(trafo_custom_readout.ReadoutMaxout),
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    # )

    # from i6_experiments.users.zeyer.nn_rf.encoder.blstm_cnn import BlstmCnnEncoder

    # BLSTM frontend.
    # This is based on "cnnblstmf2" (taken from some earlier experiments),
    # which used 512 LSTM dim, but here we reduced it even much more (due to memory/efficiency).
    # Baseline: {"dev-clean": 2.81, "dev-other": 4.72, "test-clean": 2.86, "test-other": 5.08}
    # +CTC: {"dev-clean": 1.86, "dev-other": 4.26, "test-clean": 2.1, "test-other": 4.5}
    # ConvBlstm: {"dev-clean": 2.91, "dev-other": 5.23, "test-clean": 3.84, "test-other": 5.52}
    # +CTC: {"dev-clean": 1.91, "dev-other": 4.43, "test-clean": 2.09, "test-other": 4.73}
    # name = "EncL16-DecL6-D1024-ConvBlstm-DecPosEncAbs-featBN-aux4_10_16-spm10k-bpeSample001-baseLr0.5-b100k"
    # exp = aed_train_exp(
    #     name,
    #     config_96gb_bf16_accgrad1,
    #     prefix=prefix + "/aed/",
    #     model_config={
    #         "enc_build_dict": rf.build_dict(
    #             ConformerEncoder,
    #             input_layer=rf.build_dict(
    #                 BlstmCnnEncoder,
    #                 lstm_dim=128,
    #                 num_layers=2,
    #                 time_reduction=6,
    #                 dropout=0.0,
    #                 allow_pool_last=True,
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
    # aed_ctc_timesync_recog_recomb_auto_scale(
    #     prefix=prefix + "/aed/" + name + "/aed+ctc",
    #     task=task_spm10k,
    #     aed_ctc_model=exp.get_last_fixed_epoch(),
    #     aux_ctc_layer=16,
    # )

    # TODO prior/ILM?
    # TODO recog, also with LM, maybe ILM


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
