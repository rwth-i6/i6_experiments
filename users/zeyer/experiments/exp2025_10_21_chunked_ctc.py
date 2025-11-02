"""
Chunked CTC / Conformer.

For earlier code and some reference, see:
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_conformer.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_ctc.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_aed_import.py

Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition, https://arxiv.org/abs/2309.08436
"""

from __future__ import annotations

from typing import Optional, Any, Dict

from i6_experiments.users.zeyer.utils.sis_setup import get_setup_prefix_for_module
from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed import (
    train_exp as aed_train_exp,
    _raw_sample_rate,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines import configs
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.recog_ext.aed_ctc import (
    aed_ctc_timesync_recog_recomb_auto_scale,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)

from i6_experiments.users.zeyer.nn_rf.encoder import ff
from i6_experiments.users.zeyer.nn_rf.encoder import chunked_conformer_v1

__setup_root_prefix__ = "exp2025_10_21_chunked_ctc"


def py():
    train("base", {})

    train(
        "ff6",
        {"train.aux_loss_layers": [6]},
        {"model.enc_build_dict": rf.build_dict(ff.FeedForwardEncoder, num_layers=6, out_dim=1024)},
    )
    train(
        "ff12",
        {"train.aux_loss_layers": [6, 12]},
        {"model.enc_build_dict": rf.build_dict(ff.FeedForwardEncoder, num_layers=12, out_dim=1024)},
    )
    train(
        "ff12-dec3",
        {"train.aux_loss_layers": [6, 12], "train.dec_aux_loss_layers": None, "model.dec_build_dict.num_layers": 3},
        {"model.enc_build_dict": rf.build_dict(ff.FeedForwardEncoder, num_layers=12, out_dim=1024)},
    )
    train("ff12-bug", {}, {"model.enc_build_dict": rf.build_dict(ff.FeedForwardEncoder, num_layers=12, out_dim=1024)})

    downsampling = 6

    for left_n, center_size, right_size, bs in [
        # fixing total chunk size to 35, fixing left ctx to 40, varying center/right
        (8, 5, 30, 20_000),
        (4, 10, 25, 25_000),
        (2, 20, 15, 50_000),
        # fixing total chunk size to 40, fixing left ctx to 40, varying center/right
        (4, 10, 30, 25_000),
        (2, 20, 20, 50_000),
        (1, 40, 0, 75_000),
        # fixing total chunk size to 40, fixing right ctx to 0, varying left
        (1, 40, 0, 75_000),
        (2, 40, 0, 75_000),
        (3, 40, 0, 50_000),
        # fixing total chunk size to 35, fixing right ctx to 15, varying left
        (0, 20, 15, 50_000),
        (1, 20, 15, 50_000),
        (2, 20, 15, 50_000),
        (3, 20, 15, 50_000),
        (4, 20, 15, 50_000),
        (8, 20, 15, 50_000),
    ]:
        train(
            f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}",
            {
                "model.enc_build_dict": rf.build_dict(
                    ChunkedConformerEncoder,
                    encoder_layer=rf.build_dict(ChunkedConformerEncoderLayer),
                    chunk_stride=center_size * downsampling,
                    chunk_history=left_n,
                    input_chunk_size_dim=(center_size + right_size) * downsampling,
                    end_chunk_size_dim=center_size,
                ),
                "train.batch_size": bs * configs._batch_size_factor,
            },
        )


_base_config = {
    # ("large", 100),  # 100kh in total, 4 full epochs
    # ("large", 150),  # 150kh in total, 6 full epochs
    # ("large", 200),  # 200kh in total, 8 full epochs
    # ("large", 250),  # 250kh in total, 10 full epochs
    # ("large", 500),  # 500kh in total, 20 full epochs
    "subset": "large",
    "total_k_hours": 100,
    "vocab": "spm10k",
    "model": {
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
    "train_update_func_from_n_ep": lambda n_ep: {"train": configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=0.5)},
    "train": dict_update_deep(
        configs.config_96gb_bf16_accgrad1,
        {
            "batch_size": 100_000 * configs._batch_size_factor,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            # "__train_audio_preprocess": speed_pert_librosa_config,
            # "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            # Note on max seq len stats: Before, when we used max_seq_length_default_target=75 with bpe10k,
            # out of 281241 seqs in train, we removed only 71 seqs.
            # With max seq len 19.5 secs on the audio, we also remove exactly 71 seqs.
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
    ),
    "train_post": dict_update_deep(
        configs.post_config, {"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}}
    ),
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    "dataset_train_opts": {"train_epoch_split": 1, "train_epoch_wise_filter": None},
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
}


def train(name: str, config: Dict[str, Any], config_overrides: Optional[Dict[str, Any]] = None):
    prefix = get_setup_prefix_for_module(__name__)

    config = dict_update_deep(_base_config.copy(), config.copy())
    config = dict_update_deep(config, config_overrides, dict_value_merge=False)

    train_epoch_split_per_subset = {"clean": 13, "small": 1, "medium": 2, "large": 25}
    hours_per_subset = {"clean": 13_000, "small": 250, "medium": 2_500, "large": 25_000}
    subset = config.pop("subset")
    total_k_hours = config.pop("total_k_hours")
    train_epoch_split = train_epoch_split_per_subset[subset]
    num_full_ep = total_k_hours * 1_000 / hours_per_subset[subset]
    n_ep = round(num_full_ep * train_epoch_split)

    train_update_func_from_n_ep = config.pop("train_update_func_from_n_ep")
    if train_update_func_from_n_ep:
        config = dict_update_deep(config, train_update_func_from_n_ep(n_ep))

    model_config = config.pop("model")
    train_config: Dict[str, Any] = config.pop("train")
    post_config = config.pop("train_post")

    vocab = config.pop("vocab", "spm10k")
    task = get_loquacious_task_raw_v2(vocab=vocab, subset_name=subset, train_epoch_split=train_epoch_split)

    train_vocab_opts = config.pop("train_vocab_opts")
    dataset_train_opts = config.pop("dataset_train_opts")
    env_updates = config.pop("env_updates")

    assert not config

    exp = aed_train_exp(
        name,
        train_config,
        prefix=prefix + "/aed/",
        task=task,
        model_config=model_config,
        post_config_updates=post_config,
        vocab=vocab,
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates=env_updates,
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=max(train_config["aux_loss_layers"]),
    )


class ChunkedConformerEncoder(chunked_conformer_v1.ChunkedConformerEncoder):
    """alias"""


class ChunkedConformerEncoderLayer(chunked_conformer_v1.ChunkedConformerEncoderLayer):
    """alias"""
