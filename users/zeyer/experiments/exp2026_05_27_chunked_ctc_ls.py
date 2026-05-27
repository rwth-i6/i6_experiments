"""
Chunked CTC / Conformer on LibriSpeech.

LS counterpart of :mod:`exp2025_10_21_chunked_ctc`,
trained on the same LS data + recipe as the offline AED baseline
``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
in :mod:`exp2025_08_05_aed_large`,
only the encoder is swapped for :class:`ChunkedConformerEncoderV2`.

Sweeps a handful of ``(left_ctx, center, right_lookahead)`` configs
(all numbers are encoder frames after the ``ConformerConvSubsample`` /6 downsampling,
so one frame ~= 60 ms).
"""

from __future__ import annotations

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
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    model_recog as ctc_model_recog,
)
from i6_experiments.users.zeyer.speed_pert.librosa_config import speed_pert_librosa_config
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
)

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import (
    ConformerEncoder,
    ConformerEncoderLayer,
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)


__setup_root_prefix__ = "exp2026_05_27_chunked_ctc_ls"


def py():
    # Register the offline LS AED baseline under this setup's prefix
    # with a byte-identical ``aed_train_exp(...)`` call,
    # so its ``ReturnnTrainingJob`` hash matches the existing trained model
    # at ``~/setups/2025-08-aed-large/work/.../ReturnnTrainingJob.IVB5xAuHZZA3``.
    # The bulk-import script (``import_work_directory.py``) then symlinks
    # that finished training in here -- no re-training.
    _train_ls_offline_baseline()

    # ``(left_ctx, center, right_lookahead)`` in encoder frames (~60 ms each).
    # Causal variants (R=0) explore the encoder-only-latency budget;
    # (80, 5, 4) is the reference-streaming config matching the Loquacious sweep.
    # Batch size shrinks for the larger-center variants
    # because their effective per-step compute grows;
    # the values mirror the limited-history block of :mod:`exp2025_10_21_chunked_ctc`.
    for left, center, right, bs in [
        (0, 5, 0, 75_000),
        (0, 10, 0, 75_000),
        (0, 20, 0, 75_000),
        (0, 40, 0, 75_000),
        (80, 5, 4, 50_000),
    ]:
        _train_ls(
            f"chunked-L{left}-C{center}-R{right}-v2.3",
            chunk_size=center,
            chunk_history_size=left,
            chunk_lookahead_size=right,
            batch_size=bs,
        )


def _train_ls_offline_baseline():
    """
    Re-register the offline LS AED baseline
    ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
    (originally trained in :mod:`exp2025_08_05_aed_large`)
    under this setup's prefix.

    The :func:`aed_train_exp` call is byte-identical to the original
    (encoder = :class:`ConformerEncoder`, no chunking;
    no ``recog_def`` / ``search_config`` / ``max_seqs`` -- those are not passed in the original).
    The resulting ``ReturnnTrainingJob`` hash therefore matches the existing
    ``IVB5xAuHZZA3`` in ``~/setups/2025-08-aed-large/``,
    so the bulk-import via ``import_work_directory.py`` symlinks the trained model
    instead of triggering a new training.

    The ``aed_ctc_timesync_recog_recomb_auto_scale`` call mirrors the original;
    it depends on ``get_librispeech_task_raw_v2(vocab="spm10k")`` *without* the
    ``train_vocab_opts`` / ``train_epoch_split`` kwargs (the training task uses those,
    but the recog task is built default-style, matching the original).
    """
    prefix = get_setup_prefix_for_module(__name__)
    name = "EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k"
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
                        ConformerPositionwiseFeedForward,
                        activation=rf.build_dict(rf.relu_square),
                        with_bias=False,
                    ),
                    num_heads=8,
                ),
            ),
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
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
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
        dataset_train_opts={"train_epoch_split": 1, "train_epoch_wise_filter": None},
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    )
    task_spm10k = get_librispeech_task_raw_v2(vocab="spm10k")
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task_spm10k,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    return exp


def _train_ls(name: str, *, chunk_size: int, chunk_history_size: int, chunk_lookahead_size: int, batch_size: int):
    """
    Train one chunked-CTC LS experiment.

    All hyperparameters mirror the offline LS baseline
    ``EncL16-DecL6-D1024-DecPosEncAbs-featBN-aux4_10_16-auxDec3-spm10k-bpeSample001-baseLr0.5-b100k``
    in :mod:`exp2025_08_05_aed_large`.
    The only changes are:
    encoder = :class:`ChunkedConformerEncoderV2` (v3) with the given chunking,
    and a smaller ``batch_size`` to absorb the chunked memory overhead.
    """
    prefix = get_setup_prefix_for_module(__name__)
    # ``get_librispeech_task_raw_v2`` is cached, so the same call inside ``aed_train_exp``
    # (which we don't pass ``task=`` to, letting it default to LS via the same factory)
    # returns this exact Task object -- they stay consistent without explicit threading.
    train_vocab_opts = {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}}
    dataset_train_opts = {"train_epoch_split": 1, "train_epoch_wise_filter": None}
    task = get_librispeech_task_raw_v2(vocab="spm10k", train_vocab_opts=train_vocab_opts, **dataset_train_opts)
    exp = aed_train_exp(
        name,
        config_96gb_bf16_accgrad1,
        prefix=prefix + "/aed/",
        model_config={
            "behavior_version": 24,
            "__serialization_version": 2,
            "enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
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
                    ChunkedConformerEncoderLayerV2,
                    ff=rf.build_dict(
                        ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                    ),
                    num_heads=8,
                ),
                chunk_size=chunk_size,
                chunk_history_size=chunk_history_size,
                chunk_lookahead_size=chunk_lookahead_size,
                version=3,
            ),
            "dec_build_dict": rf.build_dict(
                TransformerDecoder,
                num_layers=6,
                model_dim=1024,
                norm=rf.build_dict(rf.RMSNorm),
                ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
            ),
            "feature_batch_norm": True,
        },
        config_updates={
            **_get_cfg_lrlin_oclr_by_bs_nep_v4(100, base_lr=0.5),
            "batch_size": batch_size * _batch_size_factor,
            "max_seqs": 200,
            "optimizer.weight_decay": 1e-2,
            "accum_grad_multiple_step": 1,
            "__train_audio_preprocess": speed_pert_librosa_config,
            "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
            "aux_loss_layers": [4, 10, 16],
            "dec_aux_loss_layers": [3],
            "max_seq_length_default_target": None,
            "max_seq_length_default_input": 19.5 * _raw_sample_rate,
        },
        post_config_updates={"log_grad_norm": True, "__multi_proc_dataset_opts": {"num_workers": 25}},
        vocab="spm10k",
        train_vocab_opts=train_vocab_opts,
        dataset_train_opts=dataset_train_opts,
        env_updates={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
        recog_def=ctc_model_recog,
        search_config={"aux_loss_layers": [16]},
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=16,
    )
    return exp
