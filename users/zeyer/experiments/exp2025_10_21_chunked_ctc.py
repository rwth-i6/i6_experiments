"""
Chunked CTC / Conformer.

For earlier code and some reference, see:
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_conformer.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_ctc.py
i6_experiments/users/zeyer/experiments/exp2023_04_25_rf/chunked_aed_import.py

Chunked Attention-based Encoder-Decoder Model for Streaming Speech Recognition, https://arxiv.org/abs/2309.08436
"""

from __future__ import annotations

from typing import Optional, Any, Dict, List, Tuple, Union
from functools import cache

from sisyphus import tk

from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint, ModelDefWithCfg
from i6_experiments.users.zeyer.recog import recog_model
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
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc_recog_ext import (
    ctc_recog_recomb_labelwise_prior_auto_scale,
)
from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.ctc import (
    model_recog as ctc_model_recog,
)

from i6_experiments.users.zeyer.datasets.loquacious import (
    get_loquacious_task_raw_v2,
    get_loquacious_train_subset_dataset_v2,
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
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
    ChunkedRelPosSelfAttentionV2,
    ChunkedRotaryPosSelfAttentionV2,
)

__setup_root_prefix__ = "exp2025_10_21_chunked_ctc"


def _add_row_index_id_column(ds):
    """Add a unique ``id`` column to a HF Dataset / DatasetDict using the row index.

    Used for HF datasets that lack an ``id`` field (e.g. ``distil-whisper/
    tedlium-long-form``). The RETURNN ``HuggingFaceDataset`` wrapper needs a
    unique-per-row seq tag and ``id`` is the default ``seq_tag_column``.
    Top-level so the function reference is stable for Sisyphus hashing.
    """
    from datasets import Dataset, DatasetDict

    if isinstance(ds, DatasetDict):
        return DatasetDict(
            {
                split: ds_split.add_column("id", [f"{split}_{i:06d}" for i in range(len(ds_split))])
                for split, ds_split in ds.items()
            }
        )
    if isinstance(ds, Dataset):
        return ds.add_column("id", [f"{i:06d}" for i in range(len(ds))])
    raise TypeError(f"unexpected ds type {type(ds)}")


def py():
    _exp_base, _task_base, _aux_base = train("base", {})

    # Verify the offline-trained base model runs correctly when loaded into
    # ChunkedConformerEncoderV2 (v=3). Offline first (chunk_size=None) -- WER
    # should match the base's existing offline recog. Then sweep chunk settings
    # to see how much the offline-trained model degrades under chunked attention.
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
        ChunkedConformerEncoderLayerV2 as _ChunkLayer,
        ChunkedConformerEncoderV2 as _ChunkEnc,
    )

    _base_v2_enc = rf.build_dict(
        _ChunkEnc,
        input_layer=rf.build_dict(
            ConformerConvSubsample,
            out_dims=[32, 64, 64],
            filter_sizes=[(3, 3), (3, 3), (3, 3)],
            pool_sizes=[(1, 2)],
            strides=[(1, 1), (3, 1), (2, 1)],
        ),
        num_layers=16,
        out_dim=1024,
        encoder_layer=rf.build_dict(
            _ChunkLayer,
            ff=rf.build_dict(
                ConformerPositionwiseFeedForward,
                activation=rf.build_dict(rf.relu_square),
                with_bias=False,
            ),
            num_heads=8,
        ),
        chunk_size=None,
        version=3,
    )
    for cs, lh in [(None, 0), (5, 4), (10, 8), (20, 15)]:
        enc = dict(_base_v2_enc)
        enc["chunk_size"] = cs
        enc["chunk_lookahead_size"] = lh
        enc["chunk_history_size"] = 80 if cs is not None else 0
        recog_model_with_config_overwrite(
            model=_exp_base.get_last_fixed_epoch(),
            task=_task_base,
            recog_def=ctc_model_recog,
            config_overwrites={"enc_build_dict": enc},
            extra_config={"aux_loss_layers": [_aux_base]},
            name="base-via-v2.3",
            tag=f"L80-C{cs}-R{lh}" if cs is not None else "offline",
        )

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
        # fixing left chunk size to 80, trying smaller chunk sizes
        (4, 20, 15, 50_000),
        (8, 10, 8, 50_000),
        (16, 5, 4, 50_000),
        (40, 2, 3, 25_000),
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
        # fixing right chunk size to 15, fixing left N to 1, varying center
        (1, 20, 15, 50_000),
        (1, 40, 15, 50_000),
        (1, 100, 15, 50_000),
        # (1, 1000, 15, (10_000, 10)),  # ~8h/subepoch... will take a month to train  - broken?
        # (1, 5000, 15, (50_000, 1)),  # ~104h/subepoch..., not reasonable...
    ]:
        if isinstance(bs, tuple):
            bs, max_seqs = bs
        else:
            max_seqs = 200
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
                "train.max_seqs": max_seqs,
            },
        )

    # Vocabs:
    for vocab in ["spm1k", "spm5k", "spm10k"]:
        # CTC-only:
        # spm1k: 10.66
        # spm5k: 9.52
        # spm10k: 9.56
        left_n, center_size, right_size, bs = (16, 5, 4, 50_000)
        max_seqs = 200
        train(
            f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-{vocab}",
            {
                "vocab": vocab,
                "model.enc_build_dict": rf.build_dict(
                    ChunkedConformerEncoder,
                    encoder_layer=rf.build_dict(ChunkedConformerEncoderLayer),
                    chunk_stride=center_size * downsampling,
                    chunk_history=left_n,
                    input_chunk_size_dim=(center_size + right_size) * downsampling,
                    end_chunk_size_dim=center_size,
                ),
                "train.batch_size": bs * configs._batch_size_factor,
                "train.max_seqs": max_seqs,
            },
        )

    # V2.1 (bugged): ChunkedConformerEncoderV2, more flexible, more optimized chunking
    # epoch train time (recipe/i6_experiments/users/zeyer/returnn/tools/check_train_times.py) mean:
    #   3738.79 (v1: 7728.30)
    # CTC-only: 11.74 (v1: 9.56)
    left_n, center_size, right_size, bs = (16, 5, 4, 50_000)

    # V2.2 (bugged): using ChunkedConformerEncoderV2, setting version=2:
    #   reduce chunk sizes, history, if the input is not long enough.
    # epoch train time mean: 3726.87
    # CTC-only: 11.67 (v1: 9.56)

    # Note: version=1 and version=2 used a wrong chunking implementation. Fixed for version>=3.

    # V2.3: using ChunkedConformerEncoderV2, setting version=3.
    # First exp, try to reproduce the orig.
    # train_time_hours: 168.9 (v1: 215.6)
    # CTC-only: 9.45 (v1: 9.56)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-compat",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                version=3,
                adapt_chunk_history_for_short_seqs=False,  # compat with V1
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
        },
    )

    # V2.3: using ChunkedConformerEncoderV2, setting version=3.
    #   reduce chunk sizes, history, if the input is not long enough (adapt_chunk_history_for_short_seqs=True default)
    # train_time_hours: 168.8 (v1: 215.6; adapt_chunk_history_...=False: 168.9; offline: 66.2)
    # CTC-only: 9.46 (v1: 9.56; adapt_chunk_history_...=False: 9.45; offline: 7.32)
    # CTC+LM: 7.22 (offline: 6.12)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
        },
    )

    # Try grad checkpointing (mem_chunks_grad_checkpointing=True).
    # (In terms of WER, should really be the same.
    # if in terms of speed this is better, and same for memory consumption, we could maybe just always enable it.)
    # train_time_hours: 201.1 (v1: 215.6; ..._checkpointing=False: 168.8) (but requires less memory)
    # CTC-only: 9.52 (..._checkpointing=False: 9.46)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-gdckpt",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                version=3,
                mem_chunks_grad_checkpointing=True,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
        },
    )

    # Rope instead of relpos selfatt (ChunkedRotaryPosSelfAttentionV2).
    # (We don't expect really improvements in terms of WER. Hopefully mostly the same.
    #  However, we can hope to have better speed here, maybe also less memory consumption. Check that.)
    # train_time_hours: 237.1 (vs 168.8) (slow apply_rope. better in run2 below with newer RETURNN)
    # CTC-only: 9.31 (vs 9.46)
    name = f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-rope"
    exp, task, aux_ctc_layer = train(
        name,
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # Recog-time chunk-size / lookahead sweep, as the negative control for the dyn sweep:
    # this model never saw chunk_size != center_size or lookahead != right_size during training,
    # so it is expected to degrade off-canonical.
    for cs, lh in [
        (center_size, right_size),
        (center_size * 2, right_size),
        (center_size * 4, right_size),
        (center_size * 8, right_size),
        (center_size, right_size // 2),
        (None, 0),
    ]:
        recog_model_with_config_overwrite(
            model=exp.get_last_fixed_epoch(),
            task=task,
            recog_def=ctc_model_recog,
            config_overwrites={
                "enc_build_dict.chunk_size": cs,
                "enc_build_dict.chunk_lookahead_size": lh,
            },
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            name=name,
            tag=f"L{left_n * center_size}-C{cs}-R{lh}" if cs is not None else "offline",
        )

    # Non-dyn (fixed chunk) + rope + ctembed: the fixed-chunk control for the dyn-pool ablation,
    # holding rope and ctembed constant so only dyn-vs-fixed differs.
    # In the no-rope/no-ctembed comparison, fixed beat dyn (9.46 vs 9.66); this checks if that still holds.
    name = f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-rope-ctembed"
    exp, task, aux_ctc_layer = train(
        name,
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # Recog-time chunk-size / lookahead sweep (negative control, as for -v2.3-rope above).
    for cs, lh in [
        (center_size, right_size),
        (center_size * 2, right_size),
        (center_size * 4, right_size),
        (center_size * 8, right_size),
        (center_size, right_size // 2),
        (None, 0),
    ]:
        recog_model_with_config_overwrite(
            model=exp.get_last_fixed_epoch(),
            task=task,
            recog_def=ctc_model_recog,
            config_overwrites={
                "enc_build_dict.chunk_size": cs,
                "enc_build_dict.chunk_lookahead_size": lh,
            },
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            name=name,
            tag=f"L{left_n * center_size}-C{cs}-R{lh}" if cs is not None else "offline",
        )

    # Dynamic chunking.
    # Haotian did:
    #   (I think sizes are on 10ms level.)
    #   The schedule: have chunk_size_pool(128, 256, 512, 1024, unlimited frames in my most training),
    #   chunk size is uniformly sampled from the pool and be set for whole batch.
    #   The number of chunks for state carry over is also randomly selected
    #     from 0 to max_chunk_size_in_pool // selected_chunk_size.
    #   (No right context in that setup.)
    # We do here some slightly different schedule, but similar.
    # But also varying the lookahead.
    # train_time_hours: 103.9 (vs 168.8)
    # CTC-only: 9.66 (vs 9.46)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
        },
    )

    # Dynamic chunking + rope.
    # train_time_hours: 128.4 (without rope: 103.9; not dynamic, without rope: 168.8) (slow apply_rope, see run2)
    # CTC-only: 9.55 (without rope: 9.66; not dynamic, without rope: 9.46)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Chunk-type embed (ctembed) (use_chunk_type_embedding=True)
    # Using with dynamic chunking + rope as base.
    # train_time_hours: 128.3 (vs 128.4)
    # CTC-only: 9.41 (vs 9.55; no dyn, no ctembed, just rope: 9.31)
    # Recog-time chunk-size sweep (CTC-only, last ep):
    #   C5-LA2   10.14
    #   C5-LA4  (canonical)  9.41
    #   C10-LA4  9.00
    #   C20-LA4  8.61
    #   C40-LA4  8.26
    #   offline  7.80
    name = f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed"
    exp, task, aux_ctc_layer = train(
        name,
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # CTC-only recog-time chunk-size / lookahead sweep on the last fixed epoch.
    for cs, lh in [
        (center_size, right_size),
        (center_size * 2, right_size),
        (center_size * 4, right_size),
        (center_size * 8, right_size),
        (center_size, right_size // 2),
        (None, 0),
    ]:
        recog_model_with_config_overwrite(
            model=exp.get_last_fixed_epoch(),
            task=task,
            recog_def=ctc_model_recog,
            config_overwrites={
                "enc_build_dict.chunk_size": cs,
                "enc_build_dict.chunk_lookahead_size": lh,
            },
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            name=name,
            tag=f"L{left_n * center_size}-C{cs}-R{lh}" if cs is not None else "offline",
        )

    # Overlap at recog only: this model trained WITHOUT overlap; does overlap-averaging help at decode?
    # Exploratory: the weights never saw overlap, and with ctembed + C=5/overlaps=2 the per-chunk output
    # region (chunk_size_dim=4) is smaller than the ctembed center boundary (chunk_size=5), an off-by-one.
    # Result: overlap at recog HURTS this no-overlap-trained model.
    #   C5-R4-ov2 10.65 (vs C5-R4 9.41);  C5-R2-ov2 18.10 (vs C5-R2 10.14).
    # (The weights never saw overlap; the C=5/overlaps=2 ctembed off-by-one above makes R2 far worse.)
    for cs, lh in [(center_size, right_size), (center_size, right_size // 2)]:
        recog_model_with_config_overwrite(
            model=exp.get_last_fixed_epoch(),
            task=task,
            recog_def=ctc_model_recog,
            config_overwrites={
                "enc_build_dict.chunk_size": cs,
                "enc_build_dict.chunk_lookahead_size": lh,
                "enc_build_dict.chunk_num_overlaps": 2,
            },
            extra_config={"aux_loss_layers": [aux_ctc_layer]},
            name=name,
            tag=f"L{left_n * center_size}-C{cs}-R{lh}-ov2",
        )

    # Streaming-vs-offline log-prob consistency test (minimal: 3 train seqs).
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_streaming_consistency_test import (
        make_streaming_consistency_test_job,
    )

    _stream_consistency_job = make_streaming_consistency_test_job(
        model=exp.get_last_fixed_epoch(),
        dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k", num_seqs=3),
        aux_loss_layers=[aux_ctc_layer],
        segment_seconds=10.0,
        version=2,  # bumped after fixing per-segment truncation
    )
    tk.register_output(
        f"{get_setup_prefix_for_module(__name__)}/aed/{name}/streaming-consistency.json",
        _stream_consistency_job.out_files["diff_stats.json"],
    )

    # Same comparison, but using the KV-cache streaming (v2) with overlap-and-trim.
    _stream_consistency_job_v2 = make_streaming_consistency_test_job(
        model=exp.get_last_fixed_epoch(),
        dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k", num_seqs=3),
        aux_loss_layers=[aux_ctc_layer],
        segment_seconds=10.0,
        version=1,
        use_kv_cache_v2=True,
    )
    tk.register_output(
        f"{get_setup_prefix_for_module(__name__)}/aed/{name}/streaming-consistency-kvcache.json",
        _stream_consistency_job_v2.out_files["diff_stats.json"],
    )

    # CTC recog using the KV-cache streaming encoder on the standard eval sets.
    # Should give WER matching the offline recog if the encoder log-probs match.
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_streaming_v2 import (
        model_recog_ctc_streaming_v2,
    )

    recog_model_with_config_overwrite(
        model=exp.get_last_fixed_epoch(),
        task=task,
        recog_def=model_recog_ctc_streaming_v2,
        extra_config={
            "aux_loss_layers": [aux_ctc_layer],
            "max_seqs": 1,  # streaming impl requires batch=1
            "streaming_segment_seconds": 10.0,
        },
        name=name,
        tag="streaming-kvcache-v2-seg10",
    )

    # Tedlium: segmented (HF Open ASR Leaderboard) + long-form (distil-whisper/
    # tedlium-long-form, full ~17-min talks). Both use streaming-kvcache recog.
    import dataclasses as _dataclasses
    import functools as _functools
    from i6_core.datasets.huggingface import TransformAndMapHuggingFaceDatasetJob
    from i6_experiments.users.zeyer.datasets.loquacious import get_vocab_by_str as _get_vocab
    from i6_experiments.users.zeyer.datasets.hf_open_asr_leaderboard import (
        HuggingFaceDataset as _HFDataset,
        hacked_sclite_score_recog_out as _hf_score,
        get_asr_leaderboard_hf_data_dir as _get_hf_leaderboard_dir,
    )
    from i6_experiments.users.zeyer.recog import recog_model as _recog_model

    @_functools.cache
    def _get_tedlium_longform_hf_dir():
        _job = TransformAndMapHuggingFaceDatasetJob(
            "distil-whisper/tedlium-long-form",
            transform=_add_row_index_id_column,
        )
        _job.rqmt.update({"cpu": 4, "time": 1, "mem": 16})
        _job.add_alias("datasets/tedlium-long-form")
        tk.register_output("datasets/tedlium-long-form", _job.out_dir)
        return _job.out_dir

    _vocab_obj = _get_vocab("spm10k")
    _tedlium_seg = _HFDataset(
        hf_data_dir=_get_hf_leaderboard_dir("tedlium"),
        name="tedlium",
        split="test",
        vocab=_vocab_obj,
        sorting_seq_len_column="audio_length_s",
    )
    _tedlium_long = _HFDataset(
        hf_data_dir=_get_tedlium_longform_hf_dir(),
        name="tedlium-long-form",
        split="test",
        vocab=_vocab_obj,
        seq_ordering="default",  # long-form has no duration column; skip sort
        sorting_seq_len_column="",
    )

    def _hf_score_long_rqmt(dataset, recog_output):
        # Wrap the standard sclite scorer so that for long-form recordings,
        # the downstream ScliteJob runs on a partition
        # with enough wallclock (default mini_task short partition = 30 min,
        # which gets killed mid-alignment).
        res = _hf_score(dataset, recog_output)
        score_job = res.main_measure_value.creator
        score_job.rqmt = {"cpu": 1, "mem": 4.0, "time": 4.0}
        score_job.set_rqmt("run", score_job.rqmt)
        return res

    _task_tedlium = _dataclasses.replace(task, score_recog_output_func=_hf_score_long_rqmt)

    _tedlium_res = _recog_model(
        task=_task_tedlium,
        model=exp.get_last_fixed_epoch(),
        recog_def=model_recog_ctc_streaming_v2,
        config={
            "aux_loss_layers": [aux_ctc_layer],
            "max_seqs": 1,
            "streaming_segment_seconds": 10.0,
        },
        eval_sets={"tedlium-seg.test": _tedlium_seg, "tedlium-long.test": _tedlium_long},
    )
    tk.register_output(
        f"{get_setup_prefix_for_module(__name__)}/aed/{name}/streaming-kvcache-v2-seg10-tedlium",
        _tedlium_res.output,
    )

    # Newer RETURNN. This has a faster apply_rope.
    # Still not really faster than relpos self-att.
    # This is because when we explicitly do the self-att computation, and using the relpos trick,
    # there RoPE is actually not really cheaper.
    # CTC-only: 9.52 (original dyn-rope-ctembed run: 9.41; diff is run-to-run + RETURNN-version variance).
    # train_time_hours: faster with rope: 107.3 (with rope (old): 128.4, without rope: 103.9; not dynamic, without rope: 168.8)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-run2",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "train._run_version": 2,  # trigger hash change to run again with new RETURNN
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Test relpos-self-att again.
    # CTC-only: 9.65 (rope variant dyn-rope-ctembed: 9.41).
    # train(
    #     f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-ctembed",
    #     {
    #         "model.enc_build_dict": rf.build_dict(
    #             ChunkedConformerEncoderV2,
    #             encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
    #             chunk_size=center_size,
    #             chunk_history_size=left_n * center_size,
    #             chunk_lookahead_size=right_size,
    #             chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
    #             chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
    #             chunk_lookahead_size_train_pool=[right_size, right_size // 2],
    #             use_chunk_type_embedding=True,
    #             version=3,
    #         ),
    #         "train.batch_size": bs * configs._batch_size_factor,
    #         "train.max_seqs": max_seqs,
    #     },
    # )

    # Relpos self-att with learnable pos emb.
    # CTC-only: 9.99 (plain relpos dyn-ctembed: 9.65; rope dyn-rope-ctembed: 9.41).
    # train(
    #     f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-relposL-ctembed",
    #     {
    #         "model.enc_build_dict": rf.build_dict(
    #             ChunkedConformerEncoderV2,
    #             encoder_layer=rf.build_dict(
    #                 ChunkedConformerEncoderLayerV2,
    #                 self_att=rf.build_dict(ChunkedRelPosSelfAttentionV2, learnable_pos_emb=True),
    #             ),
    #             chunk_size=center_size,
    #             chunk_history_size=left_n * center_size,
    #             chunk_lookahead_size=right_size,
    #             chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
    #             chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
    #             chunk_lookahead_size_train_pool=[right_size, right_size // 2],
    #             use_chunk_type_embedding=True,
    #             version=3,
    #         ),
    #         "train.batch_size": bs * configs._batch_size_factor,
    #         "train.max_seqs": max_seqs,
    #     },
    # )

    # Dyn-v2 (dynV2): Try to make it faster:
    # Offline more often. Also sometimes without any overhead, sometimes without any history.
    # train_time_hours: 100.1 (vs 128.3)
    # CTC-only: 11.0 (vs 9.41) (but no overfitting, just bad) (see dynV3)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dynV2-rope-ctembed",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8] + [None] * 4,
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2, 0],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2, 0],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # Oversampling small center_size: see -v2.3-dynCx3-rope-ctembed below.

    # DynV3: Less often offline.
    # CTC-only: 10.49 (standard dyn dyn-rope-ctembed: 9.41; dynV2: 11.0).
    # Note: dynV3 without the 0 in history/lookahead train pool == the dyn-rope-ctembed baseline (9.41),
    # which beats dynV3 (10.49) -> the 0 hurt.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dynV3-rope-ctembed",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2, 0],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2, 0],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # DynV4: like DynV3 but ALWAYS keep history (no 0 in the history train pool, as the baseline "dyn"),
    # while keeping 0 in the LOOKAHEAD train pool.
    # Motivation: history clearly helps WER (L0 24% vs L80 9.5%),
    # and the DynV3 ablation showed the 0-history option hurt (10.49 vs 9.41).
    # But we still want low latency, which the lookahead drives.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dynV4-rope-ctembed",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2, 0],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Overlapping chunks (chunk_num_overlaps=2)
    # train_time_hours: 391.4 (vs 168.8) !! (overlaps require twice as much compute, but also smaller batch size)
    #   (maybe could get away with less training?)
    # CTC-only: 9.26 (vs 9.46)
    # Is overlap's gain just ~2x compute, and is lookahead needed?
    # See follow-ups -v2.3-2xtrain and -C5-R0-v2.3-overlap below.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-overlap",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_num_overlaps=2,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
        },
    )

    # As v2.3-overlap, plus MSE loss between overlapping encoder views.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-overlap-mse",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_num_overlaps=2,
                overlap_mse_loss_scale=1.0,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
        },
    )

    # Dyn + rope + ctembed + overlap.
    # CTC-only: 10.20 (top CTC head specifically degraded vs non-dyn overlap; see projects notes)
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-overlap",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                chunk_num_overlaps=2,
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # As v2.3-dyn-rope-ctembed-overlap, plus MSE loss between overlapping encoder views.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-overlap-mse",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                chunk_num_overlaps=2,
                use_chunk_type_embedding=True,
                overlap_mse_loss_scale=1.0,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Dynamic overlaps. (Also newer RETURNN, faster rope.)
    # CTC-only: 9.83 (no overlap dyn-rope-ctembed: 9.41; the fixed-overlap -overlap variant above is worse still).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-overlapD",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                chunk_num_overlaps=2,
                chunk_num_overlaps_train_pool=[2, 1],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Isolation: dyn + rope + overlapD WITHOUT ctembed.
    # Decisive test of whether the ctembed/overlap interaction regresses overlap+ctembed,
    # since overlap alone (relpos, no ctembed) helped (9.46 -> 9.26)
    # but overlap+ctembed (9.83) is worse than ctembed-no-overlap (9.41).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-overlapD",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                chunk_num_overlaps=2,
                chunk_num_overlaps_train_pool=[2, 1],
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # overlapD with ctembedfix.
    # As v2.3-dyn-rope-ctembed-overlapD,
    # but with the ctembed center/lookahead boundary aligned to the per-chunk output region (chunk_size_dim)
    # instead of raw chunk_size.
    # Fixes the off-by-one for C=5/overlaps=2 (chunk_size_dim=4 < chunk_size=5):
    # the last center-marked frame's output was dropped in _unchunk.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-overlapD-ctembedfix",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                chunk_num_overlaps=2,
                chunk_num_overlaps_train_pool=[2, 1],
                use_chunk_type_embedding=True,
                chunk_type_embedding_at_output_boundary=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # Limited history experiments (vary history size L; C5-R4).
    # CTC-only: L0 24.03 (no history); L40 9.41; L80 = the -v2.3 baseline above (9.46).
    # Latency behavior (align-stats latency metric, measured vs the TIMIT word END):
    # L0 emits each word ~300ms BEFORE its reference end (strong negative offset), L80 ~0.
    # This is not future-peeking and not a metric bug.
    # With no history the L0 encoder must lean heavily on the lookahead (right) frames,
    # but the lookahead never emits (only the center frames do),
    # so training shifts those emissions forward into the visible center region.
    # An emergent behavior forced by the tight no-history constraint,
    # and it co-occurs with the bad 24% WER (premature, low-confidence spikes).
    # L80 doesn't need this:
    # the history makes each center chunk recognizable on its own,
    # so it emits where the labels actually occur (offset ~0) at 9.46% WER.
    for ls_, rs_ in [(0, 4), (40, 4), (80, 4)]:
        train(
            f"chunked-L{ls_}-C{center_size}-R{rs_}-v2.3",
            {
                "model.enc_build_dict": rf.build_dict(
                    ChunkedConformerEncoderV2,
                    encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                    chunk_size=center_size,
                    chunk_history_size=ls_,
                    chunk_lookahead_size=rs_,
                    version=3,
                ),
                "train.batch_size": bs * configs._batch_size_factor,
                "train.max_seqs": max_seqs,
            },
        )

    # Long-context experiments: interleave standard ChunkedConformerEncoderLayerV2
    # with a recurrent layer (Mamba-2 SSD or DeltaNet delta-rule).
    # 16 total layers, alternating [std, rec, std, rec, ...], so 8 standard + 8 recurrent.
    # Same outer chunked structure (chunk_size, history, lookahead, chunk-type-embedding)
    # as the v2.3-dyn-rope-ctembed baseline, so results are directly comparable.
    # No overlap-dynamic here -- overlapD was dropped from these hybrids (it regressed).
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2_mamba2 import (
        Mamba2ChunkedLayerV2,
        BidirMamba2ChunkedLayer,
    )
    from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2_deltanet import (
        DeltaNetChunkedLayerV2,
        BidirDeltaNetChunkedLayer,
    )

    _std_layer_spec = rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2)
    _mamba2_layer_spec = rf.build_dict(
        Mamba2ChunkedLayerV2,
        d_state=128,
        d_conv=4,
        expand=2,
        head_dim=64,
        n_groups=1,
        block_len=64,
        version=2,
    )
    _deltanet_layer_spec = rf.build_dict(
        DeltaNetChunkedLayerV2,
        d_state=128,
        head_dim=64,
        block_len=32,
        version=3,
    )
    # Bidir variants: forward = causal scan with cross-chunk continuity,
    # backward = per-chunk reverse over [center+lookahead]. Outputs averaged at center.
    _mamba2_bidir_layer_spec = rf.build_dict(
        BidirMamba2ChunkedLayer,
        d_state=128,
        d_conv=4,
        expand=2,
        head_dim=64,
        n_groups=1,
        block_len=64,
        version=2,
    )
    _deltanet_bidir_layer_spec = rf.build_dict(
        BidirDeltaNetChunkedLayer,
        d_state=128,
        head_dim=64,
        block_len=32,
        version=3,
        # Separate fwd/bwd DeltaNet blocks (full capacity per direction); ConMamba pattern doesn't
        # carry over for DeltaNet (no scan-internal weights to share independently of projections).
        bidir_share_weights=False,
    )
    _num_layers = 16
    _mamba2_per_layer = [_std_layer_spec if i % 2 == 0 else _mamba2_layer_spec for i in range(_num_layers)]
    _deltanet_per_layer = [_std_layer_spec if i % 2 == 0 else _deltanet_layer_spec for i in range(_num_layers)]
    _mamba2_bidir_per_layer = [_std_layer_spec if i % 2 == 0 else _mamba2_bidir_layer_spec for i in range(_num_layers)]
    _deltanet_bidir_per_layer = [
        _std_layer_spec if i % 2 == 0 else _deltanet_bidir_layer_spec for i in range(_num_layers)
    ]
    # Bidir Mamba-2 with the bwd-SSD super-batch processed in slices of 256
    # (see :class:`BidirMamba2ChunkedLayer`'s ``ssd_super_batch_chunk``);
    # lets us train at the same ``bs // 2`` as the causal hybrids and deltanet-bidir,
    # for a fair memory/throughput comparison.
    _mamba2_bidir_ssdchunk_layer_spec = rf.build_dict(
        BidirMamba2ChunkedLayer,
        d_state=128,
        d_conv=4,
        expand=2,
        head_dim=64,
        n_groups=1,
        block_len=64,
        ssd_super_batch_chunk=256,
        version=2,
    )
    _mamba2_bidir_ssdchunk_per_layer = [
        _std_layer_spec if i % 2 == 0 else _mamba2_bidir_ssdchunk_layer_spec for i in range(_num_layers)
    ]

    def _hybrid_enc_build_dict(per_layer):
        return rf.build_dict(
            ChunkedConformerEncoderV2,
            encoder_layer=_std_layer_spec,
            encoder_layer_per_layer=per_layer,
            chunk_size=center_size,
            chunk_history_size=left_n * center_size,
            chunk_lookahead_size=right_size,
            chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
            chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
            chunk_lookahead_size_train_pool=[right_size, right_size // 2],
            use_chunk_type_embedding=True,
            version=3,
        )

    # CTC-only: 10.52 (vs conformer dyn-rope-ctembed 9.41; Mamba-2 SSD hybrid encoder).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-mamba2",
        {
            "model.enc_build_dict": _hybrid_enc_build_dict(_mamba2_per_layer),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # CTC-only: 11.09 (vs conformer dyn-rope-ctembed 9.41; DeltaNet hybrid encoder).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-deltanet",
        {
            "model.enc_build_dict": _hybrid_enc_build_dict(_deltanet_per_layer),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # Bidir variants. Mamba2-bidir matches the ConMamba / Speech Slytherin hybrid:
    # shared in_proj + out_proj, per-direction conv1d/A/D/dt_bias, average + shared gate.
    # DeltaNet-bidir has fully separate forward / backward DeltaNetBlock instances (no scan-
    # internal weights to share, so the binary choice is full duplication).
    # Plain mamba2-bidir (bs/4) dropped -- superseded by mamba2-bidir-ssdchunk256 (bs/2, fairer compare).
    # CTC-only: 11.41 (uni deltanet 11.09 -- bidir does NOT help here).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-deltanet-bidir",
        {
            "model.enc_build_dict": _hybrid_enc_build_dict(_deltanet_bidir_per_layer),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    # Bidir Mamba-2 with ``ssd_super_batch_chunk=256``, run at ``bs // 2``
    # to compare directly against the causal mamba2 + bidir deltanet at the same batch size.
    # See the existing ``-mamba2-bidir`` variant for the ``bs // 4`` fallback that doesn't need the chunking option.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-mamba2-bidir-ssdchunk256",
        {
            "model.enc_build_dict": _hybrid_enc_build_dict(_mamba2_bidir_ssdchunk_per_layer),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # More epochs (2xtrain).
    # Is overlap's gain (9.46 -> 9.26) just ~2x compute, and is lookahead even needed?
    # Train the plain v2.3 baseline for 2x as long (total_k_hours 100 -> 200, i.e. 8 full epochs),
    # then compare its CTC-only to overlap's 9.26.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-2xtrain",
        {
            "total_k_hours": 200,
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
        },
    )

    # Does overlap remove the need for lookahead?
    # v2.3-overlap but with chunk_lookahead_size=0 (R0).
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R0-v2.3-overlap",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=0,
                chunk_num_overlaps=2,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor // 2,
            "train.max_seqs": max_seqs,
        },
    )

    # dynCx3:
    # Oversample the small (streaming) center_size in the train pool,
    # on the best dyn config (dyn-rope-ctembed).
    # Pool: C x3 vs x1 for 2C/4C/8C/offline, so the deployment chunk (C=5) is seen ~3/7 instead of 1/5.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dynCx3-rope-ctembed",
        {
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size] * 3 + [center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )

    # --- Offline -> chunked finetune comparison (all at ~2x base compute) ---
    # base-2xtrain / dyn-rope-ctembed-2xtrain: from scratch for 2x epochs (total_k_hours 200).
    # impBase: init from the 1x base, finetune dyn-rope-ctembed for 1x epochs,
    #   so 100 base + 100 finetune = 200 -- same budget as the 2xtrain controls.
    # Finetune LR: keep warmup but short (step_peak_fraction=0.01, ~1 epoch), sweep peak via base_lr.
    # Partial preload (ignore_missing): conformer blocks + decoder transfer; rope/ctembed init fresh.
    train("base-2xtrain", {"total_k_hours": 200})
    # CTC-only: 8.52 (1x dyn-rope-ctembed 9.41; clear gain). AED+CTC first-pass dev 7.25.
    train(
        f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-2xtrain",
        {
            "total_k_hours": 200,
            "model.enc_build_dict": rf.build_dict(
                ChunkedConformerEncoderV2,
                encoder_layer=rf.build_dict(ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2),
                chunk_size=center_size,
                chunk_history_size=left_n * center_size,
                chunk_lookahead_size=right_size,
                chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                use_chunk_type_embedding=True,
                version=3,
            ),
            "train.batch_size": bs * configs._batch_size_factor,
            "train.max_seqs": max_seqs,
            "lm_recog_extra.__serialization_version_stats": 2,
        },
    )
    _impbase_prefix = f"chunked-L{left_n * center_size}-C{center_size}-R{right_size}-v2.3-dyn-rope-ctembed-impBase"
    for _ft_base_lr in [0.5, 0.25, 0.1]:
        train(
            f"{_impbase_prefix}-baseLr{_ft_base_lr}",
            {
                "train_update_func_from_n_ep": (
                    lambda n_ep, _lr=_ft_base_lr: {
                        "train": configs._get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep, base_lr=_lr, step_peak_fraction=0.01)
                    }
                ),
                "model.enc_build_dict": rf.build_dict(
                    ChunkedConformerEncoderV2,
                    encoder_layer=rf.build_dict(
                        ChunkedConformerEncoderLayerV2, self_att=ChunkedRotaryPosSelfAttentionV2
                    ),
                    chunk_size=center_size,
                    chunk_history_size=left_n * center_size,
                    chunk_lookahead_size=right_size,
                    chunk_size_train_pool=[center_size, center_size * 2, center_size * 4, center_size * 8, None],
                    chunk_history_size_train_pool=[left_n * center_size, left_n * center_size // 2],
                    chunk_lookahead_size_train_pool=[right_size, right_size // 2],
                    use_chunk_type_embedding=True,
                    version=3,
                ),
                "train.batch_size": bs * configs._batch_size_factor,
                "train.max_seqs": max_seqs,
                "train.preload_from_files": {
                    "base": {
                        "prefix": "",
                        "filename": _exp_base.get_last_fixed_epoch().checkpoint,
                        "init_for_train": True,
                        "ignore_missing": True,  # adds chunk_type_embedding and RoPE
                    }
                },
                "lm_recog_extra.__serialization_version_stats": 2,
            },
        )

    # New observations 15.5.2026 - 29.5.2026:
    # - RoPE made much faster (faster apply_rope).
    # - Realization: RoPE actually not expected to be faster than relpos self-att.
    # - But RoPE was better? Why? This needs some more investigation.
    # - Trying also with learnable pos emb (relposL). But worse. Stopping further investigations now.
    # - RoPE also helps for dyn-ctembed vs dyn-rope-ctembed.
    # - Overlap helps on its own but regresses in dyn-rope-ctembed.
    # - Overlap dynamic (overlapD) regresses less but still regresses.
    # - To understand when does overlap regress, testing also:
    #   - dyn-rope-ctembed-overlapD-ctembedfix (ctembedfix)
    #   - dyn-rope-overlapD (without ctembed)
    # - DynV3 also not good. Needs more work.
    # - Running Mamba-2 SSD and DeltaNet, and bidir variants.
    # - WBE/TSE. Goal is to also compute actual latency.

    # To report for next time:
    # - Dyn-pool comparison (rope+ctembed fixed, only the train pools vary), dev / test (train h):
    #   dyn 9.41 / 10.29 (128.3h; run2 9.52 / 10.21, 107.3h),
    #   dynCx3 9.47 / 10.28 (120.8h, oversample small C, no gain),
    #   dynV3 10.33 / 10.99 (97.1h, adds 0 to history+lookahead pools),
    #   dynV2 10.85 / 11.71 (100.1h, worst).
    #   non-dyn and dynV4 still running.
    #   Decisive knob: a 0 in the history/lookahead pools hurts, standard dyn best.
    # - R0-v2.3-overlap run. (TODO put result here once ready)
    # - Overlap at recog only (-ov2): hurts; C5-R4-ov2 10.65 (vs 9.41), C5-R2-ov2 18.10 (vs 10.14).
    # - 2xtrain (TODO put result here once ready)
    # - dyn-rope-ctembed-overlap-mse: MSE helps the overlap variant (dev 10.20 -> 9.99),
    #   but overlap still regresses vs no-overlap dyn-rope-ctembed (9.41).
    #   Reference: overlap 10.20 / 11.03, overlap-mse 9.99 / 10.85, overlapD 9.75 / 10.56;
    #   overlapD-ctembedfix 9.76 / 10.63 (~= overlapD, ctembedfix no help);
    #   dyn-rope-overlapD (no ctembed) 10.41 / 11.44.
    # - dyn-rope-ctembed-impBase (finetune from offline base): clear gain.
    #   baseLr0.25 best (dev 8.83 / test 9.68), baseLr0.5 ~tied (8.84 / 9.71),
    #   baseLr0.1 weaker (8.97 / 9.85); vs from-scratch dyn-rope-ctembed 9.41 / 10.29.
    # - base-2xtrain: dev 6.58 / test 7.41 (vs base 1x 7.32 / 8.10).
    # - dyn-rope-ctembed-2xtrain: dev 8.49 / test 9.19 (213.7h),
    #   clear gain over 1x (9.41 / 10.29).
    #   AED+CTC first-pass dev 7.25 / test 7.90.
    #   (Best from-scratch chunked streaming result so far.)
    # - longform: so far ONLY chunked-L80-C5-R4-v2.3-dyn-rope-ctembed, streaming-KV seg10.
    #   seg.test 5.12, long.test 4.97,
    #   but these are two SEPARATE HF datasets:
    #   seg = HF Open-ASR-Leaderboard "tedlium" (its own segmentation + text normalization),
    #   long = distil-whisper/tedlium-long-form (full talks).
    #   NOT verified to use the same reference text / normalization / ignored-region handling,
    #   so seg-vs-long is NOT a valid comparison yet.
    #   TODO: verify reference comparability before comparing the two numbers,
    #   and run long-form for more models (other chunk sizes, offline base),
    #   to make long-form itself comparable across models.
    # - Streaming emission latency (TIMIT test, mean over words; +Lms / CTC-only dev WER).
    #   The mean is essentially the structural E[latency] = (C/2 + R) frames * ~58ms,
    #   so at matched geometry it carries little signal beyond C and R.
    #   At L80-C5-R4 (expected ~380ms) the well-behaved variants just cluster there (+-40ms is noise):
    #   non-dyn 411 / 9.46, rope 421 / 9.31, dyn 366 / 9.66, dyn-rope 371 / 9.55,
    #   dyn-rope-ctembed 360 / 9.41, dynCx3 406 / 9.47.
    #   The only real deviations are premature-emission variants, all with bad WER (WER already flags them):
    #   L0-C5-R4 +77 / 24.1, dynV3 +247 / 10.49, dynV2 +193 / 11.0.
    #   The chunk-geometry means (C10-R8 512, C20-R15 920, C40-R0 528, C100-R15 1937) do NOT cleanly equal
    #   (C/2 + R) and are confounded (C40-R0 mean 528ms vs its own first-word 1299ms),
    #   so mean-latency is unreliable for large chunks; first-word latency is the cleaner metric there.
    #   Net: latency adds little beyond WER + (C, R); it mainly detects premature emission.
    #   offline base +inf (whole seq needed).
    # - Linear-attention encoders, all worse than conformer dyn-rope-ctembed 9.41 / 10.29:
    #   mamba2 dev 10.47 / test 11.35 (174.0h),
    #   deltanet 11.09 / 11.93 (164.3h),
    #   deltanet-bidir 11.41 / 12.28 (197.9h, bidir HURTS vs uni).
    #   mamba2 best of the linear-attn set.
    #   TODO still running: mamba2-bidir, mamba2-bidir-ssdchunk256.
    # TODO fill here until next time...


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
    # TODO (for later): Bug: train_vocab_opts/dataset_train_opts are not actually plumbed through aed_train_exp.
    #   Not a quick fix: patching now would silently change every training run and break
    #   comparability with all existing results.
    #   Revisit when starting a fresh batch of experiments where breakage is acceptable.
    "train_vocab_opts": {"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
    "dataset_train_opts": {"train_epoch_split": 1, "train_epoch_wise_filter": None},
    "env_updates": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
    "lm_recog_extra": {},
}


def train(
    name: str,
    config: Dict[str, Any],
    config_overrides: Optional[Dict[str, Any]] = None,
    *,
    recog_def_ctc_only: bool = True,
):
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
    lm_recog_extra_config = config.pop("lm_recog_extra")

    assert not config

    aux_ctc_layer = max(
        [i for i in train_config["aux_loss_layers"] if i <= model_config["enc_build_dict"]["num_layers"]]
    )

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
        recog_def=ctc_model_recog if recog_def_ctc_only else None,
        search_config={"aux_loss_layers": [aux_ctc_layer]} if recog_def_ctc_only else None,
    )
    aed_ctc_timesync_recog_recomb_auto_scale(
        prefix=prefix + "/aed/" + name + "/aed+ctc",
        task=task,
        aed_ctc_model=exp.get_last_fixed_epoch(),
        aux_ctc_layer=aux_ctc_layer,
    )
    if vocab == "spm10k":
        lm_name, lm = get_lm(prefix=prefix, vocab=vocab)
        ctc_recog_recomb_labelwise_prior_auto_scale(
            prefix=f"{prefix}/aed/{name}/ctc+lm-v2/{lm_name}",
            task=task,
            ctc_model=exp.get_last_fixed_epoch(),
            extra_config={
                "aux_loss_layers": [
                    max(
                        [
                            i
                            for i in train_config["aux_loss_layers"]
                            if i <= model_config["enc_build_dict"]["num_layers"]
                        ]
                    )
                ],
                **lm_recog_extra_config,
            },
            lm=lm,
            prior_dataset=get_loquacious_train_subset_dataset_v2(vocab="spm10k"),
        )

    # TIMIT forced-align stats (WBE + streaming latency) for every CTC model.
    # ff* (feed-forward decoder) and *-bug experiments are not part of the
    # chunked-CTC study, so they're skipped. Chunk geometry for the latency
    # metric is read from the encoder build dict;
    # offline / full-context models have no chunk geometry => None.
    if not (name.startswith("ff") or name.endswith("-bug")):
        _enc_bd = model_config.get("enc_build_dict") if isinstance(model_config, dict) else None
        _chunk_c, _chunk_la = _chunk_geometry_from_enc_build_dict(_enc_bd)
        _run_align_stats(
            name,
            exp,
            aux_ctc_layer,
            chunk_center_frames=_chunk_c,
            chunk_lookahead_frames=_chunk_la,
            vocab_str=vocab,
        )

    return exp, task, aux_ctc_layer


def recog_model_with_config_overwrite(
    *,
    model: ModelWithCheckpoint,
    task,
    recog_def,
    config_overwrites: Optional[Dict[str, Any]] = None,
    extra_config: Optional[Dict[str, Any]] = None,
    name: str,
    tag: str,
):
    """Run :func:`recog_model` after deep-merging ``config_overwrites`` into the model's
    own config (dotted keys supported via :func:`dict_update_deep`). ``extra_config``
    is passed through to ``recog_model`` as its top-level ``config`` arg. Registers
    the result under ``name`` via :func:`tk.register_output`. Returns the score result.

    Useful e.g. for sweeping recog-time chunking on a dyn-trained model without
    touching shared infra.
    """
    _prefix = get_setup_prefix_for_module(__name__)
    if config_overwrites:
        orig_def = model.definition
        if isinstance(orig_def, ModelDefWithCfg):
            new_cfg = dict_update_deep(orig_def.config, config_overwrites)
            new_def = ModelDefWithCfg(orig_def.model_def, new_cfg)
        else:
            new_def = ModelDefWithCfg(orig_def, dict(config_overwrites))
        model = ModelWithCheckpoint(definition=new_def, checkpoint=model.checkpoint)
    res = recog_model(task=task, model=model, recog_def=recog_def, config=extra_config)
    tk.register_output(f"{_prefix}/aed/{name}/ctc-recog-sweep/{tag}", res.output)
    return res


@cache
def get_lm(*, prefix: str, vocab: str, num_full_ep: int = 5, split: int = 10) -> Tuple[str, ModelWithCheckpoint]:
    from sisyphus import tk
    from i6_experiments.users.zeyer.utils.dict_update import dict_update_deep
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.configs import (
        config_96gb_bf16_accgrad1,
        _get_cfg_lrlin_oclr_by_bs_nep_v4,
    )
    from i6_experiments.users.zeyer.decoding.perplexity import (
        get_lm_perplexities_for_task_evals,
    )
    from i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.lm import lm_model_def, lm_train_def
    from i6_experiments.users.zeyer.train_v4 import train as _train, ModelDefWithCfg

    from i6_experiments.users.zeyer.datasets.loquacious import (
        get_loquacious_task_raw_v2,
        get_loquacious_text_only_dataset,
    )

    import returnn.frontend as rf
    from returnn.frontend.decoder.transformer import TransformerDecoder

    n_ep = round(num_full_ep * split)
    # orig name: trafo-n32-d1024-noAbsPos-rmsNorm-ffGated-rope-noBias-drop01-b400_20k-nEp...-spm10k
    name = f"trafo-n32-d1024-nFullEp{num_full_ep}-nEp{n_ep}-{vocab}"
    exp = _train(
        f"{prefix}/lm/{name}",
        config=dict_update_deep(
            config_96gb_bf16_accgrad1,
            {
                **_get_cfg_lrlin_oclr_by_bs_nep_v4(n_ep),
                "batch_size": 20_000,
                "max_seqs": 400,
                "optimizer.weight_decay": 1e-2,
                "calculate_exp_loss": True,
            },
        ),
        train_dataset=get_loquacious_text_only_dataset(vocab="spm10k", train_epoch_split=split),
        model_def=ModelDefWithCfg(
            lm_model_def,
            {
                "_model_def_dict": rf.build_dict(
                    TransformerDecoder,
                    encoder_dim=None,
                    num_layers=32,
                    model_dim=1024,
                    pos_enc=None,
                    norm=rf.build_dict(rf.RMSNorm),
                    ff=rf.build_dict(rf.decoder.transformer.FeedForwardGated),
                    decoder_layer_opts=dict(self_att=rf.build_dict(rf.RotaryPosCausalSelfAttention, with_bias=False)),
                    dropout=0.1,
                    att_dropout=0.1,
                )
            },
        ),
        train_def=lm_train_def,
    )

    task = get_loquacious_task_raw_v2(vocab=vocab)
    perplexities_nlm = get_lm_perplexities_for_task_evals(task, label_level="task", lm=exp.get_last_fixed_epoch())
    for eval_set_name, ppl in perplexities_nlm.items():
        tk.register_output(f"{prefix}/lm/{name}/ppl/{eval_set_name}", ppl)

    return name, exp.get_last_fixed_epoch()


class ChunkedConformerEncoder(chunked_conformer_v1.ChunkedConformerEncoder):
    """alias"""


class ChunkedConformerEncoderLayer(chunked_conformer_v1.ChunkedConformerEncoderLayer):
    """alias"""


# ---------------------------------------------------------------------------
# Word-boundary alignment evaluation on TIMIT (forced-align + WBE/TSE).
# ---------------------------------------------------------------------------


def _aed_ctc_model_forced_align_step(*, model, extern_data, **_kwargs):
    """
    Forward step: per-frame best-path CTC alignment
    from the AED model's top CTC head.

    Mirrors :func:`exp2024_09_16_grad_align._ctc_model_forced_align_step`,
    but uses ``model.encode_and_get_ctc_log_probs`` instead of ``model(...)``,
    because the AED model's ``__call__`` performs AED decoding,
    not CTC forward.
    """
    from returnn.tensor import batch_dim
    from returnn.config import get_global_config
    from i6_experiments.users.zeyer.nn_rf.fsa import best_path_ctc

    config = get_global_config()
    default_input_key = config.typed_value("default_input")
    default_target_key = config.typed_value("target")
    source = extern_data[default_input_key]
    targets = extern_data[default_target_key]
    expected_output = rf.get_run_ctx().expected_outputs["output"]
    out_spatial_dim = expected_output.dims[-1]

    log_probs, _, enc_spatial_dim = model.encode_and_get_ctc_log_probs(source, in_spatial_dim=source.get_time_dim_tag())
    path, score = best_path_ctc(
        logits=log_probs,
        logits_normalized=True,
        input_spatial_dim=enc_spatial_dim,
        targets=targets,
        targets_spatial_dim=targets.get_time_dim_tag(),
        blank_index=model.blank_idx,
    )
    out_spatial_dim.declare_same_as(enc_spatial_dim)
    path.mark_as_default_output(shape=[batch_dim, enc_spatial_dim])
    score.mark_as_output("scores", shape=[batch_dim])


def _aed_ctc_forced_align(
    model: ModelWithCheckpoint,
    dataset,
    *,
    aux_ctc_layer: int,
    extra_config: Optional[Dict[str, Any]] = None,
    num_shards: Optional[int] = None,
) -> Union[tk.Path, List[tk.Path]]:
    """
    AED counterpart of :func:`exp2024_09_16_grad_align.ctc_forced_align`.

    Selects the AED model's top CTC aux head
    via ``aux_loss_layers=[aux_ctc_layer]`` in the forward config,
    same wiring as the existing recog path.

    Single-GPU by default (``num_shards=None``): one :func:`forward_to_hdf` job -> one HDF path.
    With ``num_shards`` set, the same forward runs sharded across a full multi-GPU node via
    :func:`batched_forward_to_hdf` (JUPITER policy: 4x GH200, flat-per-node billing, 12 h wall) and
    returns the list of shard HDFs (feed to ``HDFDataset(files=[...])``). For large corpora (e.g.
    Loquacious ~25k h) the single-GPU path would both waste 3/4 of the node and exceed the wall.
    The ``num_shards`` default of None keeps the existing single-GPU callers hash-identical.
    """
    extern_data_dict = dataset.get_extern_data()
    default_target_dict = extern_data_dict[dataset.get_default_target()]
    classes_dim = default_target_dict["sparse_dim"]
    classes_with_blank_dim = classes_dim + 1

    fwd_config = {
        "model_outputs": {
            "output": {"shape": (None,), "sparse_dim": classes_with_blank_dim},
            "scores": {"shape": ()},
        },
        "aux_loss_layers": [aux_ctc_layer],
    }
    if extra_config:
        fwd_config.update(extra_config)

    if num_shards is not None:
        from i6_experiments.users.zeyer.forward_batched import batched_forward_to_hdf

        return batched_forward_to_hdf(
            dataset=dataset,
            num_shards=num_shards,
            model=model,
            forward_step=_aed_ctc_model_forced_align_step,
            config=fwd_config,
        )

    from i6_experiments.users.zeyer.forward_to_hdf import forward_to_hdf

    return forward_to_hdf(
        dataset=dataset,
        model=model,
        forward_step=_aed_ctc_model_forced_align_step,
        config=fwd_config,
        forward_rqmt={"time": 4},
    )


def _chunk_geometry_from_enc_build_dict(enc_build_dict) -> Tuple[Optional[int], int]:
    """
    Chunk (center, lookahead) in encoder-output frames for the streaming-latency metric,
    or (None, 0) for offline / non-chunked encoders.

    Two encoder schemas carry the geometry differently:
      - ChunkedConformerEncoderV2 (the ``-v2.3`` family):
        ``chunk_size`` is the center, ``chunk_lookahead_size`` the lookahead,
        both already in encoder-output frames.
      - ChunkedConformerEncoder (the geometry sweep):
        ``end_chunk_size_dim`` is the center in encoder frames,
        ``chunk_stride`` is center * downsampling,
        ``input_chunk_size_dim`` is (center + lookahead) * downsampling (raw frames),
        so lookahead = input_chunk_size_dim // downsampling - center.
    """
    if not isinstance(enc_build_dict, dict):
        return None, 0
    if enc_build_dict.get("chunk_size") is not None:
        center = int(enc_build_dict["chunk_size"])
        return center, int(enc_build_dict.get("chunk_lookahead_size", 0) or 0)
    end_dim = enc_build_dict.get("end_chunk_size_dim")
    stride = enc_build_dict.get("chunk_stride")
    input_dim = enc_build_dict.get("input_chunk_size_dim")
    if end_dim is not None and stride and input_dim is not None:
        center = int(end_dim)
        downsampling = int(stride) // center
        return center, int(input_dim) // downsampling - center
    return None, 0


def _run_align_stats(
    name: str,
    exp,
    aux_ctc_layer: int,
    *,
    chunk_center_frames: Optional[int] = None,
    chunk_lookahead_frames: int = 0,
    vocab_str: str = "spm10k",
) -> None:
    """
    Forced-align on TIMIT val+test + WBE / TSE metric
    for the given trained model.

    Single forced-align variant (no prior).
    Registers per-(corpus,split) outputs
    ``align-stats/<name>/<corpus>-<split>/{alignment.hdf, wbe.txt, report.txt,
    latency.txt, latency-report.txt}``.
    The latency metric (:class:`CalcCtcStreamingLatencyFromHfDatasetJob`) is
    chunked-model specific; ``chunk_center_frames=None`` => offline (full sequence).
    """
    from i6_experiments.users.zeyer.datasets.loquacious import get_vocab_by_str
    from i6_experiments.users.zeyer.datasets.hf_timit_buckeye import (
        get_dataset_offset_factor,
        get_hf_word_align_dataset_config,
        get_hf_word_align_dataset_dir,
    )
    from i6_experiments.users.zeyer.alignment.ctc_wbe_from_hf_dataset import (
        CalcCtcWbeFromHfDatasetJob,
        CalcCtcStreamingLatencyFromHfDatasetJob,
    )

    prefix = get_setup_prefix_for_module(__name__) + "/align-stats/" + name
    vocab = get_vocab_by_str(vocab_str)
    model = exp.get_last_fixed_epoch()

    # The Loquacious-trained SPM10k vocab here is all-uppercase
    # (so we must uppercase the joined TIMIT / Buckeye utterance,
    # otherwise every word tokenizes to <unk> silently).
    text_case = "upper"
    for corpus in ["timit"]:
        ds_dir = get_hf_word_align_dataset_dir(corpus, text_case=text_case)
        offset_factor = get_dataset_offset_factor(corpus)
        for split in ["val", "test"]:
            tag = f"{corpus}-{split}"
            ds = get_hf_word_align_dataset_config(name=corpus, split=split, vocab=vocab, text_case=text_case)
            alignment_hdf = _aed_ctc_forced_align(model, ds, aux_ctc_layer=aux_ctc_layer)
            alignment_hdf.creator.add_alias(f"{prefix}/{tag}/forced-align")
            tk.register_output(f"{prefix}/{tag}/alignment.hdf", alignment_hdf)

            wbe_job = CalcCtcWbeFromHfDatasetJob(
                alignment_hdf=alignment_hdf,
                spm_model_file=vocab.model_file,
                blank_idx=vocab.dim,
                dataset_dir=ds_dir,
                dataset_key=split,
                dataset_offset_factor=offset_factor,
            )
            wbe_job.add_alias(f"{prefix}/{tag}/wbe-metric")
            tk.register_output(f"{prefix}/{tag}/wbe.txt", wbe_job.out_wbe)
            tk.register_output(f"{prefix}/{tag}/report.txt", wbe_job.out_report)

            latency_job = CalcCtcStreamingLatencyFromHfDatasetJob(
                alignment_hdf=alignment_hdf,
                spm_model_file=vocab.model_file,
                blank_idx=vocab.dim,
                dataset_dir=ds_dir,
                dataset_key=split,
                dataset_offset_factor=offset_factor,
                chunk_center_frames=chunk_center_frames,
                chunk_lookahead_frames=chunk_lookahead_frames,
            )
            latency_job.add_alias(f"{prefix}/{tag}/latency-metric")
            tk.register_output(f"{prefix}/{tag}/latency.txt", latency_job.out_latency)
            tk.register_output(f"{prefix}/{tag}/first-word-latency.txt", latency_job.out_first_word_latency)
            tk.register_output(f"{prefix}/{tag}/latency-report.txt", latency_job.out_report)
