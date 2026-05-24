"""Standalone test: streaming-encoder segmentation matches offline batched encoder.

Operates on synthetic mel features (skip log_mel) with a small ChunkedConformerEncoderV2
(random weights), so it iterates in seconds. Run with::

    export PYTHONPATH=recipe:/rwthfs/rz/cluster/home/az668407/setups/combined/2021-05-31/tools/returnn
    python -m i6_experiments.users.zeyer.nn_rf.encoder.chunked_streaming_test
"""

from __future__ import annotations

import math
from typing import Tuple

import torch

from returnn.util import BehaviorVersion, better_exchook
import returnn.frontend as rf
from returnn.tensor import Dim, Tensor, batch_dim

from returnn.frontend.encoder.conformer import (
    ConformerConvSubsample,
    ConformerPositionwiseFeedForward,
)
from i6_experiments.users.zeyer.nn_rf.encoder.chunked_conformer_v2 import (
    ChunkedConformerEncoderV2,
    ChunkedConformerEncoderLayerV2,
)


_FEAT_DIM = Dim(name="logmel", dimension=80, kind=Dim.Types.Feature)
_OUT_DIM = Dim(name="enc_out", dimension=64, kind=Dim.Types.Feature)


def _setup():
    BehaviorVersion.set_min_behavior_version(25)
    rf.select_backend_torch()
    if batch_dim.dyn_size_ext is None:
        batch_dim.dyn_size_ext = rf.convert_to_tensor(1, dims=[])


def _build_encoder(*, chunk_size: int, history: int, lookahead: int) -> ChunkedConformerEncoderV2:
    return rf.build_from_dict(
        rf.build_dict(
            ChunkedConformerEncoderV2,
            input_layer=rf.build_dict(
                ConformerConvSubsample,
                out_dims=[32, 32, 32],
                filter_sizes=[(3, 3), (3, 3), (3, 3)],
                pool_sizes=[(1, 2)],
                strides=[(1, 1), (3, 1), (2, 1)],  # downsampling 6
            ),
            num_layers=2,
            out_dim=_OUT_DIM.dimension,
            encoder_layer=rf.build_dict(
                ChunkedConformerEncoderLayerV2,
                ff=rf.build_dict(
                    ConformerPositionwiseFeedForward,
                    activation=rf.build_dict(rf.relu_square),
                    with_bias=False,
                ),
                num_heads=4,
            ),
            chunk_size=chunk_size,
            chunk_history_size=history,
            chunk_lookahead_size=lookahead,
            version=3,
            adapt_chunk_history_for_short_seqs=False,
        ),
        _FEAT_DIM,
    )


def _make_mel_input(seq_len_mel: int) -> Tuple[Tensor, Dim]:
    """Random mel input [B=1, T, F]."""
    time_dim = Dim(
        rf.convert_to_tensor([seq_len_mel], dims=[batch_dim], name="mel_time"),
        name="mel_time",
    )
    return rf.random_normal([batch_dim, time_dim, _FEAT_DIM]), time_dim


def _streaming_encode_mel(
    encoder: ChunkedConformerEncoderV2,
    mel: Tensor,
    mel_spatial_dim: Dim,
    *,
    segment_chunks: int,
) -> Tuple[torch.Tensor, int]:
    """Mel-level mirror of streaming_encode_and_get_ctc_log_probs (encoder-only).

    Returns (enc_torch [B, T_enc, D], offline_enc_total).
    """
    chunk_size = encoder.chunk_size
    history = encoder.chunk_history_size
    lookahead = encoder.chunk_lookahead_size
    n_overlap = encoder.chunk_num_overlaps
    ds_internal = encoder._input_downsample_factor  # noqa: SLF001

    chunk_stride = chunk_size // n_overlap
    mel_per_enc_frame = ds_internal  # mel frames per encoder frame
    chunk_stride_mel = chunk_stride * mel_per_enc_frame
    history_mel = history * mel_per_enc_frame
    lookahead_mel = lookahead * mel_per_enc_frame
    segment_mel = segment_chunks * chunk_stride_mel

    mel_torch = mel.raw_tensor  # [B=1, T_mel, F]
    mel_len = int(mel_spatial_dim.dyn_size_ext.raw_tensor.reshape(-1)[0].item())

    offline_enc_total = math.ceil(mel_len / chunk_stride_mel) * chunk_stride
    target_per_seg_full = chunk_stride * segment_chunks
    frames_taken = 0
    chunks: list[torch.Tensor] = []
    n_seg = max(1, math.ceil(mel_len / segment_mel))

    for k in range(n_seg):
        seg_start = k * segment_mel
        seg_end = min(mel_len, (k + 1) * segment_mel + lookahead_mel)
        win_start = max(0, seg_start - history_mel)
        win_end = seg_end
        if win_end <= win_start:
            continue
        seg_mel = mel_torch[:, win_start:win_end, :].contiguous()
        seg_len = seg_mel.shape[1]
        seg_dim = Dim(
            rf.convert_to_tensor([seg_len], dims=[batch_dim], name=f"mel_seg{k}"),
            name=f"mel_seg{k}",
        )
        seg_t = Tensor("mel_seg", dims=[batch_dim, seg_dim, _FEAT_DIM], dtype=mel.dtype, raw_tensor=seg_mel)

        enc_t, enc_spatial_dim = encoder(seg_t, in_spatial_dim=seg_dim)
        # Permute to [B, T, D]
        dorder = enc_t.dims
        out_dim = encoder.out_dim
        perm = [dorder.index(batch_dim), dorder.index(enc_spatial_dim), dorder.index(out_dim)]
        enc_torch = enc_t.raw_tensor.permute(*perm).contiguous()

        hist_mel = seg_start - win_start
        hist_enc = hist_mel // mel_per_enc_frame
        target = min(target_per_seg_full, offline_enc_total - frames_taken)
        if target <= 0:
            break
        central = enc_torch[:, hist_enc : hist_enc + target, :]
        if central.shape[1] < target:
            raise RuntimeError(f"Seg {k}: got {central.shape[1]} central frames, expected {target}")
        chunks.append(central)
        frames_taken += target
        if frames_taken >= offline_enc_total:
            break

    return torch.cat(chunks, dim=1), offline_enc_total


def test_streaming_matches_offline():
    _setup()
    encoder = _build_encoder(chunk_size=5, history=80, lookahead=4)
    # Long enough for several segments at segment_chunks=10 → segment_mel=300, history_mel=480 → first segment
    # has no history; subsequent segments have full history.
    mel, mel_dim = _make_mel_input(seq_len_mel=1800)  # ~3 segments at segment_chunks=10

    with torch.no_grad():
        # Offline
        enc_off_t, enc_off_dim = encoder(mel, in_spatial_dim=mel_dim)
        dorder = enc_off_t.dims
        out_dim = encoder.out_dim
        perm = [dorder.index(batch_dim), dorder.index(enc_off_dim), dorder.index(out_dim)]
        enc_off = enc_off_t.raw_tensor.permute(*perm).contiguous()
        # Streaming
        enc_str, offline_enc_total = _streaming_encode_mel(encoder, mel, mel_dim, segment_chunks=10)

    print(f"offline shape: {enc_off.shape}  streaming shape: {enc_str.shape}  expected T_enc={offline_enc_total}")
    assert enc_off.shape == enc_str.shape, f"shape mismatch: {enc_off.shape} vs {enc_str.shape}"

    diff = (enc_off - enc_str).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    print(f"max_abs_diff={max_diff:.6g}  mean_abs_diff={mean_diff:.6g}")

    # Per-frame divergence summary: which frame index has the largest diff?
    diff_per_t = diff.mean(dim=(0, 2))  # [T]
    worst_t = int(diff_per_t.argmax().item())
    print(
        f"worst frame index t={worst_t}  diff={diff_per_t[worst_t].item():.6g}  "
        f"context: t-5..t+5 diffs = {[round(diff_per_t[i].item(), 6) for i in range(max(0, worst_t - 5), min(diff_per_t.shape[0], worst_t + 6))]}"
    )

    assert max_diff < 1e-4, f"max diff {max_diff} too large"
    print("MATCH ✓")


def main():
    better_exchook.install()
    test_streaming_matches_offline()


if __name__ == "__main__":
    main()
