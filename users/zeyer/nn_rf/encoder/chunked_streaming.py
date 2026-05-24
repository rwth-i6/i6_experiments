"""Streaming CTC encode + recog for the chunked encoder.

The chunked encoder is mathematically streamable (each chunk attends only to a
local + history + lookahead window), but the default ``__call__`` runs all
chunks in one batched forward — memory scales linearly with the full audio
length.

This module provides:

- :func:`streaming_encode_and_get_ctc_log_probs` -- loops in Python over audio
  segments of a configurable duration (seconds), re-encoding ``chunk_history``
  worth of audio context per segment so the output matches the offline-batched
  ``model.encode_and_get_ctc_log_probs`` modulo floating point. Memory is
  bounded by segment duration (+ history).
- :func:`model_recog_ctc_streaming` — a RecogDef that uses the helper above
  and then runs the same CTC beam search as
  :func:`exp2024_04_23_baselines.ctc.model_recog`.

Limitations of v1:
- ``max_seqs = 1`` (single sequence per batch). Enforced via assertion;
  set ``"max_seqs": 1`` in the recog config.
- Re-encodes the history audio per segment (no KV caching). Compute overhead
  ≈ ``(segment_seconds + history_seconds) / segment_seconds`` ≈ 1.5× for a
  10-second segment and a 4.8-second history.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


_SAMPLE_RATE = 16_000
_MEL_HOP_SAMPLES = 160  # 10ms hop at 16kHz → 100Hz mel; encoder downsamples another factor `ds_internal`.


def streaming_encode_and_get_ctc_log_probs(
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    *,
    segment_seconds: float = 10.0,
) -> Tuple[Tensor, Dim]:
    """Streaming variant of ``model.encode_and_get_ctc_log_probs``.

    Returns ``(label_log_prob, enc_spatial_dim_full)`` -- same shape/semantics as
    the offline call, computed via per-segment forward passes that re-encode
    ``chunk_history`` worth of audio context before each segment so the per-chunk
    self-attention sees the same context as in the offline-batched version.

    Requires ``batch_size == 1``.
    """
    enc = model.encoder
    chunk_size = enc.chunk_size
    history = enc.chunk_history_size  # encoder frames
    lookahead = enc.chunk_lookahead_size
    n_overlap = enc.chunk_num_overlaps
    ds_internal = enc._input_downsample_factor  # noqa: SLF001

    if chunk_size is None:
        raise ValueError("Streaming requires chunk_size != None (offline mode not supported).")

    chunk_stride = chunk_size // n_overlap
    audio_per_enc_frame = ds_internal * _MEL_HOP_SAMPLES
    chunk_stride_samples = chunk_stride * audio_per_enc_frame
    history_samples = history * audio_per_enc_frame
    lookahead_samples = lookahead * audio_per_enc_frame

    # Round segment to a clean integer multiple of chunk_stride for clean alignment.
    segment_chunks = max(1, round(segment_seconds * _SAMPLE_RATE / chunk_stride_samples))
    segment_samples = segment_chunks * chunk_stride_samples

    # Identify dims + batch axis from the input Tensor.
    dim_order = data.dims
    if data_spatial_dim not in dim_order:
        raise ValueError(f"data_spatial_dim {data_spatial_dim} not in data.dims {dim_order}")
    batch_dims = [d for d in dim_order if d != data_spatial_dim and not d.is_static()]
    if len(batch_dims) != 1:
        raise ValueError(f"Streaming expects exactly one batch dim, got {batch_dims}")
    batch_dim = batch_dims[0]
    batch_axis = dim_order.index(batch_dim)
    time_axis = dim_order.index(data_spatial_dim)

    audio_torch = data.raw_tensor
    if batch_axis == 1 and time_axis == 0:
        audio_torch = audio_torch.transpose(0, 1).contiguous()
    batch_size = audio_torch.shape[0]
    if batch_size != 1:
        raise NotImplementedError(
            f"Streaming impl requires batch_size=1, got {batch_size}. Set 'max_seqs': 1 in the recog/forward config."
        )

    # Per-sequence audio length (padding-aware).
    if data_spatial_dim.dyn_size_ext is not None:
        audio_len = int(data_spatial_dim.dyn_size_ext.raw_tensor.reshape(-1)[0].item())
    else:
        audio_len = int(audio_torch.shape[1])

    # The chunked encoder uses `ceil` windowing with right-padding, so the total
    # offline enc-frame count is `ceil(audio_len / chunk_stride_samples) * chunk_stride`.
    # Each streaming segment also rounds up at its own boundary, producing a few
    # extra enc frames per segment. Without truncation, the concatenated streaming
    # output is longer than offline AND the extra frames at each segment's end
    # shift everything in subsequent segments. We therefore truncate each segment's
    # "central" portion to its intended length (full segment, except the last).
    offline_enc_total = math.ceil(audio_len / chunk_stride_samples) * chunk_stride
    target_per_seg_full = chunk_stride * segment_chunks
    frames_taken = 0

    all_central_log_probs: list[torch.Tensor] = []
    num_segments = max(1, math.ceil(audio_len / segment_samples))

    for seg_idx in range(num_segments):
        seg_start = seg_idx * segment_samples
        seg_end = min(audio_len, (seg_idx + 1) * segment_samples + lookahead_samples)
        win_start = max(0, seg_start - history_samples)
        win_end = seg_end
        if win_end <= win_start:
            continue

        seg_audio = audio_torch[:, win_start:win_end].contiguous()
        seg_len = seg_audio.shape[1]
        seg_dim = Dim(seg_len, name=f"audio_seg_{seg_idx}")
        seg_size = Tensor(
            "seg_size",
            dims=[batch_dim],
            dtype="int32",
            raw_tensor=torch.tensor([seg_len], dtype=torch.int32),
        )
        seg_dim.dyn_size_ext = seg_size
        seg_audio_tensor = Tensor(
            "audio_seg",
            dims=[batch_dim, seg_dim],
            dtype=data.dtype,
            raw_tensor=seg_audio,
        )

        log_probs_t, _enc, enc_spatial_dim = model.encode_and_get_ctc_log_probs(
            seg_audio_tensor, in_spatial_dim=seg_dim
        )
        lp_dim_order = log_probs_t.dims
        lp_batch_axis = lp_dim_order.index(batch_dim)
        lp_time_axis = lp_dim_order.index(enc_spatial_dim)
        lp_feat_axis = lp_dim_order.index(log_probs_t.feature_dim)
        perm = [lp_batch_axis, lp_time_axis, lp_feat_axis]
        seg_log_probs = log_probs_t.raw_tensor.permute(*perm).contiguous()

        hist_audio = seg_start - win_start  # samples of prepended history actually included
        hist_enc_frames = hist_audio // audio_per_enc_frame
        target = min(target_per_seg_full, offline_enc_total - frames_taken)
        if target <= 0:
            break
        central = seg_log_probs[:, hist_enc_frames : hist_enc_frames + target, :]
        if central.shape[1] < target:
            raise RuntimeError(f"Segment {seg_idx} produced {central.shape[1]} central frames, expected {target}")
        all_central_log_probs.append(central)
        frames_taken += target
        if frames_taken >= offline_enc_total:
            break

    if not all_central_log_probs:
        raise RuntimeError("Streaming produced no segments — empty audio?")

    full_log_probs_torch = torch.cat(all_central_log_probs, dim=1)
    T_enc = full_log_probs_torch.shape[1]

    enc_spatial_dim_full = Dim(T_enc, name="enc_streaming")
    enc_size = Tensor(
        "enc_streaming_size",
        dims=[batch_dim],
        dtype="int32",
        raw_tensor=torch.tensor([T_enc], dtype=torch.int32),
    )
    enc_spatial_dim_full.dyn_size_ext = enc_size
    label_log_prob = Tensor(
        "label_log_prob",
        dims=[batch_dim, enc_spatial_dim_full, model.wb_target_dim],
        dtype=str(full_log_probs_torch.dtype).split(".")[-1],
        raw_tensor=full_log_probs_torch,
    )
    label_log_prob.feature_dim = model.wb_target_dim
    return label_log_prob, enc_spatial_dim_full


def model_recog_ctc_streaming(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """RecogDef. Streaming CTC recog over fixed-duration audio segments.

    Returns ``(seq_targets, seq_log_prob, out_spatial_dim, beam_dim)``
    -- same shape as :func:`exp2024_04_23_baselines.ctc.model_recog`.
    """
    from returnn.config import get_global_config

    config = get_global_config()
    segment_seconds = config.float("streaming_segment_seconds", 10.0)
    beam_size = config.int("beam_size", 12)

    label_log_prob, enc_spatial_dim_full = streaming_encode_and_get_ctc_log_probs(
        model, data, data_spatial_dim, segment_seconds=segment_seconds
    )

    # ---- CTC beam search (copy of ctc.py:model_recog body) ----
    batch_dims_for_search = label_log_prob.remaining_dims((enc_spatial_dim_full, label_log_prob.feature_dim))

    beam_dim = Dim(1, name="initial-beam")
    batch_dims_ = [beam_dim] + batch_dims_for_search
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    label_log_prob = rf.where(
        enc_spatial_dim_full.get_mask(),
        label_log_prob,
        rf.sparse_to_dense(model.blank_idx, axis=model.wb_target_dim, label_value=0.0, other_value=-1.0e30),
    )
    label_log_prob_pre_filter, (backrefs_pre_filter,), pre_filter_beam_dim = rf.top_k(
        label_log_prob,
        k_dim=Dim(min(beam_size, model.wb_target_dim.dimension), name="pre-filter-beam"),
        axis=[model.wb_target_dim],
    )
    label_log_prob_pre_filter_ta = TensorArray.unstack(label_log_prob_pre_filter, axis=enc_spatial_dim_full)
    backrefs_pre_filter_ta = TensorArray.unstack(backrefs_pre_filter, axis=enc_spatial_dim_full)

    max_seq_len = int(enc_spatial_dim_full.get_dim_value())
    seq_targets = []
    seq_backrefs = []
    for t in range(max_seq_len):
        seq_log_prob = seq_log_prob + label_log_prob_pre_filter_ta[t]
        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob,
            k_dim=Dim(min(beam_size, beam_dim.dimension * pre_filter_beam_dim.dimension), name=f"dec-step{t}-beam"),
            axis=[beam_dim, pre_filter_beam_dim],
        )
        target = rf.gather(backrefs_pre_filter_ta[t], indices=target)
        seq_targets.append(target)
        seq_backrefs.append(backrefs)

    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)

    seq_targets__ = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        seq_targets__ = seq_targets__.push_back(target)
    out_spatial_dim = enc_spatial_dim_full
    seq_targets = seq_targets__.stack(axis=out_spatial_dim)

    return seq_targets, seq_log_prob, out_spatial_dim, beam_dim


# RecogDef API
model_recog_ctc_streaming.output_with_beam = True  # type: ignore[attr-defined]
model_recog_ctc_streaming.output_blank_label = "<blank>"  # type: ignore[attr-defined]
model_recog_ctc_streaming.batch_size_dependent = False  # type: ignore[attr-defined]
