"""Streaming encoder via KV cache, implemented as a context-managed monkey-patch
on ``_mem_chunks``.

Each segment forward runs the encoder normally. The patched ``_mem_chunks``:
1. Assigns a stable ``site_id`` to each call (counter; encoder forward is
   deterministic, so the same call site gets the same id across segments).
2. Saves the last ``mem_size`` chunks of the *accumulated* sequence
   (prev cache + current source) for the next segment's cache, so the cache
   covers the full history window even when individual segments are short.
3. If a cache entry exists for that ``site_id`` from the previous segment,
   passes it as ``external_left_context`` to the original ``_mem_chunks``.

This avoids modifying every layer / attention / conv block: the only encoder
change needed is the ``external_left_context`` parameter on ``_mem_chunks``
itself. Backward-compatible: when not inside the context, ``_mem_chunks`` is
unmodified.

Limitations of this v1: assumes ``batch_size == 1`` and resolves chunked-time
dim sizes via ``.dimension`` (static) or ``dyn_size_ext`` (single-batch
dynamic). Multi-batch dynamic support is a follow-up.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import Dict, Optional, Tuple

import torch
from returnn.tensor import Dim, Tensor, batch_dim as _batch_dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

from i6_experiments.users.zeyer.nn_rf.encoder import chunked_conformer_v2


_SAMPLE_RATE = 16_000
_MEL_HOP_SAMPLES = 160  # 10ms hop at 16 kHz; encoder downsamples by ds_internal more.

_orig_mem_chunks = chunked_conformer_v2._mem_chunks
_streaming_state = threading.local()


def _chunked_time_size(dim: Dim) -> int:
    """Return the (single-batch) static size of a chunked-time dim as a Python int.

    Works for both static dims and dynamic dims with batch=1 (reads .dyn_size_ext).
    """
    if dim.dimension is not None:
        return int(dim.dimension)
    if dim.dyn_size_ext is not None:
        return int(dim.dyn_size_ext.raw_tensor.reshape(-1)[0].item())
    raise RuntimeError(f"cannot resolve static size of dim {dim}")


def _make_dynamic_dim(name: str, size: int) -> Dim:
    """Make a per-batch-dynamic Dim with size_tensor [size] (batch=1)."""
    d = Dim(None, name=name)
    d.dyn_size_ext = Tensor(
        f"{name}_size",
        dims=[_batch_dim],
        dtype="int32",
        raw_tensor=torch.tensor([size], dtype=torch.int32),
    )
    return d


def _patched_mem_chunks(
    source: Tensor,
    *,
    spatial_dim: Dim,
    chunked_time_dim: Dim,
    mem_size: int,
    end_chunk_size_dim: Dim,
    out_spatial_dim: Optional[Dim] = None,
    external_left_context: Optional[Tensor] = None,
    external_left_context_time_dim: Optional[Dim] = None,
) -> Tuple[Tensor, Dim]:
    """Replacement that auto-feeds cache (in) and captures cache (out)."""
    state = getattr(_streaming_state, "state", None)
    if state is None:
        return _orig_mem_chunks(
            source,
            spatial_dim=spatial_dim,
            chunked_time_dim=chunked_time_dim,
            mem_size=mem_size,
            end_chunk_size_dim=end_chunk_size_dim,
            out_spatial_dim=out_spatial_dim,
            external_left_context=external_left_context,
            external_left_context_time_dim=external_left_context_time_dim,
        )

    site_id = state["counter"]
    state["counter"] += 1

    # Resolve previous-segment cache for this site, if any.
    cache_in: Optional[Tensor] = None
    cache_in_dim: Optional[Dim] = None
    prev = state["in_cache"].get(site_id)
    if prev is not None:
        cache_in, cache_in_dim = prev
        # `end_chunk_size_dim` is created fresh per forward call (line ~278 of
        # chunked_conformer_v2.py), so the cache's chunk_stride dim differs in
        # identity from the current call's. Rebind to current to make
        # rf.concat / shapes line up without allow_broadcast hacks.
        for d in list(cache_in.dims):
            if (
                d is not end_chunk_size_dim
                and d.dimension == end_chunk_size_dim.dimension
                and d.name == end_chunk_size_dim.name
            ):
                cache_in = rf.replace_dim_v2(cache_in, in_dim=d, out_dim=end_chunk_size_dim)
                break

    # Build cache_out: last `mem_size` chunks of (cache_in ++ current source_sliced),
    # optionally trimming the tail of the current source (chunks whose lookahead
    # was zero-padded because they sit at the segment boundary).
    source_sliced, _ = rf.slice(source, axis=spatial_dim, size=end_chunk_size_dim)
    cur_chunks = _chunked_time_size(chunked_time_dim)
    trim_tail = int(state.get("trim_cache_tail", 0))
    if trim_tail > 0 and trim_tail < cur_chunks:
        kept_dim = _make_dynamic_dim(f"src_kept_site{site_id}", cur_chunks - trim_tail)
        source_for_cache, _ = rf.slice(
            source_sliced, axis=chunked_time_dim, size=cur_chunks - trim_tail, out_dim=kept_dim
        )
        source_for_cache_time_dim = kept_dim
        source_for_cache_chunks = cur_chunks - trim_tail
    elif trim_tail >= cur_chunks:
        # Drop the whole current source; cache_out = cache_in unchanged.
        source_for_cache = None
        source_for_cache_time_dim = None
        source_for_cache_chunks = 0
    else:
        source_for_cache = source_sliced
        source_for_cache_time_dim = chunked_time_dim
        source_for_cache_chunks = cur_chunks

    if cache_in is not None and source_for_cache is not None:
        combined_time_dim = cache_in_dim + source_for_cache_time_dim
        combined, _ = rf.concat(
            (cache_in, cache_in_dim),
            (source_for_cache, source_for_cache_time_dim),
            out_dim=combined_time_dim,
        )
        total_chunks = _chunked_time_size(cache_in_dim) + source_for_cache_chunks
    elif cache_in is not None:
        combined = cache_in
        combined_time_dim = cache_in_dim
        total_chunks = _chunked_time_size(cache_in_dim)
    elif source_for_cache is not None:
        combined = source_for_cache
        combined_time_dim = source_for_cache_time_dim
        total_chunks = source_for_cache_chunks
    else:
        combined = None
        combined_time_dim = None
        total_chunks = 0

    if combined is not None and total_chunks > 0:
        take_n = min(mem_size, total_chunks)
        cache_out_dim = _make_dynamic_dim(f"cache_site{site_id}", take_n)
        cache_out, _ = rf.slice(
            combined,
            axis=combined_time_dim,
            start=total_chunks - take_n,
            out_dim=cache_out_dim,
        )
        state["out_cache"][site_id] = (cache_out, cache_out_dim)

    # Run original _mem_chunks with prev cache as left context.
    return _orig_mem_chunks(
        source,
        spatial_dim=spatial_dim,
        chunked_time_dim=chunked_time_dim,
        mem_size=mem_size,
        end_chunk_size_dim=end_chunk_size_dim,
        out_spatial_dim=out_spatial_dim,
        external_left_context=cache_in,
        external_left_context_time_dim=cache_in_dim,
    )


def streaming_encode_and_get_ctc_log_probs_v2(
    model,
    data: Tensor,
    data_spatial_dim: Dim,
    *,
    segment_seconds: float = 10.0,
) -> Tuple[Tensor, Dim]:
    """KV-cache streaming variant of ``model.encode_and_get_ctc_log_probs``.

    Loops over audio segments of approximately ``segment_seconds``. Between
    segments the encoder KV cache is threaded via :func:`streaming_cache` so
    history is NOT re-encoded. To match offline at chunk-lookahead boundaries,
    each segment overlaps the previous by ``lookahead`` chunks (re-process the
    previous tail now that its future is available) and the cache excludes the
    last ``2 * lookahead`` chunks of the current source (zero-padded tail +
    chunks to be re-done by next segment, which would otherwise double-count).

    Requires ``batch_size == 1``.
    """
    enc = model.encoder
    chunk_size = enc.chunk_size
    lookahead = enc.chunk_lookahead_size
    n_overlap = enc.chunk_num_overlaps
    ds_internal = enc._input_downsample_factor  # noqa: SLF001

    if chunk_size is None:
        raise ValueError("Streaming requires chunk_size != None (offline mode not supported).")

    chunk_stride = chunk_size // n_overlap
    audio_per_enc_frame = ds_internal * _MEL_HOP_SAMPLES
    chunk_stride_samples = chunk_stride * audio_per_enc_frame

    segment_chunks = max(1, round(segment_seconds * _SAMPLE_RATE / chunk_stride_samples))

    # Identify dims + batch axis (mirrors v1).
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
            f"Streaming impl requires batch_size=1, got {batch_size}. Set 'max_seqs': 1."
        )

    if data_spatial_dim.dyn_size_ext is not None:
        audio_len = int(data_spatial_dim.dyn_size_ext.raw_tensor.reshape(-1)[0].item())
    else:
        audio_len = int(audio_torch.shape[1])

    audio_total_chunks_ceil = math.ceil(audio_len / chunk_stride_samples)

    in_cache: Dict[int, Tuple[Tensor, Dim]] = {}
    outs: list[torch.Tensor] = []
    seg_idx = 0
    next_emit_chunk = 0  # global encoder-chunk index of next-to-emit chunk
    while next_emit_chunk < audio_total_chunks_ceil:
        redo = 0 if seg_idx == 0 else lookahead
        seg_start_chunk = next_emit_chunk - redo
        seg_end_chunk_target = next_emit_chunk + segment_chunks + lookahead
        seg_start_samples = seg_start_chunk * chunk_stride_samples
        seg_end_samples = min(audio_len, seg_end_chunk_target * chunk_stride_samples)
        if seg_end_samples <= seg_start_samples:
            break
        is_last = seg_end_samples >= audio_len
        seg_audio = audio_torch[:, seg_start_samples:seg_end_samples].contiguous()
        seg_len = seg_audio.shape[1]
        seg_dim = Dim(seg_len, name=f"audio_seg_{seg_idx}")
        seg_dim.dyn_size_ext = Tensor(
            "seg_size",
            dims=[batch_dim],
            dtype="int32",
            raw_tensor=torch.tensor([seg_len], dtype=torch.int32),
        )
        seg_audio_tensor = Tensor(
            "audio_seg", dims=[batch_dim, seg_dim], dtype=data.dtype, raw_tensor=seg_audio,
        )

        trim_tail = 0 if is_last else 2 * lookahead
        with streaming_cache(in_cache, trim_cache_tail=trim_tail) as out_cache:
            log_probs_t, _enc, enc_spatial_dim = model.encode_and_get_ctc_log_probs(
                seg_audio_tensor, in_spatial_dim=seg_dim
            )
        in_cache = out_cache

        lp_order = log_probs_t.dims
        lp_batch_axis = lp_order.index(batch_dim)
        lp_time_axis = lp_order.index(enc_spatial_dim)
        lp_feat_axis = lp_order.index(log_probs_t.feature_dim)
        seg_lp = log_probs_t.raw_tensor.permute(lp_batch_axis, lp_time_axis, lp_feat_axis).contiguous()
        seg_chunks_produced = seg_lp.shape[1] // chunk_stride

        first_chunk = redo
        last_chunk = seg_chunks_produced if is_last else seg_chunks_produced - lookahead
        if last_chunk <= first_chunk:
            break
        emit = seg_lp[:, first_chunk * chunk_stride : last_chunk * chunk_stride, :]
        outs.append(emit)
        next_emit_chunk += last_chunk - first_chunk
        seg_idx += 1
        if is_last:
            break

    if not outs:
        raise RuntimeError("Streaming produced no segments -- empty audio?")
    full_lp = torch.cat(outs, dim=1)
    T_enc = full_lp.shape[1]
    enc_spatial_dim_full = Dim(T_enc, name="enc_streaming_v2")
    enc_spatial_dim_full.dyn_size_ext = Tensor(
        "enc_streaming_v2_size",
        dims=[batch_dim],
        dtype="int32",
        raw_tensor=torch.tensor([T_enc], dtype=torch.int32),
    )
    label_log_prob = Tensor(
        "label_log_prob",
        dims=[batch_dim, enc_spatial_dim_full, model.wb_target_dim],
        dtype=str(full_lp.dtype).split(".")[-1],
        raw_tensor=full_lp,
    )
    label_log_prob.feature_dim = model.wb_target_dim
    return label_log_prob, enc_spatial_dim_full


@contextmanager
def streaming_cache(
    in_cache: Optional[Dict[int, Tuple[Tensor, Dim]]] = None,
    *,
    trim_cache_tail: int = 0,
):
    """Context manager: while active, ``_mem_chunks`` auto-threads streaming cache.

    :param in_cache: cache from the previous segment (yielded by a previous
        ``streaming_cache`` block), or ``None`` for the very first segment.
    :param trim_cache_tail: drop the last ``trim_cache_tail`` chunks of the
        *current* source when building ``out_cache``. Use this to exclude
        boundary chunks whose lookahead was zero-padded -- they will be
        re-processed (with proper lookahead) in the next segment.
    Yields a fresh ``out_cache`` dict populated during the forward.
    """
    state = {
        "counter": 0,
        "in_cache": in_cache or {},
        "out_cache": {},
        "trim_cache_tail": trim_cache_tail,
    }
    prev_state = getattr(_streaming_state, "state", None)
    _streaming_state.state = state
    chunked_conformer_v2._mem_chunks = _patched_mem_chunks
    try:
        yield state["out_cache"]
    finally:
        chunked_conformer_v2._mem_chunks = _orig_mem_chunks
        _streaming_state.state = prev_state


def model_recog_ctc_streaming_v2(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """RecogDef: streaming CTC recog using KV-cache streaming encoder (v2).

    Returns ``(seq_targets, seq_log_prob, out_spatial_dim, beam_dim)`` -- same
    shape as :func:`exp2024_04_23_baselines.ctc.model_recog`.
    """
    from returnn.config import get_global_config

    config = get_global_config()
    segment_seconds = config.float("streaming_segment_seconds", 10.0)
    beam_size = config.int("beam_size", 12)

    label_log_prob, enc_spatial_dim_full = streaming_encode_and_get_ctc_log_probs_v2(
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


# RecogDef API attrs
model_recog_ctc_streaming_v2.output_with_beam = True  # type: ignore[attr-defined]
model_recog_ctc_streaming_v2.output_blank_label = "<blank>"  # type: ignore[attr-defined]
model_recog_ctc_streaming_v2.batch_size_dependent = False  # type: ignore[attr-defined]
