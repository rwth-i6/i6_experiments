"""
Beam-search recog for the slow-fast-rna streaming decoders.
Both searches reuse the top_k + backref backtracking of ``exp2024_04_23_baselines.aed.model_recog``:

- :func:`frame_sync_beam_search`: ``T (+flush)`` steps, one label-or-blank per frame (RNA), blanks stripped.
  Max/sum recomb over the emitted-label seq (RNA: append every non-blank, repeats kept, no CTC target!=prev collapse);
  the ``_same_seq_labels`` + keep-argmax block of ``recog_ext.aed_ctc.model_recog_with_recomb``,
  whose CTC/AED/LM score fusion is dropped here.
- :func:`label_sync_beam_search`: AED, per-hyp eos termination, optional length norm.
  No recomb (each beam entry is already a distinct label seq).

``step`` supplies the per-decoder logits; ``init_state`` the initial state tree (gathered by backrefs each step).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Tuple
import functools
import tree

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray


def _gather_backrefs(s: Any, *, backrefs: Tensor) -> Any:
    if isinstance(s, Tensor):
        if backrefs.sparse_dim in s.dims:
            return rf.gather(s, indices=backrefs)
        return s  # beam-independent
    if isinstance(s, Dim):
        assert s.dimension or backrefs not in s.dyn_size_ext.dims
        return s
    raise TypeError(f"_gather_backrefs: unexpected type ({type(s)})")


def frame_sync_beam_search(
    *,
    batch_dims: list,
    target_dim_ext: Dim,
    bos_idx: int,
    blank_idx: int,
    enc: Tensor,
    enc_spatial_dim: Dim,
    beam_size: int,
    init_state: Callable[[list], Any],
    step: Callable[[Tensor, Tensor, Any], Tuple[Tensor, Any]],
    num_flush_frames: int = 0,
    recomb: Optional[str] = "max",
    return_alignment: bool = False,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous beam search (RNA), blanks stripped.

    :param step: ``(prev_sym, enc_t, state) -> (log_probs over target_dim_ext, new_state)``;
        prev_sym is the previous frame's choice (bos at t=0), enc_t is zeroed during the flush tail.
    :param init_state: builds the state tree over ``[beam(1)] + batch_dims``.
    :param num_flush_frames: extra emit steps for the delayed-tail labels (framewise delay_frames).
    :param recomb: "max" (Viterbi), "sum" (marginalize), or None.
    :return: (seq_targets {beam,batch,out_spatial} over target_dim_ext, seq_log_prob {beam,batch},
        out_spatial_dim, beam_dim)
    """
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [batch]
    t_total = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor) + num_flush_frames
    neg_inf = float("-inf")

    beam_dim = Dim(1, name="initial-beam")
    bd = [beam_dim] + batch_dims
    state = init_state(bd)
    prev = rf.constant(bos_idx, dims=bd, sparse_dim=target_dim_ext, dtype="int32")
    seq_log_prob = rf.constant(0.0, dims=bd)
    seq_label = _seq_label_history_init_state(vocab_dim=target_dim_ext, batch_dims=bd)
    force_blank = rf.sparse_to_dense(blank_idx, axis=target_dim_ext, label_value=0.0, other_value=neg_inf)

    seq_targets, seq_backrefs = [], []
    for t in range(t_total):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")
        audio_valid = t_t < enc_lens
        emit_valid = t_t < enc_lens + num_flush_frames
        idx = rf.where(audio_valid, t_t, enc_lens - 1)
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)
        enc_t = rf.where(audio_valid, enc_t, 0.0)  # silence during flush + padding

        log_probs, state = step(prev, enc_t, state)  # [beam, batch, vocab]
        log_probs = rf.where(emit_valid, log_probs, force_blank)  # past the emit window: forced blank, zero cost
        seq_log_prob = seq_log_prob + log_probs

        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"frame{t}-beam"), axis=[beam_dim, target_dim_ext]
        )  # each [beam, batch]; backrefs -> old beam
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        state = rf.nested.gather_nested(state, indices=backrefs)
        seq_label = rf.nested.gather_nested(seq_label, indices=backrefs)
        prev = target

        if recomb:
            got_new_label = target != blank_idx  # RNA: every non-blank is appended, repeats kept
            got_new_label_cpu = rf.copy_to_device(got_new_label, "cpu")
            if got_new_label_cpu.raw_tensor.sum().item() > 0:
                seq_label = rf.nested.mask_nested(
                    _seq_label_append(seq_label, target),
                    mask=got_new_label,
                    mask_cpu=got_new_label_cpu,
                    mask_value=seq_label,
                )
                if recomb in ("max", "sum"):
                    same_seq_labels, beam_dual_dim = _same_seq_labels(
                        seq_label.history, spatial_dim=seq_label.hist_dim, beam_dim=beam_dim
                    )
                    seq_log_prob_ext = rf.where(  # [batch, beam, beam_dual]
                        same_seq_labels,
                        rf.replace_dim_v2(seq_log_prob, in_dim=beam_dim, out_dim=beam_dual_dim),
                        neg_inf,
                    )
                    if recomb == "sum":
                        seq_log_prob = rf.reduce_logsumexp(seq_log_prob_ext, axis=beam_dual_dim)
                    argmax = rf.reduce_argmax(seq_log_prob_ext, axis=beam_dual_dim)  # -> beam_dual
                    seq_log_prob = rf.where(argmax == rf.range_over_dim(beam_dim), seq_log_prob, neg_inf)
                else:
                    raise ValueError(f"invalid recomb {recomb!r}")

    out_spatial_dim = Dim(t_total, name="out-spatial")
    aligned = _backtrack(seq_targets, seq_backrefs, beam_dim, out_spatial_dim)  # [beam, batch, t_total]
    aligned.sparse_dim = target_dim_ext
    if return_alignment:  # blanks-kept alignment, truncated to the per-seq emit window (dynamic dim)
        in_window = rf.range_over_dim(out_spatial_dim) < (enc_lens + num_flush_frames)
        align_out, align_spatial_dim = rf.masked_select(aligned, mask=in_window, dims=[out_spatial_dim])
        align_out.sparse_dim = target_dim_ext
        return align_out, seq_log_prob, align_spatial_dim, beam_dim
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(
        aligned, mask=aligned != blank_idx, dims=[out_spatial_dim]
    )
    seq_targets_out.sparse_dim = target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


def label_sync_beam_search(
    *,
    batch_dims: list,
    target_dim: Dim,
    bos_idx: int,
    eos_idx: int,
    beam_size: int,
    length_normalization_exponent: float,
    max_seq_len: Tensor,
    init_state: Callable[[list], Any],
    step: Callable[[Tensor, Any], Tuple[Tensor, Any]],
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Label-synchronous (AED) beam search; generalizes ``exp2024_04_23_baselines.aed.model_recog``
    with a caller-supplied step. Terminates each hyp on eos_idx, trimmed via the per-hyp output length.

    :param step: ``(prev_target, state) -> (logits over target_dim, new_state)``.
    :param length_normalization_exponent: 0 off, 1 full ``1/len``.
    :param max_seq_len: per-seq step cap ``[batch]``.
    :return: (seq_targets {beam,batch,out_spatial} over target_dim, seq_log_prob {beam,batch},
        out_spatial_dim, beam_dim)
    """
    beam_dim = Dim(1, name="initial-beam")
    bd = [beam_dim] + batch_dims
    state = init_state(bd)
    target = rf.constant(bos_idx, dims=bd, sparse_dim=target_dim, dtype="int32")
    ended = rf.constant(False, dims=bd)
    out_seq_len = rf.constant(0, dims=bd)
    seq_log_prob = rf.constant(0.0, dims=bd)
    ended_filter = rf.sparse_to_dense(eos_idx, axis=target_dim, label_value=0.0, other_value=-1.0e30)

    i = 0
    seq_targets, seq_backrefs = [], []
    while True:
        logits, state = step(target, state)
        label_log_prob = rf.log_softmax(logits, axis=target_dim)
        label_log_prob = rf.where(ended, ended_filter, label_log_prob)  # finished beams only continue with eos
        seq_log_prob = seq_log_prob + label_log_prob

        seq_log_prob, (backrefs, target), beam_dim = rf.top_k(
            seq_log_prob, k_dim=Dim(beam_size, name=f"dec-step{i}-beam"), axis=[beam_dim, target_dim]
        )
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        state = tree.map_structure(functools.partial(_gather_backrefs, backrefs=backrefs), state)
        ended = rf.gather(ended, indices=backrefs)
        out_seq_len = rf.gather(out_seq_len, indices=backrefs)
        i += 1

        ended = rf.logical_or(ended, target == eos_idx)
        ended = rf.logical_or(ended, rf.copy_to_device(i >= max_seq_len))
        if bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break
        out_seq_len = out_seq_len + rf.where(ended, 0, 1)

        if i > 1 and length_normalization_exponent != 0:
            # ending seq: score_i = score_{i-1} * (i/(i-1))^exp (eos shifts length by one)
            seq_log_prob *= rf.where(ended, (i / (i - 1)) ** length_normalization_exponent, 1.0)

    if i > 0 and length_normalization_exponent != 0:
        seq_log_prob *= (1 / i) ** length_normalization_exponent

    out_spatial_dim = Dim(out_seq_len, name="out-spatial")
    seq_targets_out = _backtrack(seq_targets, seq_backrefs, beam_dim, out_spatial_dim)
    seq_targets_out.sparse_dim = target_dim
    return seq_targets_out, seq_log_prob, out_spatial_dim, beam_dim


def _backtrack(seq_targets: list, seq_backrefs: list, beam_dim: Dim, out_spatial_dim: Dim) -> Tensor:
    """Resolve the per-step (target, backrefs) into ``[beam, batch, out_spatial]``."""
    seq_targets_ = []
    indices = rf.range_over_dim(beam_dim)  # final-beam -> final-beam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        seq_targets_.insert(0, rf.gather(target, indices=indices))
        indices = rf.gather(backrefs, indices=indices)  # final-beam -> prev-beam
    ta = TensorArray(seq_targets_[0])
    for target in seq_targets_:
        ta = ta.push_back(target)
    return ta.stack(axis=out_spatial_dim)


# emitted-label-sequence history + recombination, mirroring ...recog_ext.aed_ctc.


def _seq_label_history_init_state(*, vocab_dim: Dim, batch_dims: Sequence[Dim]) -> rf.State:
    hist_dim = Dim(0, name="hist0")
    history = rf.zeros(list(batch_dims) + [hist_dim], dtype="int64", sparse_dim=vocab_dim)
    return rf.State(hist_dim=hist_dim, history=history)


def _seq_label_append(state: rf.State, new_label: Tensor) -> rf.State:
    new_history, new_hist_dim = rf.cum_concat_step(new_label, prev_accum=state.history, axis=state.hist_dim)
    return rf.State(hist_dim=new_hist_dim, history=new_history)


def _same_seq_labels(seq: Tensor, *, spatial_dim: Dim, beam_dim: Dim) -> Tuple[Tensor, Dim]:
    """Pairwise ``[batch, beam, beam_dual]``: which beam pairs share the same label history."""
    seq_label_dual, beam_dual_dim = rf.replace_dim(seq, in_dim=beam_dim)
    same_seq_labels = rf.compare_bc(seq, "==", seq_label_dual)
    same_seq_labels = rf.reduce_all(same_seq_labels, axis=spatial_dim)
    if beam_dim in spatial_dim.get_size_tensor().dims:  # per-beam variable length -> also compare lengths
        seq_labels_lens = spatial_dim.get_size_tensor(device=same_seq_labels.device)
        seq_labels_dual_lens = rf.replace_dim_v2(seq_labels_lens, in_dim=beam_dim, out_dim=beam_dual_dim)
        same_seq_labels = rf.logical_and(same_seq_labels, rf.compare_bc(seq_labels_lens, "==", seq_labels_dual_lens))
    return same_seq_labels, beam_dual_dim
