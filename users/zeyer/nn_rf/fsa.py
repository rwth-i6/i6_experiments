"""
FSAs, forward score, best path
"""

from __future__ import annotations
from typing import Optional, Union, Sequence, Tuple, List, NamedTuple, Dict
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


class FSA(NamedTuple):
    """
    Finite state automaton (FSA)
    (potentially batched (ragged))
    """

    batch_dims: Sequence[Dim]  # those dims [B...] will be merged into a single B_
    label_dim: Dim  # L

    # States
    num_states_dim_ext: Dim  # S_[B] (per batch)
    num_states_dim: Dim  # S (flat/packed)
    start_states: Tensor  # [B...] -> S
    states_batch_idx: Tensor  # [S] -> B_

    # Transitions
    num_trans_dim: Dim  # A
    trans_prev_state: Tensor  # [A] -> S
    trans_next_state: Tensor  # [A] -> S
    trans_batch_label_idx: Tensor  # [A] -> (B...*L) (single merged dim)

    # Final states
    num_final_dim: Dim  # S_F
    final_states: Tensor  # [S_F]->S
    final_states_batch_idx: Tensor  # [S_F]->B_


def forward_score(*, logits: Tensor, logits_normalized: bool = False, input_spatial_dim: Dim, fsa: FSA):
    """
    Forward score using the forward algorithm,
    via dynamic programming.

    This is differentiable,
    and calculating the gradients will result implicitly in the forward-backward algorithm.

    See also :func:`best_path`.
    """
    assert logits.feature_dim == fsa.label_dim

    if not logits_normalized:
        logits = rf.log_softmax(logits, axis=logits.feature_dim)
    batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))

    # The following condition is maybe not strictly needed, and other broadcasting might work as well,
    # but just assume now for simplicity/sanity.
    assert set(batch_dims) == set(fsa.batch_dims)

    # This order of dims should be more efficient for the loop below.
    logits = logits.copy_transpose([input_spatial_dim] + batch_dims + [logits.feature_dim])

    # Merge dims to allow a single gather access for fsa_transitions_batch_label_idx.
    logits, batch_label_dim = rf.merge_dims(logits, dims=batch_dims + [logits.feature_dim])  # [T,B*L]

    device = logits.device
    seq_lens_ = rf.gather(input_spatial_dim.get_size_tensor(device=device), indices=fsa.states_batch_idx)  # [S] -> T

    scores = rf.scatter(
        rf.zeros(fsa.batch_dims, dtype=logits.dtype, device=device),
        indices=fsa.start_states,
        indices_dim=fsa.batch_dims,
        fill_value=float("-inf"),
    )  # [S], per state

    for t in range(input_spatial_dim.get_dim_value()):
        scores_in = rf.gather(scores, indices=fsa.trans_prev_state)  # [A]
        assert scores_in.dims == (fsa.num_trans_dim,)
        logits_t = rf.gather(logits, indices=t, axis=input_spatial_dim)  # [B*L]
        scores_in = _safe_add(scores_in, rf.gather(logits_t, indices=fsa.trans_batch_label_idx))  # [A]

        scores_, _ = _scatter_safe_logsumexp(
            scores_in,
            indices=fsa.trans_next_state,
            indices_dim=fsa.num_trans_dim,
            out_dim=fsa.num_states_dim,
        )  # [S]

        scores = rf.where(t < seq_lens_, scores_, scores)  # [S]

    final_scores = rf.gather(scores, indices=fsa.final_states)  # [S_F]
    final_scores_, _ = _scatter_safe_logsumexp(
        final_scores,
        indices=fsa.final_states_batch_idx,
        indices_dim=fsa.num_final_dim,
    )  # [B]
    # assert not final_scores_.isinf().all(), "no path to final state"

    return final_scores_


def ctc_loss(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    input_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
    blank_index: int,
) -> Tensor:
    """
    CTC loss

    :return: loss, shape [B]
    """
    fsa = fsa_for_ctc(targets, targets_spatial_dim, blank_index=blank_index, labels_with_blank_dim=logits.feature_dim)
    return -forward_score(
        logits=logits, logits_normalized=logits_normalized, input_spatial_dim=input_spatial_dim, fsa=fsa
    )


def forward_partial_accum_scores(*, logits: Tensor, logits_normalized: bool = False, input_spatial_dim: Dim, fsa: FSA):
    """
    Forward accum partial scores: log p(s_1,...,s_n | x_1,...,x_T)
    (for states s 1 to n).

    :return partial scores, shape [B,S_]
    """
    assert logits.feature_dim == fsa.label_dim

    if not logits_normalized:
        logits = rf.log_softmax(logits, axis=logits.feature_dim)
    batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))

    # The following condition is maybe not strictly needed, and other broadcasting might work as well,
    # but just assume now for simplicity/sanity.
    assert set(batch_dims) == set(fsa.batch_dims)

    # This order of dims should be more efficient for the loop below.
    logits = logits.copy_transpose([input_spatial_dim] + batch_dims + [logits.feature_dim])

    # Merge dims to allow a single gather access for fsa_transitions_batch_label_idx.
    logits, batch_label_dim = rf.merge_dims(logits, dims=batch_dims + [logits.feature_dim])  # [T,B*L]

    device = logits.device
    seq_lens_ = rf.gather(input_spatial_dim.get_size_tensor(device=device), indices=fsa.states_batch_idx)  # [S] -> T

    state_idx_offsets = rf.masked_scatter(
        rf.range_over_dim(fsa.num_states_dim, device=device),
        in_dim=fsa.num_states_dim,
        mask=rf.sequence_mask(batch_dims + [fsa.num_states_dim_ext], device=device),
        dims=batch_dims + [fsa.num_states_dim_ext],
    )  # [B,S_]->S

    scores = rf.scatter(
        rf.zeros(fsa.batch_dims, dtype=logits.dtype, device=device),
        indices=fsa.start_states,
        indices_dim=fsa.batch_dims,
        fill_value=float("-inf"),
    )  # [S], per state

    partial_scores = []  # T->[B,S_]

    for t in range(input_spatial_dim.get_dim_value()):
        scores_in = rf.gather(scores, indices=fsa.trans_prev_state)  # [A]
        assert scores_in.dims == (fsa.num_trans_dim,)
        logits_t = rf.gather(logits, indices=t, axis=input_spatial_dim)  # [B*L]
        scores_in = _safe_add(scores_in, rf.gather(logits_t, indices=fsa.trans_batch_label_idx))  # [A]

        scores_, _ = _scatter_safe_logsumexp(
            scores_in,
            indices=fsa.trans_next_state,
            indices_dim=fsa.num_trans_dim,
            out_dim=fsa.num_states_dim,
        )  # [S]

        scores = rf.where(t < seq_lens_, scores_, scores)  # [S]

        partial_scores.append(rf.gather(scores, indices=state_idx_offsets))  # [B,S_]

    partial_scores_, _ = rf.stack(partial_scores, out_dim=input_spatial_dim)  # [T,B,S_]

    partial_scores_ = rf.reduce_logsumexp(partial_scores_, axis=input_spatial_dim)  # [B,S_]
    return partial_scores_


def ctc_partial_scores(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    input_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
    blank_index: int,
    include_next_blank: Union[bool, str] = False,
) -> Tensor:
    """
    Forward partial scores: log p(y_n | y_1,...,y_{n-1}, x_1,...,x_T).
    """
    fsa = fsa_for_ctc(targets, targets_spatial_dim, blank_index=blank_index, labels_with_blank_dim=logits.feature_dim)
    partial_accum_scores = forward_partial_accum_scores(
        logits=logits, logits_normalized=logits_normalized, input_spatial_dim=input_spatial_dim, fsa=fsa
    )  # [B,S_]->score

    device = logits.device

    label_states = rf.range_over_dim(targets_spatial_dim, device=device) * 2 + 1  # [T_out]->S_
    label_states.sparse_dim = fsa.num_states_dim_ext
    label_states_prev_blank = label_states - 1  # [T_out]->S_
    label_states_prev_label = label_states - 2  # [T_out]->S_. but might be invalid!
    label_states_next_blank = label_states + 1  # [T_out]->S_
    label_partial_scores = rf.gather(partial_accum_scores, indices=label_states)  # [B,T_out]->score
    label_prev_blank_partial_scores = rf.gather(
        partial_accum_scores, indices=label_states_prev_blank
    )  # [B,T_out]->score
    label_prev_label_partial_scores = rf.gather(
        partial_accum_scores, indices=label_states_prev_label, clip_to_valid=True
    )  # [B,T_out]->score
    label_prev_label_partial_scores = rf.where(
        label_states_prev_label >= 0, label_prev_label_partial_scores, float("-inf")
    )
    label_next_blank_partial_scores = rf.gather(
        partial_accum_scores, indices=label_states_next_blank
    )  # [B,T_out]->score
    label_prev_partial_scores = label_prev_blank_partial_scores
    if not include_next_blank:
        pass
    elif isinstance(include_next_blank, bool) and include_next_blank:
        label_partial_scores = label_next_blank_partial_scores
    elif include_next_blank == "both":
        label_partial_scores = _logaddexp(label_partial_scores, label_next_blank_partial_scores)
    elif include_next_blank == "both_prev":
        label_partial_scores = _logaddexp(label_partial_scores, label_next_blank_partial_scores)
        label_prev_partial_scores = _logaddexp(label_prev_blank_partial_scores, label_prev_label_partial_scores)
    else:
        raise ValueError(f"invalid include_next_blank: {include_next_blank!r}")
    label_partial_scores = label_partial_scores - label_prev_partial_scores  # [B,T_out]->score
    return label_partial_scores


def best_path(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    input_spatial_dim: Dim,
    fsa: FSA,
    return_transition_indices: bool = False,
) -> Tuple[Tensor, Tensor]:
    """
    Forward score using the maximum approximation, i.e. the Viterbi algorithm,
    i.e. finding the best path (with the highest probability),
    via dynamic programming.

    The implementation is generic but should be fairly efficient.
    We go over the time steps and compute the forward score for each state.
    We have a vector over all the ragged states
    and we use :func:`scatter_argmax` for the argmax.

    Our earlier native FastBaumWelchOp/FastViterbi has the same idea
    and is also generic, however it is custom native code,
    while here we have a pure PyTorch implementation.

    The code is very similar to :func:`forward_score`,
    except that we use :func:`torch.max` instead of :func:`safe_logsumexp`.
    Additionally, we need to track the backpointers to reconstruct the best path.

    :param logits: [B...,T,F+1]
    :param logits_normalized: whether the logits are already normalized.
        If False, we apply log_softmax to the logits.
        If True, the logits are used as is.
        (Note that there is no check or really the necessity to have the logits really normalized.)
    :param input_spatial_dim: T
    :param fsa:
    :param return_transition_indices: whether to return the transition indices of the FSA (in A)
        instead of the label indices (in L).
    :return: tuple (best path, score).
        best path are the label indices in L if not return_transition_indices else transition indices in A.
        The score has shape [B...].
    """
    assert logits.feature_dim == fsa.label_dim

    if not logits_normalized:
        logits = rf.log_softmax(logits, axis=logits.feature_dim)
    batch_dims = logits.remaining_dims((input_spatial_dim, logits.feature_dim))

    # The following condition is maybe not strictly needed, and other broadcasting might work as well,
    # but just assume now for simplicity/sanity.
    assert set(batch_dims) == set(fsa.batch_dims)

    # This order of dims should be more efficient for the loop below.
    logits = logits.copy_transpose([input_spatial_dim] + batch_dims + [logits.feature_dim])

    # Merge dims to allow a single gather access for fsa_transitions_batch_label_idx.
    logits, batch_label_dim = rf.merge_dims(logits, dims=batch_dims + [logits.feature_dim])  # [T,B*L]

    device = logits.device
    seq_lens_ = rf.gather(input_spatial_dim.get_size_tensor(device=device), indices=fsa.states_batch_idx)  # [S] -> T

    scores = rf.scatter(
        rf.zeros(fsa.batch_dims, dtype=logits.dtype, device=device),
        indices=fsa.start_states,
        indices_dim=fsa.batch_dims,
        fill_value=float("-inf"),
    )  # [S], per state

    backpointers: List[Tensor] = []  # each is [S]->A

    for t in range(input_spatial_dim.get_dim_value()):
        scores_in = rf.gather(scores, indices=fsa.trans_prev_state)  # [A]
        assert scores_in.dims == (fsa.num_trans_dim,)
        logits_t = rf.gather(logits, indices=t, axis=input_spatial_dim)  # [B*L]
        scores_in = _safe_add(scores_in, rf.gather(logits_t, indices=fsa.trans_batch_label_idx))  # [A]

        idx = rf.scatter_argmax(
            scores_in,
            indices=fsa.trans_next_state,
            indices_dim=fsa.num_trans_dim,
        )  # [S] -> A (index in scores_in)
        backpointers.append(idx)
        scores_ = rf.where(idx >= 0, rf.gather(scores_in, indices=idx, clip_to_valid=True), float("-inf"))  # [S]
        scores = rf.where(t < seq_lens_, scores_, scores)  # [S]

    final_scores = rf.gather(scores, indices=fsa.final_states)  # [S_F]
    final_idx = rf.scatter_argmax(
        final_scores,
        indices=fsa.final_states_batch_idx,
        indices_dim=fsa.num_final_dim,
    )  # [B] -> S_F
    final_scores_ = rf.gather(final_scores, indices=final_idx)  # [B]
    # assert not final_scores_.raw_tensor.isinf().all(), "no path to final state"

    state_idx = rf.gather(fsa.final_states, indices=final_idx)  # [B] -> S
    best_path_ = []
    for t in range(input_spatial_dim.get_dim_value() - 1, -1, -1):
        transition_idx = rf.gather(backpointers[t], indices=state_idx)  # [B] -> A
        best_path_.append(transition_idx)
        state_idx = rf.where(
            t < input_spatial_dim.get_size_tensor(device=device),
            rf.gather(fsa.trans_prev_state, indices=transition_idx),
            state_idx,
        )  # [B] -> S

    # assert (state_idx == fsa_start_states).all()
    best_path_.reverse()
    best_path_, _ = rf.stack(best_path_, out_dim=input_spatial_dim)  # [B,T]->A

    if not return_transition_indices:
        best_path_ = rf.gather(fsa.trans_batch_label_idx, indices=best_path_)  # [B,T] -> B*L
        best_path_ %= fsa.label_dim.get_dim_value_tensor()  # [B,T] -> L
        best_path_.sparse_dim = fsa.label_dim

    return best_path_, final_scores_


def fsa_for_ctc(
    targets: Tensor,
    targets_spatial_dim: Dim,
    *,
    labels_with_blank_dim: Dim,
    blank_index: int,
) -> FSA:
    """
    Generate an FSA for CTC.
    """
    assert targets.sparse_dim
    assert blank_index >= 0
    if blank_index < targets.sparse_dim.dimension:
        # Assume blank idx already part of targets vocab.
        # (Also assume blank is not part of targets.)
        assert labels_with_blank_dim == targets.sparse_dim
    else:
        # Assume blank idx not part of targets vocab but added at the end.
        assert labels_with_blank_dim.dimension == targets.sparse_dim.dimension + 1
        assert blank_index == labels_with_blank_dim.dimension - 1

    batch_dims = targets.remaining_dims(targets_spatial_dim)
    device = targets.device

    # The FSA for the CTC label topology has states [blank,label1,blank,label2,...,blank,labelN,blank],
    # i.e. T_out*2+1 states.
    # The initial state is 0, and the final states are T_out*2-1 and T_out*2.
    # The transition from 0 to 1 is blank, 1 to 2 is label1, etc.
    # State 0 has blank self-loop, 1 has label1 self-loop, etc.
    # There is a skip from 1 to 3, 3 to 5, only when label1!=label2, etc.
    # S: num total states (summed over all batch dims)
    # A: num total transitions (summed over all batch dims)
    num_states_dim_ext = targets_spatial_dim * 2 + 1  # [B]->S_ (S_ being the states per batch)
    states_batch_idx, num_states_dim = rf.pack_padded(
        rf.expand_dim(rf.range_over_merged_dims(batch_dims, device=device), num_states_dim_ext),
        dims=batch_dims + [num_states_dim_ext],
    )  # [S]->B
    num_states_dim.name = "states"

    state_idx_offsets = rf.masked_scatter(
        rf.range_over_dim(num_states_dim, device=device),
        in_dim=num_states_dim,
        mask=rf.sequence_mask(batch_dims + [num_states_dim_ext], device=device),
        dims=batch_dims + [num_states_dim_ext],
    )  # [B,S_]->S
    state_idx_offsets = rf.gather(state_idx_offsets, indices=0, axis=num_states_dim_ext)  # [B]->S

    start_states = state_idx_offsets  # [B]->S. it's just the first state for CTC.

    # Every state has a self-loop, and then we can have a transition to the next state (except of the last).
    # This will also have some invalid transitions (when prev_label==next_label), which we will mask out later.
    trans_ext_dims = {}
    trans_prev_state_ext = {}
    trans_next_state_ext = {}
    trans_label_idx_ext = {}  # noqa

    # So first the self-loops in blanks.
    trans_ext_dims[0] = targets_spatial_dim + 1  # T_out+1
    trans_prev_state_ext[0] = rf.range_over_dim(trans_ext_dims[0], device=device) * 2  # [T_out+1]->S_
    trans_next_state_ext[0] = trans_prev_state_ext[0]  # [T_out+1]->S_
    trans_label_idx_ext[0] = rf.full(
        fill_value=blank_index, dims=[trans_ext_dims[0]], dtype=targets.dtype, device=device
    )  # [T_out+1]->L

    # Now self-loops in labels.
    trans_ext_dims[1] = targets_spatial_dim  # T_out
    trans_prev_state_ext[1] = rf.range_over_dim(trans_ext_dims[1], device=device) * 2 + 1  # [T_out]->S_
    trans_next_state_ext[1] = trans_prev_state_ext[1]  # [T_out]->S_
    trans_label_idx_ext[1] = targets  # [B,T_out]->L

    # Now transitions into blanks.
    trans_ext_dims[2] = targets_spatial_dim  # T_out
    trans_prev_state_ext[2] = rf.range_over_dim(trans_ext_dims[2], device=device) * 2 + 1  # [T_out]->S_
    trans_next_state_ext[2] = trans_prev_state_ext[2] + 1  # [T_out]->S_
    trans_label_idx_ext[2] = rf.full(
        fill_value=blank_index, dims=[trans_ext_dims[2]], dtype=targets.dtype, device=device
    )  # [T_out]->L

    # Now transitions into labels.
    trans_ext_dims[3] = targets_spatial_dim  # T_out
    trans_prev_state_ext[3] = rf.range_over_dim(trans_ext_dims[3], device=device) * 2  # [T_out]->S_
    trans_next_state_ext[3] = trans_prev_state_ext[3] + 1  # [T_out]->S_
    trans_label_idx_ext[3] = targets  # [B,T_out]->L

    # Now possible skip transitions over blank.
    trans_ext4_dim_ = targets_spatial_dim - 1  # T_out-1
    trans_prev_state_ext4_ = rf.range_over_dim(trans_ext4_dim_, device=device) * 2 + 1  # [T_out-1]->S_
    trans_next_state_ext4_ = trans_prev_state_ext4_ + 2  # [T_out-1]->S_
    trans_label_idx_ext4_, _ = rf.slice(
        targets, axis=targets_spatial_dim, start=1, size=trans_ext4_dim_
    )  # [B,T_out-1]->L
    # Filter only the transitions where the label changes.
    skip_trans_mask = (
        trans_label_idx_ext4_ != rf.slice(targets, axis=targets_spatial_dim, start=0, size=trans_ext4_dim_)[0]
    )  # [B,T_out-1]
    trans_prev_state_ext[4], trans_ext_dims[4] = rf.masked_select(
        rf.expand_dims(trans_prev_state_ext4_, batch_dims), mask=skip_trans_mask, dims=[trans_ext4_dim_]
    )  # [B,ext4]->S_
    trans_next_state_ext[4], _ = rf.masked_select(
        rf.expand_dims(trans_next_state_ext4_, batch_dims),
        mask=skip_trans_mask,
        dims=[trans_ext4_dim_],
        out_dim=trans_ext_dims[4],
    )  # [B,ext4]->S_
    trans_label_idx_ext[4], _ = rf.masked_select(
        trans_label_idx_ext4_, mask=skip_trans_mask, dims=[trans_ext4_dim_], out_dim=trans_ext_dims[4]
    )  # [B,ext4]->L

    # We want to index into the packed state indices (S), not the ext state indices (S_).
    trans_prev_state_ext = {i: state + state_idx_offsets for i, state in trans_prev_state_ext.items()}  # [B,ext]->S
    trans_next_state_ext = {i: state + state_idx_offsets for i, state in trans_next_state_ext.items()}  # [B,ext]->S
    for state in list(trans_prev_state_ext.values()) + list(trans_next_state_ext.values()):
        state.sparse_dim = num_states_dim
    # And also, we want to index into the merged batch_label_idx (B*L).
    batch_range = (
        rf.range_over_merged_dims(batch_dims, device=targets.device) * labels_with_blank_dim.get_dim_value_tensor()
    )
    trans_batch_label_idx_ext = {i: batch_range + label for i, label in trans_label_idx_ext.items()}  # [B,ext]->(B*L)
    batch_label_dim = batch_range.sparse_dim * labels_with_blank_dim
    for label in trans_batch_label_idx_ext.values():
        label.sparse_dim = batch_label_dim

    # Now pack all transition parts.
    trans_dims: Dict[int, Dim] = {}  # noqa
    trans_prev_state_: Dict[int, Tensor] = {}  # noqa
    trans_next_state_: Dict[int, Tensor] = {}  # noqa
    trans_batch_label_idx_: Dict[int, Tensor] = {}  # noqa
    for i in trans_prev_state_ext.keys():
        trans_prev_state_[i], trans_dims[i] = rf.pack_padded(
            trans_prev_state_ext[i], dims=batch_dims + [trans_ext_dims[i]]
        )
        trans_next_state_[i], _ = rf.pack_padded(
            trans_next_state_ext[i], dims=batch_dims + [trans_ext_dims[i]], out_dim=trans_dims[i]
        )
        trans_batch_label_idx_[i], _ = rf.pack_padded(
            trans_batch_label_idx_ext[i], dims=batch_dims + [trans_ext_dims[i]], out_dim=trans_dims[i]
        )

    # Now merge all transitions.
    num_trans_dim: Dim = sum(trans_dims.values())  # noqa
    num_trans_dim.name = "transitions"
    trans_prev_state, _ = rf.concat(
        *zip(trans_prev_state_.values(), trans_dims.values()),
        allow_broadcast=True,
        out_dim=num_trans_dim,
    )  # [A]->S
    trans_next_state, _ = rf.concat(
        *zip(trans_next_state_.values(), trans_dims.values()),
        allow_broadcast=True,
        out_dim=num_trans_dim,
    )  # [A]->S
    trans_batch_label_idx, _ = rf.concat(
        *zip(trans_batch_label_idx_.values(), trans_dims.values()),
        allow_broadcast=True,
        out_dim=num_trans_dim,
    )  # [A]->L

    # Final states: T_out*2-1 and T_out*2 for CTC.
    final_states_ext, final_dim_ext = rf.stack(
        [
            targets_spatial_dim.get_size_tensor(device=device) * 2 - 1,
            targets_spatial_dim.get_size_tensor(device=device) * 2,
        ]
    )  # [B,S_F_]->S_.
    final_states_ext += state_idx_offsets  # [B,S_F_]->S
    final_states_ext.sparse_dim = num_states_dim
    final_states_batch_idx_ext = rf.expand_dims(
        rf.range_over_merged_dims(batch_dims, device=device), final_states_ext.remaining_dims(batch_dims)
    )  # [B,S_F_]->B
    final_states, num_final_dim = rf.merge_dims(final_states_ext, dims=batch_dims + [final_dim_ext])  # [S_F]->S
    num_final_dim.name = "final_states"
    final_states_batch_idx, _ = rf.merge_dims(
        final_states_batch_idx_ext, dims=batch_dims + [final_dim_ext], out_dim=num_final_dim
    )  # [S_F]->B

    return FSA(
        batch_dims=batch_dims,
        label_dim=labels_with_blank_dim,
        num_states_dim_ext=num_states_dim_ext,
        num_states_dim=num_states_dim,
        start_states=start_states,
        states_batch_idx=states_batch_idx,
        num_trans_dim=num_trans_dim,
        trans_prev_state=trans_prev_state,
        trans_next_state=trans_next_state,
        trans_batch_label_idx=trans_batch_label_idx,
        num_final_dim=num_final_dim,
        final_states=final_states,
        final_states_batch_idx=final_states_batch_idx,
    )


def best_path_ctc(
    *,
    logits: Tensor,
    logits_normalized: bool = False,
    input_spatial_dim: Dim,
    targets: Tensor,
    targets_spatial_dim: Dim,
    blank_index: int,
) -> Tuple[Tensor, Tensor]:
    """
    :func:`best_path` with :func:`fsa_for_ctc` for the CTC label topology.

    :param logits: [B...,T,F+1]
    :param logits_normalized: whether the logits are already normalized.
        If False, we apply log_softmax to the logits.
        If True, the logits are used as is.
        (Note that there is no check or really the necessity to have the logits really normalized.)
    :param targets: [B...,T_out] -> F
    :param input_spatial_dim: T
    :param targets_spatial_dim: T_out
    :param blank_index: in F+1
    :return: tuple (best path, score).
        best path are the label indices [B...,T]->F+1.
        The score has shape [B...].
    """
    assert logits.feature_dim
    fsa = fsa_for_ctc(targets, targets_spatial_dim, blank_index=blank_index, labels_with_blank_dim=logits.feature_dim)
    return best_path(logits=logits, logits_normalized=logits_normalized, fsa=fsa, input_spatial_dim=input_spatial_dim)


def _safe_add(a: Tensor, b: Tensor) -> Tensor:
    """safe add, handles the case of -inf values."""
    return rf.where(rf.is_finite(a), a + b, a)


def _logaddexp(a: Tensor, b: Tensor) -> Tensor:
    import torch

    with torch.no_grad():
        max_x = rf.maximum(a, b)

    return max_x + rf.log(rf.exp(a - max_x) + rf.exp(b - max_x))


def _scatter_safe_logsumexp(
    source: Tensor, *, indices: Tensor, indices_dim: Dim, out_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
    """
    Like :func:`torch.scatter_reduce_` but doing safe_logsumexp as in :func:`safe_logsumpexp`.

    https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_reduce_.html

    As we are reducing, usually D_out < D_src.

    Note, there is also scatter_logsumexp
    (https://pytorch-scatter.readthedocs.io/en/1.4.0/_modules/torch_scatter/logsumexp.html)
    but this does not have the "safe" aspect as in :func:`safe_logsumexp`.

    :param source: for each index, the value to scatter into output, e.g. [D_src,...]
    :param indices: indices in dim in output. e.g. [D_src]->D_out
    :param indices_dim: D_src
    :return: tensor [D_out,...] with the scattered updates
    """
    import torch

    with torch.no_grad():
        max_x = rf.scatter(source, indices=indices, indices_dim=indices_dim, mode="max", out_dim=out_dim)  # [D_out,...]
        max_x_ = rf.gather(max_x, indices=indices, axis=out_dim)  # [D_src,...]
        max_x = rf.stop_gradient(max_x)
        max_x_ = rf.stop_gradient(max_x_)
    src_ = rf.exp(source - max_x_)
    tensor = rf.scatter(src_, indices=indices, indices_dim=indices_dim, mode="sum", out_dim=out_dim)
    tensor = rf.log(tensor)
    tensor = rf.where(rf.is_neg_infinite(max_x), rf.zeros((), dtype=source.dtype, device=source.device), tensor)
    tensor += max_x
    return tensor, out_dim


def setup_module(**_kwargs):  # run by pytest
    rf.select_backend_torch()


def test_fsa_for_ctc():
    batch_dim = Dim(2, name="batch")
    labels_dim = Dim(5, name="labels")
    targets_spatial_dim = Dim(rf.convert_to_tensor([4, 2], dims=[batch_dim]), name="targets_spatial")
    targets = rf.convert_to_tensor(
        [[1, 2, 3, 3], [2, 4, 0, 0]], dims=[batch_dim, targets_spatial_dim], sparse_dim=labels_dim
    )
    fsa = fsa_for_ctc(targets, targets_spatial_dim, labels_with_blank_dim=labels_dim, blank_index=0)

    assert fsa.batch_dims == [batch_dim]
    assert fsa.label_dim == labels_dim  # L

    # States
    num_states1, num_states2 = (targets_spatial_dim.dyn_size * 2 + 1).tolist()
    assert fsa.num_states_dim.dyn_size == num_states1 + num_states2  # S
    assert fsa.start_states.dims == (batch_dim,)  # [B...] -> S
    assert fsa.start_states.dtype.startswith("int")
    assert fsa.start_states.raw_tensor.tolist() == [0, num_states1]
    assert fsa.states_batch_idx.dims == (fsa.num_states_dim,)  # [S] -> B_
    assert fsa.states_batch_idx.sparse_dim == batch_dim
    assert fsa.states_batch_idx.raw_tensor.tolist() == [0] * num_states1 + [1] * num_states2

    # Transitions
    # Those are specific for the given targets above.
    trans1 = [
        # self-loops in blanks:
        [(0, 0, 0), (2, 2, 0), (4, 4, 0), (6, 6, 0), (8, 8, 0)],
        # self-loops in labels:
        [(1, 1, 1), (3, 3, 2), (5, 5, 3), (7, 7, 3)],
        # transitions to blanks:
        [(1, 2, 0), (3, 4, 0), (5, 6, 0), (7, 8, 0)],
        # transitions to labels:
        [(0, 1, 1), (2, 3, 2), (4, 5, 3), (6, 7, 3)],
        # skip transitions:
        [(1, 3, 2), (3, 5, 3)],
    ]
    trans2 = [
        # self-loops in blanks:
        [(0, 0, 0), (2, 2, 0), (4, 4, 0)],
        # self-loops in labels:
        [(1, 1, 2), (3, 3, 4)],
        # transitions to blanks:
        [(1, 2, 0), (3, 4, 0)],
        # transitions to labels:
        [(0, 1, 2), (2, 3, 4)],
        # skip transitions:
        [(1, 3, 4)],
    ]
    # Reorder them in the way we currently construct them in the code.
    # Also apply the state index offset and the batch label index offset.
    trans_ = [
        t1 + [(s1 + num_states1, s2 + num_states1, l + labels_dim.dimension) for (s1, s2, l) in t2]
        for (t1, t2) in zip(trans1, trans2)
    ]
    trans = sum(trans_, [])
    assert fsa.trans_prev_state.dims == (fsa.num_trans_dim,)  # [A] -> S
    assert fsa.trans_prev_state.sparse_dim == fsa.num_states_dim
    assert fsa.trans_prev_state.raw_tensor.tolist() == [x[0] for x in trans]
    assert fsa.trans_next_state.dims == (fsa.num_trans_dim,)  # [A] -> S
    assert fsa.trans_next_state.sparse_dim == fsa.num_states_dim
    assert fsa.trans_next_state.raw_tensor.tolist() == [x[1] for x in trans]
    assert fsa.trans_batch_label_idx.dims == (fsa.num_trans_dim,)
    assert fsa.trans_batch_label_idx.sparse_dim == batch_dim * labels_dim
    assert fsa.trans_batch_label_idx.raw_tensor.tolist() == [x[2] for x in trans]
    assert fsa.num_trans_dim.dyn_size == len(trans)  # A  -- test this last

    # Final states
    assert fsa.num_final_dim.get_dim_value() == 2 * batch_dim.dimension  # S_F
    assert fsa.final_states.dims == (fsa.num_final_dim,)  # [S_F]->S
    assert fsa.final_states.sparse_dim == fsa.num_states_dim
    assert fsa.final_states.raw_tensor.tolist() == [
        num_states1 - 2,
        num_states1 - 1,
        num_states1 + num_states2 - 2,
        num_states1 + num_states2 - 1,
    ]
    assert fsa.final_states_batch_idx.dims == (fsa.num_final_dim,)  # [S_F]->B
    assert fsa.final_states_batch_idx.sparse_dim == batch_dim
    assert fsa.final_states_batch_idx.raw_tensor.tolist() == [0, 0, 1, 1]


def test_best_path_ctc():
    import torch

    batch_dim = Dim(2, name="batch")
    labels_dim = Dim(5, name="labels")
    targets_spatial_dim = Dim(rf.convert_to_tensor([4, 2], dims=[batch_dim]), name="targets_spatial")
    targets: Tensor = rf.convert_to_tensor(
        [[1, 2, 3, 3], [2, 4, 0, 0]], dims=[batch_dim, targets_spatial_dim], sparse_dim=labels_dim
    )

    time_dim = Dim(rf.convert_to_tensor([11, 7], dims=[batch_dim]), name="time")
    logits = rf.zeros([batch_dim, time_dim, labels_dim], feature_dim=labels_dim)
    best_path_, score = best_path_ctc(
        logits=logits,
        input_spatial_dim=time_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=0,
    )
    assert best_path_.dims_set == {batch_dim, time_dim}
    assert score.dims == (batch_dim,)
    print(best_path_.raw_tensor)
    print(score.raw_tensor)

    # Mask repeated
    mask = best_path_ != rf.shift_right(best_path_, axis=time_dim, pad_value=0)
    targets_, time_dim_ = rf.masked_select(best_path_, mask=mask, dims=[time_dim])
    mask: Tensor = targets_ != 0  # non-blank  # noqa
    targets_, time_dim_ = rf.masked_select(targets_, mask=mask, dims=[time_dim_])
    targets_ = targets_.copy_masked(0)
    print(targets_.raw_tensor)
    assert targets.raw_tensor.tolist() == targets_.raw_tensor.tolist()

    score_ref = rf.log(rf.constant(1 / labels_dim.dimension, dims=())) * time_dim.get_size_tensor()
    print(score_ref.raw_tensor)
    torch.testing.assert_allclose(score.raw_tensor, score_ref.raw_tensor)


def _ref_best_path(am_scores: Tensor, am_spatial_dim: Dim, fsa: FSA) -> Tuple[Tensor, Tensor]:
    """
    Reference pure Python/PyTorch Viterbi algorithm, to find the best path/alignment.
    This is mostly intended for testing.

    :param am_scores: (time, batch, dim), in +log space
    :param am_spatial_dim:
    :param fsa:
    :return: (alignment, obs_scores), alignment is (time, batch), obs_scores is (batch,), in +log space
    """
    import torch
    from collections import defaultdict

    assert len(fsa.batch_dims) == 1
    batch_dim = fsa.batch_dims[0]
    am_scores = am_scores.copy_transpose([am_spatial_dim, batch_dim, am_scores.feature_dim])

    n_time, n_batch, dim = am_scores.raw_tensor.shape
    am_seq_len = am_spatial_dim.get_size_tensor().raw_tensor
    assert am_seq_len.shape == (n_batch,)

    zero_score = float("-inf")
    alignment = torch.zeros((n_time, n_batch), dtype=torch.int32)
    obs_scores = torch.full((n_batch,), zero_score, dtype=am_scores.raw_tensor.dtype)
    n_edges = fsa.num_trans_dim.get_dim_value()

    def search() -> List[Dict[int, Tuple[float, int]]]:
        start_idx = fsa.start_states.raw_tensor[batch_idx].item()
        states: Dict[int, Tuple[float, int]] = defaultdict(lambda: (zero_score, -1))  # state-idx -> score/edge
        states[start_idx] = (0.0, -1)
        res: List[Dict[int, Tuple[float, int]]] = []
        for t in range(n_time):
            if t >= am_seq_len[batch_idx]:
                break
            scores: Dict[int, List[Tuple[float, int]]] = defaultdict(list)  # state-idx -> list[score/edge]
            for edge_idx in range(n_edges):
                from_idx = fsa.trans_prev_state.raw_tensor[edge_idx].item()
                to_idx = fsa.trans_next_state.raw_tensor[edge_idx].item()
                emission_idx = fsa.trans_batch_label_idx.raw_tensor[edge_idx].item() % dim
                batch_idx_ = fsa.trans_batch_label_idx.raw_tensor[edge_idx].item() // dim
                if batch_idx_ != batch_idx:
                    continue
                if from_idx not in states or states[from_idx][0] == zero_score:
                    continue
                assert 0 <= emission_idx < dim
                score = states[from_idx][0] + am_scores.raw_tensor[t, batch_idx, emission_idx].item()
                scores[to_idx].append((score, edge_idx))
            states.clear()
            for state_idx in scores.keys():
                states[state_idx] = max(scores[state_idx], key=lambda _item: (_item[0], -_item[1]))
            res.append(dict(states))
        assert len(res) == am_seq_len[batch_idx]
        return res

    def select_best():
        """
        :return: nothing, fill alignment and obs_scores
        """
        scores: Dict[int, float] = {
            state_idx.item(): fwd_search_res[am_seq_len[batch_idx] - 1][state_idx.item()][0]
            for i, state_idx in enumerate(fsa.final_states.raw_tensor)
            if fsa.final_states_batch_idx.raw_tensor[i].item() == batch_idx
        }  # final state_idx -> score
        end_idx = max(scores, key=lambda state_idx_: scores[state_idx_])
        state_idx = end_idx
        for t in reversed(range(am_seq_len[batch_idx])):
            if state_idx not in fwd_search_res[t]:  # no path?
                alignment[t, batch_idx] = 0
                continue
            score, edge_idx = fwd_search_res[t][state_idx]
            if t == am_seq_len[batch_idx] - 1:
                obs_scores[batch_idx] = score
            from_idx = fsa.trans_prev_state.raw_tensor[edge_idx].item()
            emission_idx = fsa.trans_batch_label_idx.raw_tensor[edge_idx].item() % dim
            batch_idx_ = fsa.trans_batch_label_idx.raw_tensor[edge_idx].item() // dim
            assert batch_idx_ == batch_idx
            alignment[t, batch_idx] = emission_idx
            state_idx = from_idx

    for batch_idx in range(n_batch):
        fwd_search_res = search()
        select_best()

    return (
        rf.convert_to_tensor(alignment, dims=[am_spatial_dim, batch_dim], name="path"),
        rf.convert_to_tensor(obs_scores, dims=[batch_dim], name="scores"),
    )


def test_best_path_ctc_to_ref():
    import torch

    batch_dim = Dim(2, name="batch")
    labels_dim = Dim(5, name="labels")
    targets_spatial_dim = Dim(rf.convert_to_tensor([4, 2], dims=[batch_dim]), name="targets_spatial")
    targets = rf.convert_to_tensor(
        [[1, 2, 3, 3], [2, 4, 0, 0]], dims=[batch_dim, targets_spatial_dim], sparse_dim=labels_dim
    )

    time_dim = Dim(rf.convert_to_tensor([11, 7], dims=[batch_dim]), name="time")
    logits = rf.random_normal([batch_dim, time_dim, labels_dim], stddev=2.0, feature_dim=labels_dim)
    logits = rf.log_softmax(logits, axis=labels_dim)
    fsa = fsa_for_ctc(targets, targets_spatial_dim, labels_with_blank_dim=labels_dim, blank_index=0)
    print("Best path:")
    best_path_, score = best_path(
        logits=logits,
        logits_normalized=True,
        input_spatial_dim=time_dim,
        fsa=fsa,
    )
    assert best_path_.dims_set == {batch_dim, time_dim}
    assert score.dims == (batch_dim,)
    best_path_ = best_path_.copy_masked(0)
    print(best_path_.raw_tensor)
    print(score.raw_tensor)

    print("Best path reference:")
    best_path_ref, score_ref = _ref_best_path(logits, time_dim, fsa)
    print(best_path_ref.raw_tensor)
    print(score_ref.raw_tensor)

    torch.testing.assert_allclose(score.raw_tensor, score_ref.raw_tensor)
    torch.testing.assert_allclose(
        best_path_.copy_compatible_to_dims_raw([time_dim, batch_dim]), best_path_ref.raw_tensor
    )


def test_ctc_loss():
    import torch

    batch_dim = Dim(2, name="batch")
    labels_dim = Dim(5, name="labels")
    targets_spatial_dim = Dim(rf.convert_to_tensor([4, 2], dims=[batch_dim]), name="targets_spatial")
    targets: Tensor = rf.convert_to_tensor(
        [[1, 2, 3, 3], [2, 4, 0, 0]], dims=[batch_dim, targets_spatial_dim], sparse_dim=labels_dim
    )

    time_dim = Dim(rf.convert_to_tensor([11, 7], dims=[batch_dim]), name="time")
    logits = rf.random_normal([batch_dim, time_dim, labels_dim], stddev=2.0, feature_dim=labels_dim)
    log_probs = rf.log_softmax(logits, axis=labels_dim)

    loss = ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        input_spatial_dim=time_dim,
        targets=targets,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=0,
    )
    print("CTC loss:")
    print(loss.raw_tensor)

    print("CTC loss reference:")
    # Reference implementation.
    # We need to convert the logits to the right format.
    log_probs = log_probs.copy_transpose([time_dim, batch_dim, labels_dim])
    loss_ref = torch.nn.functional.ctc_loss(
        log_probs.raw_tensor,
        targets.raw_tensor,
        time_dim.get_size_tensor().raw_tensor,
        targets_spatial_dim.get_size_tensor().raw_tensor,
        blank=0,
        reduction="none",
        zero_infinity=True,
    )
    print(loss_ref)
    torch.testing.assert_allclose(loss.raw_tensor, loss_ref)
