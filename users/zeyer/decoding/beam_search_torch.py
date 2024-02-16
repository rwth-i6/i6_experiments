"""
Beam search
"""

from __future__ import annotations
from typing import Sequence, Tuple, List, TypeVar

from dataclasses import dataclass, replace as dataclass_replace
import functools
import torch
import tree

from .interface_torch import LabelScorerIntf, StateObjTensorExt, StateObjIgnored


@dataclass
class BeamSearchOpts:
    beam_size: int  # e.g. 12
    length_normalization_exponent: float  # e.g. 1 to enable, 0 to disable
    bos_label: int
    eos_label: int
    num_labels: int


def beam_search(
    label_scorer: LabelScorerIntf,
    *,
    batch_size: int,
    max_seq_len: torch.Tensor,
    device: torch.device,
    opts: BeamSearchOpts,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    beam search

    :param label_scorer:
    :param batch_size:
    :param max_seq_len: e.g. use encoder length. shape [Batch]
    :param device:
    :param opts:
    :return: seq_targets, seq_log_prob, out_seq_len:
        seq_targets: [Batch,FinalBeam,OutSeqLen]
        seq_log_prob: [Batch,FinalBeam]
        out_seq_len: [Batch,FinalBeam]
    """
    # Eager-mode implementation of beam search.
    # Initial state.
    beam_size = 1
    state = label_scorer.get_initial_state(batch_size=batch_size, device=device)
    target = torch.full([batch_size, beam_size], opts.bos_label, device=device)
    ended = torch.full([batch_size, beam_size], False, device=device)
    out_seq_len = torch.full([batch_size, beam_size], 0, device=device)
    seq_log_prob = torch.full([batch_size, beam_size], 0.0, device=device)

    masked_finished_log_prob = torch.where(
        torch.arange(0, opts.num_labels, device=device) == opts.eos_label, 0.0, -1.0e30
    )  # [Vocab]

    i = 0
    seq_targets = []
    seq_backrefs = []
    while True:
        label_log_prob, new_state = label_scorer.score_and_update_state(prev_state=state, prev_label=target)
        # label_log_prob: [Batch,InBeam,Vocab]
        # new_state: all tensors have [Batch,InBeam,...]

        # Filter out finished beams
        label_log_prob = torch.where(ended[:, :, None], masked_finished_log_prob[None, None, :], label_log_prob)
        seq_log_prob = seq_log_prob[:, :, None] + label_log_prob  # [Batch,InBeam,Vocab]
        seq_log_prob, (backrefs, target) = top_k_nd(seq_log_prob, k=opts.beam_size, dim=[1, 2])  # all [Batch,Beam]
        beam_size = seq_log_prob.shape[1]
        seq_targets.append(target)
        seq_backrefs.append(backrefs)
        state = tree.map_structure(functools.partial(batch_gather_, indices=backrefs), new_state)  # [Batch,Beam,...]
        ended = batch_gather(ended, indices=backrefs)  # [Batch,Beam]
        out_seq_len = batch_gather(out_seq_len, indices=backrefs)  # [Batch,Beam]
        i += 1

        ended = ended | (target == opts.eos_label)
        ended = ended | (i >= max_seq_len)[:, None].to(device)  # [Batch,Beam]
        if ended.all():
            break
        out_seq_len = out_seq_len + torch.where(ended, 0, 1)

        if i > 1 and opts.length_normalization_exponent != 0:
            # Length-normalized scores, so we evaluate score_t/len.
            # If seq ended, score_i/i == score_{i-1}/(i-1), thus score_i = score_{i-1}*(i/(i-1))
            # Because we count with EOS symbol, shifted by one.
            seq_log_prob *= torch.where(
                ended,
                (i / (i - 1)) ** opts.length_normalization_exponent,
                1.0,
            )

    if i > 1 and opts.length_normalization_exponent != 0:
        # All seq_log_prob will be normalized by (1/(out_seq_len+1)**length_normalization_exponent.
        seq_log_prob *= (1 / (i - 1)) ** opts.length_normalization_exponent

    # Backtrack via backrefs, resolve beams.
    seq_targets_ = []
    indices = torch.arange(beam_size, device=device)[None, :].expand(batch_size, -1)  # [Batch,FinalBeam] -> FinalBeam
    for backrefs, target in zip(seq_backrefs[::-1], seq_targets[::-1]):
        # indices: [Batch,FinalBeam] -> Beam
        # backrefs: [Batch,Beam] -> PrevBeam
        seq_targets_.insert(0, batch_gather(target, indices=indices))  # [Batch,FinalBeam]
        indices = batch_gather(backrefs, indices=indices)  # [Batch,FinalBeam] -> PrevBeam

    seq_targets = torch.stack(seq_targets_, dim=2)  # [Batch,FinalBeam,OutSeqLen]

    return seq_targets, seq_log_prob, out_seq_len


# noinspection PyShadowingBuiltins
def top_k_nd(
    source: torch.Tensor, *, k: int, dim: Sequence[int], sorted: bool = True
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    # Derived from returnn.torch.frontend._backend.TorchBackend.top_k.
    # Move axis to the end, in the right order.
    dim = [(d + source.ndim) % source.ndim for d in dim]
    source = source.permute([d for d in range(source.ndim) if d not in dim] + list(dim))
    source_flat = source.flatten(start_dim=source.ndim - len(dim))
    values, indices = torch.topk(source_flat, k=k, dim=-1, largest=True, sorted=sorted)
    indices_out = []
    for i in reversed(list(range(len(dim)))):
        a_dim = source.shape[source.ndim - len(dim) + i]
        indices_out_ = indices % a_dim
        indices = indices // a_dim
        indices_out.insert(0, indices_out_)
    return values, indices_out


def batch_gather(values: torch.Tensor, *, indices: torch.Tensor) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...]
    """
    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[0]
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good, [Batch,IndexDim]
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]  # [Batch,IndexDim=1]
    else:
        indices_flat = indices.flatten(1)  # [Batch,FlatIndexDim]
    indices_flat_bc = indices_flat.reshape(list(indices_flat.shape) + [1] * (values.ndim - 2))  # [Batch,IndexDim,1s...]
    indices_flat_exp = indices_flat_bc.expand(indices_flat.shape + values.shape[2:])  # [Batch,IndexDim,ValuesDims...]
    out = torch.gather(values, dim=1, index=indices_flat_exp.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(1)
    else:
        out = out.unflatten(1, indices.shape[1:])
    assert out.shape == indices.shape + values.shape[2:]
    return out


T = TypeVar("T")


def batch_gather_(values: T, *, indices: torch.Tensor) -> T:
    """calls :func:`batch_gather`"""
    if isinstance(values, torch.Tensor):
        return batch_gather(values, indices=indices)
    elif isinstance(values, StateObjTensorExt):
        return dataclass_replace(values, tensor=batch_gather(values=values.tensor, indices=indices))
    elif isinstance(values, StateObjIgnored):
        return dataclass_replace(values)
    elif values is None:
        return None
    else:
        raise TypeError(f"batch_gather_: unexpected {values} ({type(values).__name__})")
