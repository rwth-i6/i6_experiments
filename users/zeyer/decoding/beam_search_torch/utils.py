"""
Utils
"""

from __future__ import annotations
from typing import TypeVar, Sequence, List, Tuple, Dict
import dataclasses
import torch

from .interface import (
    StateObjTensorExt,
    StateObjIgnored,
)


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
        return dataclasses.replace(values, tensor=batch_gather(values=values.tensor, indices=indices))
    elif isinstance(values, StateObjIgnored):
        return dataclasses.replace(values)
    elif values is None:
        return None
    else:
        raise TypeError(f"batch_gather_: unexpected {values} ({type(values).__name__})")


def combine_individual_seq_scores(
    prev_individual_seq_scores: Dict[str, torch.Tensor],
    individual_scores: Dict[str, torch.Tensor],
    *,
    beam_backrefs: torch.Tensor,
    labels: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    :param prev_individual_seq_scores: key -> [Batch,InBeam]
    :param individual_scores: key -> [Batch|1,InBeam|1,Vocab|1]
    :param beam_backrefs: [OutBeam] -> InBeam
    :param labels: [Batch,OutBeam] -> Vocab
    :return: individual_seq_scores: key -> [Batch,OutBeam]
    """
    individual_seq_scores = {}
    for k, score_ext in individual_scores.items():
        score_ext: torch.Tensor  # [Batch|1,InBeam|1,Vocab|1]
        score = _gather_label_score(score_ext, beam_backrefs=beam_backrefs, labels=labels)  # [Batch|1,OutBeam|1]
        if k in prev_individual_seq_scores:
            prev_seq_score = prev_individual_seq_scores[k]  # [Batch|1,InBeam|1]
            if prev_seq_score.shape[1] > 1:
                prev_seq_score = batch_gather(prev_seq_score, indices=beam_backrefs)
            seq_score = prev_seq_score + score
        else:
            seq_score = score
        individual_seq_scores[k] = seq_score
    return individual_seq_scores


def _gather_label_score(score_ext: torch.Tensor, *, beam_backrefs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    :param score_ext: [Batch|1,InBeam|1,Vocab|1].
        broadcast in InBeam only expected together with broadcast in Batch.
    :param beam_backrefs: [Batch,OutBeam] -> InBeam
    :param labels: [Batch,OutBeam] -> Vocab
    :return: [Batch|1,OutBeam|1]
    """
    # Derived from batch_gather.
    if score_ext.shape[0] == 1 < beam_backrefs.shape[0]:  # broadcast
        assert score_ext.shape[1] == 1  # also expect broadcast, not yet implemented otherwise
        assert score_ext.shape[2] == 1  # not yet implemented otherwise
        return score_ext.squeeze(2)  # [Batch=1,OutBeam=1]
    if score_ext.shape[1] == 1 and score_ext.shape[2] == 1:  # broadcast
        return score_ext.squeeze(2)  # [Batch,OutBeam=1]
    if score_ext.shape[2] == 1:  # broadcast
        score_ext = batch_gather(score_ext, indices=beam_backrefs)  # [Batch,OutBeam,Vocab=1]
        return score_ext.squeeze(2)  # [Batch,OutBeam]
    # score_ext: [Batch,InBeam,Vocab], no broadcasting
    num_labels = score_ext.shape[2]
    score_ext = score_ext.flatten(1)  # [Batch,InBeam*Vocab]
    return batch_gather(score_ext, indices=beam_backrefs * num_labels + labels)  # [Batch,OutBeam]
