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
    """
    :param source: [Batch...,SourceDims...]
    :param k: how many
    :param dim: axes of SourceDims, multiple dims to search in
    :param sorted: sorted output, see :func:`torch.topk`
    :return: values [Batch...,k], list of indices per dim, each [Batch...,k]
    """
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


def batch_gather(
    values: torch.Tensor, *, indices: torch.Tensor, batch_dim: int = 0, index_dim: int = 1
) -> torch.Tensor:
    """
    :param values: shape [Batch,Indices,ValuesDims...], e.g. [Batch,InBeam,...]
    :param indices: shape [Batch,IndicesDims...] -> Indices, e.g. [Batch,OutBeam] -> InBeam
    :param batch_dim: in values. in indices, batch is assumed first.
    :param index_dim: in values. must be >batch_dim (not implemented otherwise).
        in indices, index dims are expected after batch.
    :return: shape [Batch,IndicesDims...,ValuesDims...], e.g. [Batch,OutBeam,...],
        if batch_dim=0 and index_dim=1.
        Batch and index dim stays at the same place, index dim is replaced by indices dims from indices.
    """
    # Derived from returnn.torch.frontend._backend.TorchBackend.gather.
    # Case indices.dims_set.intersection(source.dims_set - {axis}).
    # We cannot use index_select in this case. Need to fallback to gather.
    assert indices.shape[0] == values.shape[batch_dim] and batch_dim < index_dim
    num_index_own_dims = indices.ndim - 1
    if num_index_own_dims == 1:
        indices_flat = indices  # good, [Batch,IndexDim]
    elif num_index_own_dims == 0:
        indices_flat = indices[:, None]  # [Batch,IndexDim=1]
    else:
        indices_flat = indices.flatten(1)  # [Batch,FlatIndexDim]
    indices_flat_bc = indices_flat.reshape(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else 1)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,1s...].
    indices_flat_exp = indices_flat_bc.expand(
        [
            indices_flat.shape[0] if i == batch_dim else (indices_flat.shape[1] if i == index_dim else d)
            for i, d in enumerate(values.shape)
        ]
    )  # batch_dim=0, index_dim=1 -> [Batch,IndexDim,ValuesDims...]
    out = torch.gather(values, dim=index_dim, index=indices_flat_exp.type(torch.int64))
    if num_index_own_dims == 1:
        pass  # nothing to do
    elif num_index_own_dims == 0:
        out = out.squeeze(index_dim)
    else:
        out = out.unflatten(index_dim, indices.shape[1:])
    if batch_dim == 0 and index_dim == 1:
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


def gather_(values: T, *, indices: torch.Tensor) -> T:
    """wraps ``values[indices]``"""
    if isinstance(values, torch.Tensor):
        return values[indices]
    elif isinstance(values, StateObjTensorExt):
        return dataclasses.replace(values, tensor=values.tensor[indices])
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
    :param beam_backrefs: [Batch,OutBeam] -> InBeam
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


def ensure_label_in_beam(
    *,
    seq_log_prob: torch.Tensor,
    seq_log_prob_ext: torch.Tensor,
    backrefs: torch.Tensor,
    labels: torch.Tensor,
    required_label: torch.Tensor,
    required_label_beam_idx: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Make sure the given label is in the beam.

    Also see :func:`returnn.tf.util.basic.beam_search`.

    :param seq_log_prob: [Batch,Beam]
    :param seq_log_prob_ext: [Batch,PrevBeam,Vocab]. needed to get required label score.
    :param backrefs: [Batch,Beam] -> PrevBeam
    :param labels: [Batch,Beam] -> Vocab
    :param required_label: [Batch] -> Vocab
    :param required_label_beam_idx: At what beam index (in Beam) to put the label into,
        and also where to expect the prev required label (in PrevBeam).
        (Currently only 0 supported, but we might also support -1 later
        (like in :func:`returnn.tf.util.basic.beam_search`)
        or even custom backrefs.)
    :return: new (seq_log_prob, backrefs, labels)
    """
    assert required_label_beam_idx == 0, "not implemented otherwise currently"
    required_label_prev_beam_idx = required_label_beam_idx

    required_label_score = batch_gather(
        seq_log_prob_ext[:, required_label_prev_beam_idx],  # [Batch,Vocab]
        indices=required_label,
    )  # [Batch]
    del seq_log_prob_ext

    batch_size, beam_size = seq_log_prob.shape
    device = seq_log_prob.device
    found = (labels == required_label[:, None]) & (backrefs == required_label_prev_beam_idx)  # [Batch,Beam]
    found: torch.Tensor
    # Case: we found the required label in the beam already -> reorder such that it is in the right beam idx
    # Other case: did not find -> remove the last entry in the beam, put required label into the right beam idx
    # If we found it, we can also remove this entry,
    # so then can put the required label into the right beam idx in both cases.
    # Use topk to select the right indices.
    # Extend beam at the end by the required label.
    labels_ = torch.concat([labels, required_label[:, None]], dim=1)  # [Batch,Beam+1]
    # Note that we found it either not at all or only max once. Thus found_ is exactly once True per batch.
    found_ = torch.concat([found, ~(found.any(dim=1, keepdim=True))], dim=1)  # [Batch,Beam+1]
    indices = torch.where(found_, 100, -torch.arange(beam_size + 1).to(device)[None, :])  # [Batch,Beam+1]
    assert required_label_beam_idx == 0  # currently designed that indices order is such that first is required label
    _, indices = torch.topk(indices, k=beam_size, dim=1, sorted=True)  # [Batch,Beam] -> Beam+1
    backrefs_ = torch.concat(
        [backrefs, torch.full([batch_size, 1], required_label_prev_beam_idx, device=device, dtype=backrefs.dtype)],
        dim=1,
    )  # [Batch,Beam+1]
    seq_log_prob_ = torch.concat([seq_log_prob, required_label_score[:, None]], dim=1)  # [Batch,Beam+1]

    seq_log_prob = batch_gather(seq_log_prob_, indices=indices)
    backrefs = batch_gather(backrefs_, indices=indices)
    labels = batch_gather(labels_, indices=indices)
    return seq_log_prob, backrefs, labels


def masked_select(values: torch.Tensor, mask: torch.Tensor, *, out_len: int) -> torch.Tensor:
    """
    This has the advantage over :func:`torch.masked_select`
    that we do not need to perform a CUDA synchronization.
    We can avoid that when we know the output length in advance.

    :param values:
    :param mask:
    :param out_len:
    :return: values.flatten()[idx], i.e. shape [out_len]
    """
    if values.shape != mask.shape:
        assert values.dim() == mask.dim()
        shape = [max(d1, d2) for d1, d2 in zip(values.shape, mask.shape)]
        values = values.expand(*shape)
        mask = mask.expand(*shape)
    idx = nonzero(mask.flatten(), out_len=out_len)  # [out_len]
    return values[idx]


def nonzero(mask: torch.Tensor, *, out_len: int) -> torch.Tensor:
    """
    This has the advantage over :func:`torch.nonzero`
    that we do not need to perform a CUDA synchronization.
    We can avoid that when we know the output length in advance.

    :param mask: flattened mask, bool
    :param out_len:
    :return: indices of True elements, shape [out_len]
    """
    assert mask.dim() == 1 and mask.dtype == torch.bool
    # Sort currently does not support bool dtype on CUDA, thus cast to int.
    idx = torch.argsort(mask.flatten().to(torch.int8), stable=True, descending=True)  # [in_len]
    idx = idx[:out_len]  # [out_len]
    return idx
