"""
Utils
"""

from __future__ import annotations
from typing import TypeVar, Sequence, List, Tuple
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
    if values.shape[1] == 1:  # broadcast case
        assert num_index_own_dims == 1  # not implemented otherwise, maybe also unexpected
        return values
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
