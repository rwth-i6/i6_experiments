from __future__ import annotations
from typing import Tuple
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def apply_durations(enc: Tensor, *, in_spatial_dim: Dim, durations: Tensor) -> Tuple[Tensor, Dim]:
    """
    Apply durations to encoder output.

    :param enc: [batch..., in_spatial_dim, feat]
    :param in_spatial_dim:
    :param durations: [batch..., in_spatial_dim] -> int32 durations
    :return: expanded_enc: [batch..., out_spatial_dim, feat], out_spatial_dim
    """
    # Similar to masked_select
    durations = durations.copy_masked(0, dims=[in_spatial_dim])
    idxs = rf.cumsum(durations, spatial_dim=in_spatial_dim)  # [batch...,in_spatial_dim] -> idx in out_spatial_dim + 1
    new_size = rf.gather(idxs, indices=in_spatial_dim.get_dim_value_tensor() - 1, axis=in_spatial_dim)  # [batch...]
    out_spatial_dim = Dim(new_size, name="expanded_" + in_spatial_dim.name)
    out_spatial_dim_ext = out_spatial_dim + 1
    rel_idx_counts = rf.scatter(
        rf.ones((), device=enc.device, dtype="int32"),
        indices=idxs,
        indices_dim=in_spatial_dim,
        out_dim=out_spatial_dim_ext,
    )
    # rel_idx_counts: [batch...,out_spatial_dim+1] -> count of how many times each index was selected
    idxs_ = rf.cumsum(rel_idx_counts, spatial_dim=out_spatial_dim_ext)
    # idxs_: [batch...,out_spatial_dim+1] -> idx in in_spatial_dim
    idxs_, _ = rf.slice(idxs_, axis=out_spatial_dim_ext, size=out_spatial_dim)  # remove last element
    # idxs_: [batch...,out_spatial_dim] -> idx in in_spatial_dim (potentially with invalid indices in padded area)
    return rf.gather(enc, indices=idxs_, axis=in_spatial_dim, clip_to_valid=True), out_spatial_dim


# TODO tests...
