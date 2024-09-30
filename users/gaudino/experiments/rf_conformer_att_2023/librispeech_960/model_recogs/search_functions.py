from __future__ import annotations

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.tensor_array import TensorArray

import torch

def remove_blank_and_eos(hyps: Tensor, max_out_len, batch_dims, beam_dim, target_dim, blank_idx: int, eos_idx: int) -> Tensor:
    batch_size = batch_dims[0].get_dim_value()
    beam_size = beam_dim.get_dim_value()
    out_spatial_dim = Dim(int(max_out_len), name=f"out-spatial")
    out_seq_lens = torch.zeros((beam_size, batch_size), dtype=torch.int32)
    seq_targets_ = None
    # remove blank and eos
    for i in range(batch_size):
        seq_targets_beam_ = None
        for j in range(beam_size):
            hyp = hyps[i, j]
            hyp_shifted = torch.roll(hyp, 1)
            mask_repeat = hyp != hyp_shifted
            hyp = torch.masked_select(hyp, mask_repeat)
            mask_blank = torch.logical_and(hyp != blank_idx, hyp != eos_idx)
            hyp = torch.masked_select(hyp, mask_blank)
            out_seq_lens[j,i] = hyp.shape[0]
            hyp_pad = torch.nn.functional.pad(
                hyp, (0, max_out_len - hyp.shape[0]), mode="constant", value=0
            )
            hyp_tensor = rf.Tensor(
                name=f"hyp_{i}_{j}",
                dims=[out_spatial_dim],
                dtype="int64",
                raw_tensor=hyp_pad,
                sparse_dim=target_dim,
            )
            if not seq_targets_beam_:
                seq_targets_beam_ = TensorArray(hyp_tensor)
                seq_targets_beam_.push_back(hyp_tensor)
            else:
                seq_targets_beam_.push_back(hyp_tensor)

        seq_targets_beam = seq_targets_beam_.stack(axis=beam_dim)
        # seq_targets_beam.dims[1].dyn_size_ext = out_seq_lens_tensor
        if not seq_targets_:
            seq_targets_ = TensorArray(seq_targets_beam)
            seq_targets_.push_back(seq_targets_beam)
        else:
            seq_targets_.push_back(seq_targets_beam)

    out_seq_lens_tensor = rf.Tensor(
        name="out_seq_lens",
        dims=[beam_dim, batch_dims[0]],
        dtype="int32",
        raw_tensor=out_seq_lens.to(torch.int32),
    )

    seq_targets = seq_targets_.stack(axis=batch_dims[0])
    seq_targets.dims[2].dyn_size_ext = out_seq_lens_tensor

    seq_targets = seq_targets.copy_transpose(
        [out_spatial_dim] + batch_dims + [beam_dim]
    )

    return seq_targets, out_spatial_dim

def remove_eos_from_start_and_end(hyp: torch.Tensor, eos_idx: int) -> torch.Tensor:

    mask_eos = hyp != eos_idx
    non_zero_indices = torch.nonzero(mask_eos, as_tuple=False)
    if len(non_zero_indices) == 0:
        return hyp
    first_non_zero = non_zero_indices[0]
    last_non_zero = non_zero_indices[-1]

    hyp = hyp[first_non_zero:last_non_zero+1]

    return hyp