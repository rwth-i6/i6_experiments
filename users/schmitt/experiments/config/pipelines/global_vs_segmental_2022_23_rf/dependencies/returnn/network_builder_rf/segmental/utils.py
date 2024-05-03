from typing import Optional, Sequence, Tuple

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf



def get_non_blank_mask(x: Tensor, blank_idx: int):
  non_blank_mask = x != rf.convert_to_tensor(blank_idx)
  return rf.where(non_blank_mask, rf.sequence_mask(x.dims), rf.convert_to_tensor(False))


def get_masked(
        input: Tensor, mask: Tensor, mask_dim: Dim, batch_dims: Sequence[Dim], result_spatial_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:

  import torch

  if not result_spatial_dim:
    new_lens = rf.reduce_sum(rf.cast(mask, "int32"), axis=mask_dim)
    result_spatial_dim = Dim(name=f"{mask_dim.name}_masked", dimension=rf.copy_to_device(new_lens, "cpu"))
  else:
    new_lens = rf.copy_to_device(result_spatial_dim.get_size_tensor(), input.device)
  # max number of non-blank targets in the batch
  result_spatial_size = rf.cast(rf.reduce_max(new_lens, axis=batch_dims), "int32")
  mask_axis = mask.get_axis_from_description(mask_dim)
  # scatter indices
  idxs = rf.cast(mask, "int32").copy_template()
  idxs.raw_tensor = torch.cumsum(mask.raw_tensor.to(
    torch.int32), dim=mask_axis, dtype=torch.int32) - 1
  idxs = rf.where(mask, idxs, result_spatial_size)
  # scatter non-blank targets
  # blanks are scattered to the last position of each batch
  result_spatial_dim_temp = result_spatial_dim + 1
  result = rf.scatter(
    input, indices=idxs, indices_dim=mask_dim, out_dim=result_spatial_dim_temp)
  # remove accumulated blanks at the last position
  result = result.copy_transpose([result_spatial_dim_temp] + batch_dims)
  result_raw_tensor = result.raw_tensor
  result = result.copy_template_replace_dim_tag(0, result_spatial_dim)
  result.raw_tensor = result_raw_tensor[:-1]
  result = result.copy_transpose(batch_dims + [result_spatial_dim])

  return result, result_spatial_dim
