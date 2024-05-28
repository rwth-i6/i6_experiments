from typing import Optional, Sequence, Tuple
import torch

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel


def get_non_blank_mask(x: Tensor, blank_idx: int):
  non_blank_mask = x != rf.convert_to_tensor(blank_idx)
  return rf.where(non_blank_mask, rf.sequence_mask(x.dims), rf.convert_to_tensor(False))


def get_masked(
        input: Tensor, mask: Tensor, mask_dim: Dim, batch_dims: Sequence[Dim], result_spatial_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
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


def get_unmasked(
        input: Tensor, input_spatial_dim: Dim, mask: Tensor, mask_spatial_dim: Dim
):
  mask_shifted = rf.shift_right(mask, axis=mask_spatial_dim, pad_value=False)
  mask_axis = mask.get_axis_from_description(mask_spatial_dim)

  # changelog
  # 24.05.24: changed int32->int64 and added 'clip_to_valid=True' since i got an CUDA idx out of bounds error
  # when testing a new feature. weirdly, i did not see this error in the log.run.1 file of existing trainings
  # using this function.
  cumsum = rf.cast(mask_shifted, "int64").copy_template()
  cumsum.raw_tensor = torch.cumsum(
    mask_shifted.raw_tensor.to(torch.int64), dim=mask_axis, dtype=torch.int64
  )
  return rf.gather(
    input,
    indices=cumsum,
    axis=input_spatial_dim,
    clip_to_valid=True
  )


def get_segment_starts_and_lens(
        non_blank_mask: Tensor,
        align_targets: Tensor,
        align_targets_spatial_dim: Dim,
        model: SegmentalAttentionModel,
        batch_dims: Sequence[Dim],
        out_spatial_dim: Dim
):
  targets_range = rf.range_over_dim(align_targets_spatial_dim, dtype="int32")
  targets_range = rf.expand_dim(targets_range, batch_dims[0])
  non_blank_positions, _ = get_masked(
    targets_range, non_blank_mask, align_targets_spatial_dim, batch_dims, out_spatial_dim
  )
  starts = rf.maximum(
    rf.convert_to_tensor(0, dtype="int32"), non_blank_positions - model.center_window_size // 2)
  ends = rf.minimum(
    rf.copy_to_device(align_targets_spatial_dim.get_size_tensor() - 1, non_blank_positions.device),
    non_blank_positions + model.center_window_size // 2
  )
  lens = ends - starts + 1

  return starts, lens


def get_emit_ground_truth(
        align_targets: Tensor,
        blank_idx: int
):
  non_blank_mask = get_non_blank_mask(align_targets, blank_idx)
  result = rf.where(non_blank_mask, rf.convert_to_tensor(1), rf.convert_to_tensor(0))
  sparse_dim = Dim(name="emit_ground_truth", dimension=2)
  # result = rf.expand_dim(result, sparse_dim)
  result.sparse_dim = sparse_dim
  torch.set_printoptions(threshold=10000)

  return result, sparse_dim
