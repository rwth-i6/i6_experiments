from typing import Optional, Sequence, Tuple, List
import torch

from returnn.tensor import Tensor, Dim
import returnn.frontend as rf


def get_non_blank_mask(x: Tensor, blank_idx: int, device=rf.get_default_device()):
  non_blank_mask = x != rf.convert_to_tensor(blank_idx, device=device)
  return rf.where(non_blank_mask, rf.sequence_mask(x.dims, device=device), rf.convert_to_tensor(False, device=device))


def get_masked(
        input: Tensor, mask: Tensor, mask_dim: Dim, batch_dims: Sequence[Dim], result_spatial_dim: Optional[Dim] = None
) -> Tuple[Tensor, Dim]:
  if not result_spatial_dim:
    new_lens = rf.reduce_sum(rf.cast(mask, "int32"), axis=mask_dim)
    result_spatial_dim = Dim(name=f"{mask_dim.name}_masked", dimension=rf.copy_to_device(new_lens, "cpu"))
  else:
    new_lens = rf.copy_to_device(result_spatial_dim.get_size_tensor(), input.device)
  # max number of non-blank targets in the batch
  result_spatial_size = rf.cast(rf.reduce_max(new_lens, axis=new_lens.dims), "int32")
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
  rem_dims = result.remaining_dims([result_spatial_dim_temp] + batch_dims)
  result = result.copy_transpose([result_spatial_dim_temp] + batch_dims + rem_dims)
  result_raw_tensor = result.raw_tensor
  result = result.copy_template_replace_dim_tag(0, result_spatial_dim)
  result.raw_tensor = result_raw_tensor[:-1]
  result = result.copy_transpose(batch_dims + [result_spatial_dim] + rem_dims)

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
        align_targets_spatial_dim: Dim,
        center_window_size: Optional[int],
        batch_dims: Sequence[Dim],
        out_spatial_dim: Dim
):
  targets_range = rf.range_over_dim(align_targets_spatial_dim, dtype="int32")
  for batch_dim in batch_dims:
    targets_range = rf.expand_dim(targets_range, batch_dim)
  non_blank_positions, _ = get_masked(
    targets_range, non_blank_mask, align_targets_spatial_dim, batch_dims, out_spatial_dim
  )
  if center_window_size is None:
    starts = None
    lens = None
  else:
    starts = rf.maximum(
      rf.convert_to_tensor(0, dtype="int32"), non_blank_positions - center_window_size // 2)
    ends = rf.minimum(
      rf.copy_to_device(align_targets_spatial_dim.get_size_tensor() - 1, non_blank_positions.device),
      non_blank_positions + center_window_size // 2
    )
    lens = ends - starts + 1

  return starts, lens, non_blank_positions


def get_emit_ground_truth(
        align_targets: Tensor,
        blank_idx: int
):
  non_blank_mask = get_non_blank_mask(align_targets, blank_idx)
  result = rf.where(non_blank_mask, rf.convert_to_tensor(1), rf.convert_to_tensor(0))
  sparse_dim = Dim(name="emit_ground_truth", dimension=2)
  # result = rf.expand_dim(result, sparse_dim)
  result.sparse_dim = sparse_dim

  return result, sparse_dim


def copy_tensor_replace_dim_tag(tensor: Tensor, old_dim_tag: Dim, new_dim_tag: Dim):
  tensor_raw = tensor.raw_tensor
  tensor = tensor.copy_template_replace_dim_tag(
    tensor.get_axis_from_description(old_dim_tag), new_dim_tag
  )
  tensor.raw_tensor = tensor_raw
  return tensor


def get_linear_alignment(
        non_blank_targets: Tensor,
        non_blank_targets_spatial_dim: Dim,
        enc_spatial_dim: Dim,
        batch_dims: Sequence[Dim],
        blank_idx: int,
):
  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.get_size_tensor(), non_blank_targets.device)
  non_blank_targets_spatial_sizes = rf.copy_to_device(
    non_blank_targets_spatial_dim.get_size_tensor(), non_blank_targets.device)

  # linearly distributed label positions over encoder dimension
  linear_label_positions = rf.range_over_dim(non_blank_targets_spatial_dim)
  linear_label_positions = rf.cast(
    (linear_label_positions + 0.5) * (enc_spatial_sizes / non_blank_targets_spatial_sizes),
    "int32"
  )
  # set positions, which are too large, to T+1 (cut off this dummy frame later)
  linear_label_positions = rf.where(
    linear_label_positions < enc_spatial_sizes,
    linear_label_positions,
    enc_spatial_sizes
  )
  enc_spatial_dim_ext = enc_spatial_dim + 1
  linear_label_positions.sparse_dim = enc_spatial_dim_ext

  # scatter non-blank targets into zero tensor
  linear_alignment = rf.scatter(
    non_blank_targets,
    indices=linear_label_positions,
    indices_dim=non_blank_targets_spatial_dim,
    out_dim=enc_spatial_dim_ext,
  )
  # replace all non-blank frames with blank
  linear_label_positions_scattered = rf.scatter(
    linear_label_positions,
    indices=linear_label_positions,
    indices_dim=non_blank_targets_spatial_dim,
    out_dim=enc_spatial_dim_ext,
  )
  linear_alignment = rf.where(
    rf.range_over_dim(enc_spatial_dim_ext) != linear_label_positions_scattered,
    blank_idx,
    linear_alignment
  )
  # cut off dummy frame
  linear_alignment = linear_alignment.copy_transpose([enc_spatial_dim_ext] + batch_dims)
  linear_alignment_raw = linear_alignment.raw_tensor
  linear_alignment = linear_alignment.copy_template_replace_dim_tag(
    linear_alignment.get_axis_from_description(enc_spatial_dim_ext),
    enc_spatial_dim,
  )
  linear_alignment.raw_tensor = linear_alignment_raw[:-1]
  return linear_alignment


def log_softmax_sep_blank(
        logits: Tensor,
        target_dim: Dim,
        blank_idx: int,
):
  """
  Split logits into blank and non-blank and then calculate sigmoid and softmax separately and then recombine
  """
  assert blank_idx == 0, "Only blank_idx=0 is supported"

  blank_dim = Dim(name="blank", dimension=1)
  non_blank_dim = target_dim - 1
  blank_logits, label_logits = rf.split(
    logits, axis=target_dim, out_dims=[blank_dim, non_blank_dim]
  )
  blank_log_prob = rf.log(rf.sigmoid(blank_logits))
  emit_log_prob = rf.log(rf.sigmoid(-blank_logits))
  label_log_prob = rf.log_softmax(label_logits, axis=non_blank_dim) + rf.squeeze(emit_log_prob, axis=blank_dim)
  log_prob, _ = rf.concat((blank_log_prob, blank_dim), (label_log_prob, non_blank_dim), out_dim=target_dim)

  return log_prob


def cumsum(x: Tensor, dim: Dim):
  orig_dims = x.dims
  x = x.copy_transpose([dim] + x.remaining_dims([dim]))
  x_raw = x.raw_tensor
  x = x.copy_template()
  x.raw_tensor = torch.cumsum(x_raw, dim=x.get_axis_from_description(dim), dtype=x_raw.dtype)
  x = x.copy_transpose(orig_dims)
  return x


def scatter_w_masked_indices(
        x: Tensor,
        mask: Tensor,
        scatter_indices: Tensor,
        result_spatial_dim: Dim,
        indices_dim: Dim,
        batch_dims: List[Dim],
):
  scatter_spatial_dim_sizes = rf.copy_to_device(result_spatial_dim.get_size_tensor())
  scatter_spatial_dim_sizes_max = rf.cast(rf.reduce_max(scatter_spatial_dim_sizes, axis=scatter_spatial_dim_sizes.dims), "int32")
  # scatter out-of-bounds indices to extended last position
  indices = rf.where(
    mask,
    scatter_indices,
    scatter_spatial_dim_sizes_max
  )
  # scatter according to indices into extended dim
  result_spatial_dim_temp = result_spatial_dim + 1
  result = rf.scatter(
    x, indices=indices, indices_dim=indices_dim, out_dim=result_spatial_dim_temp)
  # remove accumulated results at the last position
  rem_dims = result.remaining_dims([result_spatial_dim_temp] + batch_dims)
  result = result.copy_transpose([result_spatial_dim_temp] + batch_dims + rem_dims)
  result_raw_tensor = result.raw_tensor
  result = result.copy_template_replace_dim_tag(0, result_spatial_dim)
  result.raw_tensor = result_raw_tensor[:-1]
  result = result.copy_transpose(batch_dims + [result_spatial_dim] + rem_dims)

  return result
