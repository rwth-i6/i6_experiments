import torch
import torchaudio
from typing import Optional



from returnn.tensor import Dim, single_step_dim, TensorDict, batch_dim
import returnn.frontend as rf


def ctc_align_get_center_positions(ctc_alignment: torch.Tensor, blank_idx: int):
  is_label_position = ctc_alignment[0] != blank_idx  # type: torch.Tensor

  # Find indices where the mask is True
  true_indices = torch.where(is_label_position)[0]

  # Find differences between consecutive True indices
  diffs = torch.diff(true_indices)

  # Identify the start of each new patch
  patch_starts = torch.cat([torch.tensor([0], device=diffs.device), torch.where(diffs > 1)[0] + 1])

  # Identify the end of each patch
  patch_ends = torch.cat([torch.where(diffs > 1)[0], torch.tensor([len(true_indices) - 1], device=diffs.device)])

  # Compute the center index for each patch
  centers = (true_indices[patch_starts] + true_indices[patch_ends]) // 2

  return centers


def _forced_align(
        *,
        model,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
):
  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  assert len(batch_dims) == 1, f"expected one batch dim, but got {batch_dims}"

  collected_outputs = {}
  enc_args, enc_spatial_dim = model.encode(
    data, in_spatial_dim=data_spatial_dim,
  )
  ctc_logits = enc_args
  ctc_log_probs = rf.log_softmax(ctc_logits, axis=model.target_dim_w_blank)
  ctc_log_probs = ctc_log_probs.copy_transpose([batch_dim, enc_spatial_dim, model.target_dim_w_blank])

  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
  non_blank_targets_spatial_sizes = rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext)
  print('target dyn size', non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor)
  print('target max size', non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor.max())

  alignment_rf = rf.zeros(dims=batch_dims + [enc_spatial_dim], sparse_dim=model.target_dim_w_blank, dtype="int32")
  torch_ctc_log_probs = ctc_log_probs.raw_tensor
  batch_size = torch_ctc_log_probs.shape[0]
  for b in range(batch_size):
    input_len_b = enc_spatial_sizes.raw_tensor[b].item()
    target_len_b = non_blank_targets_spatial_sizes.raw_tensor[b].item()
    alignment_b, _ = torchaudio.functional.forced_align(
      log_probs=ctc_log_probs.raw_tensor[b, :input_len_b][None].contiguous(),
      targets=non_blank_targets.raw_tensor[b, :target_len_b][None],
      input_lengths=enc_spatial_sizes.raw_tensor[b][None],
      target_lengths=non_blank_targets_spatial_sizes.raw_tensor[b][None],
      blank=model.blank_idx,
    )

    alignment_rf.raw_tensor[b, :input_len_b] = alignment_b[0]

  return alignment_rf, enc_spatial_dim