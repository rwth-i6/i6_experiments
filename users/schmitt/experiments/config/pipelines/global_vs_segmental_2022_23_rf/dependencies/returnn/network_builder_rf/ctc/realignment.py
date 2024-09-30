import torch
import torchaudio
from typing import Optional

from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.model import CtcModel

from returnn.tensor import Dim, single_step_dim, TensorDict
import returnn.frontend as rf


def ctc_align_get_center_positions(ctc_alignment: torch.Tensor, blank_idx: int) -> torch.Tensor:
  is_label_repetition = torch.logical_and(
    ctc_alignment[0, 1:] == ctc_alignment[0, :-1],
    ctc_alignment[0, 1:] != blank_idx
  )
  zeros = torch.zeros((1,), dtype=torch.bool, device=is_label_repetition.device)
  is_label_repetition = torch.cat([is_label_repetition, zeros], dim=0)

  # create new mask
  is_center_position = torch.zeros_like(
    ctc_alignment[0], dtype=torch.bool, device=ctc_alignment.device)

  if is_label_repetition.any():
    # Find indices where the mask is True
    true_indices = torch.where(is_label_repetition)[0]

    # Find differences between consecutive True indices
    diffs = torch.diff(true_indices)

    # Identify the start of each new patch
    patch_starts = torch.cat([torch.tensor([0], device=diffs.device), torch.where(diffs > 1)[0] + 1])

    # Identify the end of each patch
    patch_ends = torch.cat([torch.where(diffs > 1)[0], torch.tensor([len(true_indices) - 1], device=diffs.device)])

    # Compute the center index for each patch
    centers = (true_indices[patch_starts] + true_indices[patch_ends]) // 2
    centers += 1  # each patch is one too small because of ctc_alignment[0, 1:] == ctc_alignment[0, :-1]
    is_center_position[centers] = True

  # now add the center positions, where the label is not repeated (not captured by the above)
  shifted_left = torch.cat([ctc_alignment[0, 1:], torch.tensor([0], device=ctc_alignment.device)], dim=0)
  shifted_right = torch.cat([torch.tensor([0], device=ctc_alignment.device), ctc_alignment[0, :-1]], dim=0)
  is_no_label_repetition = torch.logical_and(
    ctc_alignment[0] != shifted_right,
    ctc_alignment[0] != shifted_left
  )
  is_no_label_repetition = torch.logical_and(is_no_label_repetition, ctc_alignment[0] != blank_idx)

  is_center_position = torch.logical_or(is_center_position, is_no_label_repetition)
  centers = torch.where(is_center_position)[0]

  return centers


def model_realign_(
        *,
        model: CtcModel,
        data: rf.Tensor,
        data_spatial_dim: Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: Dim,
):
  batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim))
  assert len(batch_dims) == 1, f"expected one batch dim, but got {batch_dims}"

  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  ctc_projection_layer = getattr(model.encoder, f"enc_aux_logits_{len(model.encoder.layers)}")
  ctc_logits = ctc_projection_layer(enc_args["enc"])
  ctc_log_probs = rf.log_softmax(ctc_logits, axis=model.align_target_dim)

  enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
  non_blank_targets_spatial_sizes = rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext)

  alignment_rf = rf.zeros(dims=batch_dims + [enc_spatial_dim], sparse_dim=model.align_target_dim, dtype="int32")

  for b in range(ctc_log_probs.raw_tensor.size(0)):
    input_len_b = enc_spatial_sizes.raw_tensor[b].item()
    target_len_b = non_blank_targets_spatial_sizes.raw_tensor[b].item()
    alignment_b, _ = torchaudio.functional.forced_align(
      log_probs=ctc_log_probs.raw_tensor[b, :input_len_b][None],
      targets=non_blank_targets.raw_tensor[b, :target_len_b][None],
      input_lengths=enc_spatial_sizes.raw_tensor[b][None],
      target_lengths=non_blank_targets_spatial_sizes.raw_tensor[b][None],
      blank=model.blank_idx,
    )

    center_positions = ctc_align_get_center_positions(alignment_b, model.blank_idx)

    assert center_positions.size(0) == target_len_b, f"expected {target_len_b} center positions, but got {center_positions.size(0)}"

    center_position_mask = torch.zeros_like(alignment_b, dtype=torch.bool, device=alignment_b.device)
    center_position_mask[0, center_positions] = True
    alignment_b[0, ~center_position_mask[0]] = model.blank_idx
    alignment_rf.raw_tensor[b, :input_len_b] = alignment_b[0]

  return alignment_rf, enc_spatial_dim


def _returnn_v2_forward_step(*, model, extern_data: TensorDict, **_kwargs_unused):
  import returnn.frontend as rf
  from returnn.tensor import Tensor, Dim, batch_dim
  from returnn.config import get_global_config

  if rf.is_executing_eagerly():
    batch_size = int(batch_dim.get_dim_value())
    for batch_idx in range(batch_size):
      seq_tag = extern_data["seq_tag"].raw_tensor[batch_idx].item()
      print(f"batch {batch_idx + 1}/{batch_size} seq_tag: {seq_tag!r}")

  config = get_global_config()
  default_input_key = config.typed_value("default_input")
  data = extern_data[default_input_key]
  data_spatial_dim = data.get_time_dim_tag()
  realign_def = config.typed_value("_realign_def")

  default_target_key = config.typed_value("target")
  targets = extern_data[default_target_key]
  targets_spatial_dim = targets.get_time_dim_tag()

  realign_out = realign_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    non_blank_targets=targets,
    non_blank_targets_spatial_dim=targets_spatial_dim,
  )

  if len(realign_out) == 2:
    # realign results including viterbi_align,
    # out_spatial_dim,
    viterbi_align, out_spatial_dim = realign_out
  else:
    raise ValueError(f"unexpected num outputs {len(realign_out)} from recog_def")
  assert isinstance(viterbi_align, Tensor)
  assert isinstance(out_spatial_dim, Dim)
  rf.get_run_ctx().mark_as_output(viterbi_align, "viterbi_align", dims=[batch_dim, out_spatial_dim])


_v2_forward_out_alignment_filename = "realignment.hdf"


def _returnn_v2_get_forward_callback():
  from typing import TextIO
  import numpy as np
  from returnn.tensor import Tensor, TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config
  from returnn.datasets.hdf import SimpleHDFWriter

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.alignment_file: Optional[SimpleHDFWriter] = None

    def init(self, *, model):
      import gzip

      self.alignment_file = SimpleHDFWriter(
        filename=_v2_forward_out_alignment_filename, dim=model.align_target_dim.dimension, ndim=1
      )

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      viterbi_align: Tensor = outputs["viterbi_align"]  # [T]
      assert len(viterbi_align.dims) == 1, f"expected hyps to be 1D, but got {viterbi_align.dims}"
      assert viterbi_align.dims[0].dyn_size_ext, f"viterbi_align {viterbi_align} does not define seq lengths"

      seq_len = viterbi_align.dims[0].dyn_size_ext.raw_tensor.item()
      viterbi_align_raw = viterbi_align.raw_tensor[:seq_len]

      hdf.dump_hdf_numpy(
        hdf_dataset=self.alignment_file,
        data=viterbi_align_raw[None],  # [1, T]
        seq_lens=np.array([seq_len]),  # [1]
        seq_tags=[seq_tag],
      )

    def finish(self):
      self.alignment_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
