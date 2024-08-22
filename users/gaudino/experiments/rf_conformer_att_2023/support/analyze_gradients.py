import os.path
from typing import Optional, Dict, List, Callable
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer
from returnn.frontend.attention import RelPosSelfAttention
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder,
)

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.conformer_import_moh_att_train import Model

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import (
  forward_sequence as forward_sequence_global,
  get_s_and_att_efficient as get_s_and_att_efficient_global,
  get_s_and_att as get_s_and_att_global,
)
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.visualization.visualization import plot_att_weights

from i6_experiments.users.gaudino.experiments.rf_conformer_att_2023.librispeech_960.model_recogs.model_forward import model_forward

from .dump_att_weights import dump_hdfs, scatter_att_weights

from sisyphus import Path


def _get_scattered_grad_tensor(
        grad_list: List[torch.Tensor],
        att_weights: rf.Tensor,
        name: str,
        segment_starts: rf.Tensor,
        new_slice_dim: rf.Dim,
        align_targets_spatial_dim: rf.Dim,
):
  grads_raw = torch.stack(grad_list, dim=1)[None, :, :, None]
  grads_rf = att_weights.copy_template(name=name)
  grads_rf.raw_tensor = grads_raw

  grads_rf = scatter_att_weights(
    att_weights=grads_rf,
    segment_starts=segment_starts,
    new_slice_dim=new_slice_dim,
    align_targets_spatial_dim=align_targets_spatial_dim,
  )

  return grads_rf


def analyze_weights(model: SegmentalAttentionModel):
  s_size = model.label_decoder.get_lstm().out_dim.dimension
  embed_size = model.label_decoder.target_embed.out_dim.dimension
  enc_dim = model.encoder.out_dim.dimension

  readout_in_weights = model.label_decoder.readout_in_w_current_frame.weight.raw_tensor

  readout_in_att_weights = readout_in_weights[s_size + embed_size:s_size + embed_size + enc_dim]
  readout_in_cur_frame_weights = readout_in_weights[s_size + embed_size + enc_dim:]

  print("readout_in_att_weights mean", torch.mean(torch.abs(readout_in_att_weights)))
  print("readout_in_cur_frame_weights mean", torch.mean(torch.abs(readout_in_cur_frame_weights)))
  exit()


code_obj_to_func = None
captured_tensors = None  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor


def _trace_func(frame, event, arg):
  """
  Trace func to get intermediate outputs.
  """
  func = code_obj_to_func.get(frame.f_code)
  if func:
    if event == "call":
      captured_tensors.setdefault(func, []).append({})
    else:
      for k, v in frame.f_locals.items():
        if not isinstance(v, rf.Tensor):
          continue
        prev = captured_tensors[func][-1].get(k, None)
        if prev is None or prev[-1] is not v:
          # print(f"{func.__qualname__} tensor var changed: {k} = {v}")
          captured_tensors[func][-1].setdefault(k, []).append(v)
    return _trace_func


def copy_lower_triangular_matrix_to_upper(matrix: torch.Tensor):
  # Get the indices of the lower triangular part (excluding the diagonal)
  lower_tri_indices = torch.tril_indices(row=matrix.size(0), col=matrix.size(1), offset=-1)
  # Get the values from the lower triangular part (excluding the diagonal)
  lower_tri_values = matrix[lower_tri_indices[0], lower_tri_indices[1]]
  # Swap the indices to get the positions for the upper triangular part
  upper_tri_indices = lower_tri_indices.flip(0)
  # Create a copy of the original matrix to keep it unchanged
  full_matrix = matrix.clone()
  # Place the values in the corresponding upper triangular positions
  full_matrix[upper_tri_indices[0], upper_tri_indices[1]] = lower_tri_values

  return full_matrix

def set_trace_variables(
        funcs_to_trace_list: List,
):
  funcs_to_trace_list = funcs_to_trace_list
  global code_obj_to_func
  global captured_tensors
  code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
  captured_tensors = {}


def process_captured_tensors(
        layer_mapping: Dict,
        process_func: Optional[Callable] = None
):
  tensor_list = []

  for tensor_name, var_path in list(layer_mapping.items()):
    new_out = captured_tensors
    for k in var_path:
      new_out = new_out[k]

    tensor_list.append(process_func(new_out) if process_func else new_out)
    del layer_mapping[tensor_name]

  return tensor_list if len(tensor_list) > 1 else tensor_list[0]


def analyze_gradients(
        model: Model,
        data: rf.Tensor,
        data_spatial_dim: rf.Dim,
        targets: rf.Tensor,
        targets_spatial_dim: rf.Dim,
        seq_tags: rf.Tensor,
):
  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio
  batch_dims = data.remaining_dims(data_spatial_dim)

  config = get_global_config()

  # start by analyzing the weights
  # analyze_weights(model)

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

  non_blank_targets = targets
  non_blank_targets_spatial_dim = targets_spatial_dim
  center_positions_hdf = None
  segment_starts_hdf = None
  segment_lens_hdf = None

  target_blank_idx = None

  max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

  ref_alignment_hdf = Path(config.typed_value("ref_alignment_hdf", str))
  ref_alignment_blank_idx = config.typed_value("ref_alignment_blank_idx", int)
  ref_alignment_vocab_path = Path(config.typed_value("ref_alignment_vocab_path", str))
  json_vocab_path = Path(config.typed_value("json_vocab_path", str))

  with torch.enable_grad():
    # ----------------------------------------------------------------------------------

    dump_hdfs(
      batch_dims=batch_dims,
      seq_tags=seq_tags,
      align_targets=non_blank_targets,
      align_target_dim=model.target_dim.dimension
    )

    set_trace_variables(
      funcs_to_trace_list=[
        forward_sequence_global,
        get_s_and_att_global,
        GlobalAttDecoder.loop_step,
        GlobalAttDecoder.decode_logits,
      ]
    )

    sys.settrace(_trace_func)
    model_forward(
      model=model,
      data=data,
      data_spatial_dim=data_spatial_dim,
      ground_truth=non_blank_targets,
    )
    sys.settrace(None)

    att_weights = process_captured_tensors(
      layer_mapping={
        f"att_weight_step{i}": (GlobalAttDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)},
    )
    att_weights, _ = rf.stack(att_weights, out_dim=non_blank_targets_spatial_dim)

    energy_in = process_captured_tensors(
      layer_mapping={
        f"energy_step{i}": (GlobalAttDecoder.loop_step, i, "energy_in", -1) for i in range(max_num_labels)},
    )
    energy_in, _ = rf.stack(energy_in, out_dim=non_blank_targets_spatial_dim)

    energies = process_captured_tensors(
      layer_mapping={
        f"energy_step{i}": (GlobalAttDecoder.loop_step, i, "energy", -1) for i in range(max_num_labels)},
    )
    energies, _ = rf.stack(energies, out_dim=non_blank_targets_spatial_dim)

    label_model_states = process_captured_tensors(
      layer_mapping={
        f"s_step{i}": (GlobalAttDecoder.loop_step, i, "s", -1) for i in range(max_num_labels)},
    )
    label_model_states, _ = rf.stack(label_model_states, out_dim=non_blank_targets_spatial_dim)
    label_model_states.feature_dim = label_model_states.remaining_dims(
      batch_dims + [non_blank_targets_spatial_dim])[0]

    logits = process_captured_tensors(
      layer_mapping={"logits": (GlobalAttDecoder.decode_logits, 0, "logits", -1)}
    )
    assert isinstance(logits, rf.Tensor)
    log_probs = rf.log_softmax(logits, axis=model.target_dim)
    log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

      # shapes
      # print("att_weights", att_weights)  # [B, T, S, 1]
      # print("logits", log_probs)  # [B, S, V]
      # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]


    att_weight_tensor_list = [
      (att_weights, "att_weights"),
      (energies, "energies"),
    ]
    if config.bool("analyze_att_weight_dropout", False):
      att_weights_drop01 = rf.dropout(att_weights, 0.1, on_forward=True, axis=batch_dims + [enc_spatial_dim])
      att_weight_tensor_list.append((att_weights_drop01, "att_weights_drop01_broadcast_to_labels"))

    # plot attention weights for non-trafo decoders
    # (for trafo decoders, the attention weights are plotted in the loop above)

    dirname = f"cross-att/enc-layer-{12}"

    for tensor, tensor_name in att_weight_tensor_list:
      if energies is None:
        continue

      tensor_dirname = os.path.join(dirname, tensor_name)
      if os.path.exists(tensor_dirname):
        continue

      assert non_blank_targets_spatial_dim in tensor.dims
      if tensor.dims != tuple(batch_dims + [non_blank_targets_spatial_dim, enc_spatial_dim, model.label_decoder.att_num_heads]):
        tensor_transposed = tensor.copy_transpose(
          batch_dims + [non_blank_targets_spatial_dim, enc_spatial_dim, model.label_decoder.att_num_heads])
      else:
        tensor_transposed = tensor

      dump_hdfs(
        att_weights=tensor_transposed,
        batch_dims=batch_dims,
        dirname=tensor_dirname,
        seq_tags=seq_tags,
      )

      plot_att_weights(
        att_weight_hdf=Path(os.path.join(tensor_dirname, "att_weights.hdf")),
        targets_hdf=Path("targets.hdf"),
        seg_starts_hdf=segment_starts_hdf,
        seg_lens_hdf=segment_lens_hdf,
        center_positions_hdf=center_positions_hdf,
        target_blank_idx=target_blank_idx,
        ref_alignment_blank_idx=ref_alignment_blank_idx,
        ref_alignment_hdf=ref_alignment_hdf,
        ref_alignment_json_vocab_path=ref_alignment_vocab_path,
        json_vocab_path=json_vocab_path,
        segment_whitelist=list(seq_tags.raw_tensor),
        plot_name=tensor_dirname
      )

      # _info_mixing_analysis(
      #   input_=x_linear,
      #   representation=enc_args["enc"],
      #   log_probs=log_probs,
      #   targets=non_blank_targets,
      #   batch_dims=batch_dims,
      #   seq_tags=seq_tags,
      #   enc_spatial_dim=enc_spatial_dim,
      #   targets_spatial_dim=non_blank_targets_spatial_dim,
      #   input_name="enc-12",
      #   json_vocab_path=json_vocab_path
      # )


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
  default_target_key = config.typed_value("target")
  align_targets = extern_data[default_target_key]
  align_targets_spatial_dim = align_targets.get_time_dim_tag()
  analyze_gradients_def = config.typed_value("_analyze_gradients_def")

  analyze_gradients_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=align_targets,
    targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )

  # analyze_att_energy_gradients(
  #   model=model,
  #   data=data,
  #   data_spatial_dim=data_spatial_dim,
  #   align_targets=align_targets,
  #   align_targets_spatial_dim=align_targets_spatial_dim,
  #   seq_tags=extern_data["seq_tag"],
  # )


def _returnn_v2_get_forward_callback():
  from returnn.tensor import TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      pass

    def init(self, *, model):
      pass

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      pass

    def finish(self):
      pass

  return _ReturnnRecogV2ForwardCallbackIface()
