import os.path
from typing import Optional, Dict, List, Callable
import sys
import ast

import numpy as np
import torch
import matplotlib.pyplot as plt
import mpl_toolkits

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.encoder.conformer import ConformerEncoder, ConformerEncoderLayer
from returnn.frontend.attention import RelPosSelfAttention, dot_attention
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.encoder.ff import LinearEncoder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import TrafoAttention
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.realignment import ctc_align_get_center_positions
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.model import CtcModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import get_alignment_args
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence_efficient as forward_sequence_efficient_segmental,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence as forward_sequence_segmental,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import (
  forward_sequence as forward_sequence_global,
  get_s_and_att_efficient as get_s_and_att_efficient_global,
  get_s_and_att as get_s_and_att_global,
)
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.visualization.visualization import plot_att_weights

from .dump_att_weights import dump_hdfs, scatter_att_weights

from sisyphus import Path

fontsize_axis = 16
fontsize_ticks = 14


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


def dump_self_att(
        model: SegmentalAttentionModel,
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

  if isinstance(model, SegmentalAttentionModel):
    (
      segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim,
      non_blank_mask
    ) = get_alignment_args(
      model=model,
      align_targets=targets,
      align_targets_spatial_dim=targets_spatial_dim,
      batch_dims=batch_dims,
    )
    center_positions_hdf = None  # set below
    segment_starts_hdf = None  # set below
    segment_lens_hdf = None  # set below
  else:
    segment_starts = None
    segment_lens = None
    center_positions = None
    non_blank_targets = targets
    non_blank_targets_spatial_dim = targets_spatial_dim
    center_positions_hdf = None
    segment_starts_hdf = None
    segment_lens_hdf = None

  if isinstance(model, CtcModel):
    target_blank_idx = model.blank_idx
  elif isinstance(model.label_decoder, SegmentalAttLabelDecoder):
    target_blank_idx = model.blank_idx
  else:
    target_blank_idx = None

  # ------------------- run encoder and capture encoder input -----------------
  enc_layer_cls = type(model.encoder.layers[0])
  if isinstance(model.encoder, ConformerEncoder):
    self_att_cls = type(model.encoder.layers[0].self_att)
  else:
    # just use some dummy value, it is anyway not used in this case
    self_att_cls = RelPosSelfAttention
  set_trace_variables(
    funcs_to_trace_list=[
      model.encoder.encode,
      type(model.encoder).__call__,
      enc_layer_cls.__call__,
      self_att_cls.__call__,
      self_att_cls.attention,
      dot_attention,
      rf.Sequential.__call__,
    ]
  )

  collected_outputs = {}
  sys.settrace(_trace_func)
  enc_args, enc_spatial_dim = model.encoder.encode(
    data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
  sys.settrace(None)

  enc_key_dim = enc_spatial_dim.copy()
  dummy_att_head_dim = rf.Dim(1, name="att_head")

  energies_single_head_list = []
  for enc_layer_idx in range(9, 10):
    print(f"Processing encoder layer {enc_layer_idx}")

    if self_att_cls is RelPosSelfAttention:
      enc_att_weights = process_captured_tensors(
        layer_mapping={"att_weights": (self_att_cls.__call__, enc_layer_idx, "att_weights", -1)},
      )
      enc_att_energies = process_captured_tensors(
        layer_mapping={"energies": (self_att_cls.__call__, enc_layer_idx, "scores", -1)},
      )
    else:
      enc_att_weights = process_captured_tensors(
        layer_mapping={"att_weights": (dot_attention, enc_layer_idx, "att_weights", -1)},
      )
      enc_att_energies = process_captured_tensors(
        layer_mapping={"energies": (dot_attention, enc_layer_idx, "energy", -1)},
      )

    assert isinstance(enc_att_energies, rf.Tensor)
    assert isinstance(enc_att_weights, rf.Tensor)


    enc_att_head_dim = enc_att_weights.remaining_dims(batch_dims + enc_att_weights.get_dyn_size_tags())[0]
    enc_kv_dim = enc_att_weights.remaining_dims(batch_dims + [enc_att_head_dim, enc_spatial_dim])[0]

    for head in range(enc_att_head_dim.dimension):
      energies_single_head = rf.gather(
        enc_att_energies,
        indices=rf.constant(head, dims=batch_dims),
        axis=enc_att_head_dim,
      )

      energies_single_head = rf.expand_dim(energies_single_head, dim=dummy_att_head_dim)
      energies_single_head = utils.copy_tensor_replace_dim_tag(energies_single_head, enc_kv_dim, enc_key_dim)
      energies_single_head = energies_single_head.copy_transpose(
        batch_dims + [enc_spatial_dim, enc_key_dim, dummy_att_head_dim])

      energies_single_head_list.append(energies_single_head)

  energies_stacked_over_heads, stack_dim = rf.stack(energies_single_head_list)
  energies_stacked_mean = rf.reduce_mean(energies_stacked_over_heads, axis=stack_dim)
  energies_single_head_list.append(energies_stacked_mean)

  return energies_single_head_list, enc_spatial_dim, enc_key_dim, dummy_att_head_dim


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
  dump_self_att_def = config.typed_value("_dump_self_att_def")

  energies_list, query_dim, kv_dim, singleton_dim = dump_self_att_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    targets=align_targets,
    targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )

  for i, energies in enumerate(energies_list):
    rf.get_run_ctx().mark_as_output(
      energies,
      f"energies_head-{i}",
      dims=[batch_dim, query_dim, kv_dim, singleton_dim]
    )


def _returnn_v2_get_forward_callback():
  from returnn.tensor import TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      self.num_heads = 9
      self.energies_hdfs: List[Optional[SimpleHDFWriter]] = [None] * self.num_heads

    def init(self, *, model):
      self.energies_hdfs = [
        SimpleHDFWriter(filename=f"self-att-energies_head-{i}.hdf", dim=1, ndim=3)
        for i in range(self.num_heads)
      ]

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      for i, (tensor_name, energies) in enumerate(outputs.data.items()):
        # energies: rf.Tensor = outputs["energies"]

        query_dim = energies.dims[0]
        query_len = query_dim.dyn_size_ext.raw_tensor
        kv_dim = energies.dims[1]
        kv_len = kv_dim.dyn_size_ext.raw_tensor

        energies.raw_tensor = energies.raw_tensor[:query_len, :kv_len]

        hdf.dump_hdf_rf(
          hdf_dataset=self.energies_hdfs[i],
          data=energies,
          batch_dim=None,
          seq_tags=[seq_tag],
        )

    def finish(self):
      for energies_hdf in self.energies_hdfs:
        energies_hdf.close()

  return _ReturnnRecogV2ForwardCallbackIface()
