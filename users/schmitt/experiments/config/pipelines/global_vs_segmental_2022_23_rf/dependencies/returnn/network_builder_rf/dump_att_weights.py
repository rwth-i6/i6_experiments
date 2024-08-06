import os.path
from typing import Optional, Dict, List
import sys

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.decoder import (
  GlobalAttDecoder,
  GlobalAttEfficientDecoder
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import forward_sequence as forward_sequence_global
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import get_alignment_args
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence, forward_sequence_efficient
)
from i6_experiments.users.schmitt import hdf


def dump_hdfs(
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        att_weights: Optional[rf.Tensor] = None,
        center_positions: Optional[rf.Tensor] = None,
        segment_lens: Optional[rf.Tensor] = None,
        segment_starts: Optional[rf.Tensor] = None,
        align_targets: Optional[rf.Tensor] = None,
        align_target_dim: Optional[int] = None,
        dirname: Optional[str] = None,
):
  assert len(batch_dims) == 1
  assert (align_target_dim is None) == (align_targets is None), "align_target_dim is required if align_targets is given"
  if dirname is not None and not os.path.exists(dirname):
    os.makedirs(dirname)

  for tensor_name, tensor, dim, ndim in (
          ("att_weights", att_weights, 1, 3),
          ("center_positions", center_positions, 1, 1),
          ("seg_lens", segment_lens, 1, 1),
          ("seg_starts", segment_starts, 1, 1),
          ("targets", align_targets, align_target_dim, 1),
  ):
    if tensor is None:
      continue

    filename = f"{tensor_name}.hdf"
    if dirname is not None:
      filename = os.path.join(dirname, filename)

    hdf_dataset = SimpleHDFWriter(
      filename=filename,
      dim=dim,
      ndim=ndim,
      extend_existing_file=os.path.exists(filename)
    )
    hdf.dump_hdf_rf(
      hdf_dataset=hdf_dataset,
      data=tensor,
      batch_dim=batch_dims[0],
      seq_tags=seq_tags,
    )
    hdf_dataset.close()


def scatter_att_weights(
        att_weights: rf.Tensor,
        segment_starts: rf.Tensor,
        new_slice_dim: rf.Dim,
        align_targets_spatial_dim: rf.Dim,
        batch_dims: List[rf.Dim],
):
  # scatter the attention weights to the align_targets_spatial_dim to get shape [T, S]
  scatter_indices = segment_starts + rf.range_over_dim(new_slice_dim)
  align_targets_spatial_sizes = rf.copy_to_device(
    align_targets_spatial_dim.dyn_size_ext, device=att_weights.device
  )
  att_weights = utils.scatter_w_masked_indices(
    x=att_weights,
    mask=scatter_indices < align_targets_spatial_sizes,
    scatter_indices=scatter_indices,
    result_spatial_dim=align_targets_spatial_dim,
    indices_dim=new_slice_dim,
    batch_dims=batch_dims
  )

  return att_weights


def dump_att_weights(
        model: SegmentalAttentionModel,
        data: rf.Tensor,
        data_spatial_dim: rf.Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: rf.Dim,
        seq_tags: rf.Tensor,
):
  if data.feature_dim and data.feature_dim.dimension == 1:
    data = rf.squeeze(data, axis=data.feature_dim)
  assert not data.feature_dim  # raw audio

  batch_dims = data.remaining_dims(data_spatial_dim)

  enc_args, enc_spatial_dim = model.encoder.encode(data, in_spatial_dim=data_spatial_dim)

  max_num_frames = rf.reduce_max(enc_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

  code_obj_to_func = {}
  captured_tensors = {}  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor

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
            print(f"{func.__qualname__} tensor var changed: {k} = {v}")
            captured_tensors[func][-1].setdefault(k, []).append(v)
      return _trace_func

  if isinstance(model, SegmentalAttentionModel):
    (
      segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim, non_blank_mask
    ) = get_alignment_args(
      model=model,
      align_targets=align_targets,
      align_targets_spatial_dim=align_targets_spatial_dim,
      batch_dims=batch_dims,
    )
    max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

    if model.center_window_size is not None:
      # we want a static slice dim
      # either use window size of model or encoder len
      new_slice_dim = rf.Dim(min(model.center_window_size, max_num_frames), name="att_window")
    else:
      new_slice_dim = None
    if isinstance(model.label_decoder, SegmentalAttLabelDecoder):
      if model.use_joint_model:
        if type(model.label_decoder) is SegmentalAttLabelDecoder:
          assert model.label_decoder_state == "joint-lstm", "not implemented yet, simple to extend"
          funcs_to_trace_list = [
            forward_sequence,
            SegmentalAttLabelDecoder.loop_step,
          ]
          code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
          _layer_mapping = {
            f"att_weight_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_frames)
          }

          targets = align_targets
          targets_spatial_dim = align_targets_spatial_dim

          sys.settrace(_trace_func)
          forward_sequence(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            non_blank_targets=targets,
            non_blank_targets_spatial_dim=targets_spatial_dim,
            segment_starts=segment_starts,
            segment_lens=segment_lens,
            center_positions=center_positions,
            batch_dims=batch_dims,
          )
          sys.settrace(None)
          att_weights_ = []
          for tensor_name, var_path in _layer_mapping.items():
            new_out = captured_tensors
            for k in var_path:
              new_out = new_out[k]
            old_slice_dim = new_out.remaining_dims(batch_dims + [model.label_decoder.att_num_heads])
            assert len(old_slice_dim) == 1
            old_slice_dim = old_slice_dim[0]
            max_slice_size = rf.reduce_max(old_slice_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

            # at the end of the seq, there may be less than center_window_size attention weights
            # in this case, just pad with zeros until we reach center_window_size
            if max_slice_size < model.center_window_size:
              new_out, _ = rf.pad(
                new_out,
                axes=[old_slice_dim],
                padding=[(0, model.center_window_size - max_slice_size)],
                out_dims=[new_slice_dim],
                value=0.0,
              )
            else:
              new_out = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)

            att_weights_.append(new_out)

          att_weights__ = TensorArray(att_weights_[0])
          for target in att_weights_:
            att_weights__ = att_weights__.push_back(target)
          att_weights = att_weights__.stack(axis=align_targets_spatial_dim)
        else:
          assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
          if model.label_decoder_state == "joint-lstm":
            targets = align_targets
            targets_spatial_dim = align_targets_spatial_dim
            non_blank_mask_ = None
            non_blank_mask_spatial_dim = None
          else:
            targets = non_blank_targets
            targets_spatial_dim = non_blank_targets_spatial_dim
            non_blank_mask_ = non_blank_mask
            non_blank_mask_spatial_dim = align_targets_spatial_dim

          funcs_to_trace_list = [
            forward_sequence_efficient,
            SegmentalAttEfficientLabelDecoder.__call__,
          ]
          code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
          _layer_mapping = {"att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1)}

          sys.settrace(_trace_func)
          forward_sequence_efficient(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=targets,
            targets_spatial_dim=targets_spatial_dim,
            segment_starts=segment_starts,
            segment_lens=segment_lens,
            center_positions=center_positions,
            batch_dims=batch_dims,
            non_blank_mask=non_blank_mask_,
            non_blank_mask_spatial_dim=non_blank_mask_spatial_dim,
          )
          sys.settrace(None)
          for tensor_name, var_path in _layer_mapping.items():
            new_out = captured_tensors
            for k in var_path:
              new_out = new_out[k]
            old_slice_dim = new_out.remaining_dims(batch_dims + [model.label_decoder.att_num_heads, align_targets_spatial_dim])
            att_weights = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)
      else:
        if type(model.label_decoder) is SegmentalAttLabelDecoder:
          funcs_to_trace_list = [
            forward_sequence,
            SegmentalAttLabelDecoder.loop_step,
          ]
          code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
          _layer_mapping = {
            f"att_weight_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)
          }

          sys.settrace(_trace_func)
          forward_sequence(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            non_blank_targets=non_blank_targets,
            non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
            segment_starts=segment_starts,
            segment_lens=segment_lens,
            center_positions=center_positions,
            batch_dims=batch_dims,
          )
          sys.settrace(None)
          att_weights_ = []
          for tensor_name, var_path in _layer_mapping.items():
            new_out = captured_tensors
            for k in var_path:
              new_out = new_out[k]
            old_slice_dim = new_out.remaining_dims(batch_dims + [model.label_decoder.att_num_heads])
            assert len(old_slice_dim) == 1
            old_slice_dim = old_slice_dim[0]
            max_slice_size = rf.reduce_max(old_slice_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

            # at the end of the seq, there may be less than center_window_size attention weights
            # in this case, just pad with zeros until we reach center_window_size
            if max_slice_size < new_slice_dim.dimension:
              new_out, _ = rf.pad(
                new_out,
                axes=[old_slice_dim],
                padding=[(0, new_slice_dim.dimension - max_slice_size)],
                out_dims=[new_slice_dim],
                value=0.0,
              )
            else:
              new_out = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)

            att_weights_.append(new_out)

          att_weights__ = TensorArray(att_weights_[0])
          for target in att_weights_:
            att_weights__ = att_weights__.push_back(target)
          att_weights = att_weights__.stack(axis=non_blank_targets_spatial_dim)
        else:
          assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
          funcs_to_trace_list = [
            forward_sequence_efficient,
            SegmentalAttEfficientLabelDecoder.__call__,
          ]
          code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
          _layer_mapping = {"att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1)}

          sys.settrace(_trace_func)
          forward_sequence_efficient(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=non_blank_targets,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            segment_starts=segment_starts,
            segment_lens=segment_lens,
            center_positions=center_positions,
            batch_dims=batch_dims,
          )
          sys.settrace(None)
          for tensor_name, var_path in _layer_mapping.items():
            new_out = captured_tensors
            for k in var_path:
              new_out = new_out[k]
            old_slice_dim = new_out.remaining_dims(
              batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
            att_weights = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)

      # attention weights need to be transferred from slice dim to align_targets_spatial_dim (T)
      att_weights = scatter_att_weights(
        att_weights=att_weights,
        segment_starts=segment_starts,
        new_slice_dim=new_slice_dim,
        align_targets_spatial_dim=align_targets_spatial_dim,
      )
      dump_targets = align_targets
      dump_targets_dim = model.align_target_dim.dimension
    else:
      assert type(model.label_decoder) is GlobalAttEfficientDecoder, "not implemented yet"
      funcs_to_trace_list = [
        forward_sequence_global,
        GlobalAttEfficientDecoder.__call__,
      ]
      code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
      _layer_mapping = {"att_weights": (GlobalAttEfficientDecoder.__call__, 0, "att_weights", -1)}

      sys.settrace(_trace_func)
      forward_sequence_global(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=non_blank_targets,
        targets_spatial_dim=non_blank_targets_spatial_dim,
        batch_dims=batch_dims,
        center_positions=center_positions,
      )
      sys.settrace(None)
      for tensor_name, var_path in _layer_mapping.items():
        new_out = captured_tensors
        for k in var_path:
          new_out = new_out[k]
        att_weights = new_out.copy_transpose(
          batch_dims + [non_blank_targets_spatial_dim, enc_spatial_dim, model.label_decoder.att_num_heads])

      dump_targets = non_blank_targets
      dump_targets_dim = model.target_dim.dimension
  else:
    max_num_labels = rf.reduce_max(align_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()
    if type(model.label_decoder) is GlobalAttDecoder:
      funcs_to_trace_list = [
        forward_sequence_global,
        GlobalAttDecoder.loop_step,
      ]
      code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
      _layer_mapping = {
        f"att_weight_step{i}": (GlobalAttDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)
      }

      sys.settrace(_trace_func)
      forward_sequence_global(
        model=model.label_decoder,
        enc_args=enc_args,
        enc_spatial_dim=enc_spatial_dim,
        targets=align_targets,
        targets_spatial_dim=align_targets_spatial_dim,
        batch_dims=batch_dims,
      )
      sys.settrace(None)
      att_weights_ = []
      for tensor_name, var_path in _layer_mapping.items():
        new_out = captured_tensors
        for k in var_path:
          new_out = new_out[k]

        att_weights_.append(new_out)

      att_weights__ = TensorArray(att_weights_[0])
      for target in att_weights_:
        att_weights__ = att_weights__.push_back(target)
      att_weights = att_weights__.stack(axis=align_targets_spatial_dim)
      att_weights = att_weights.copy_transpose(batch_dims + [align_targets_spatial_dim, enc_spatial_dim, model.label_decoder.att_num_heads])

      dump_targets = align_targets
      dump_targets_dim = model.target_dim.dimension
      center_positions = None
      segment_lens = None
      segment_starts = None
    else:
      raise NotImplementedError

  dump_hdfs(
    att_weights=att_weights,
    center_positions=center_positions,
    segment_lens=segment_lens,
    segment_starts=segment_starts,
    align_targets=dump_targets,
    align_target_dim=dump_targets_dim,
    seq_tags=seq_tags,
    batch_dims=batch_dims,
  )


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
  dump_att_weight_def = config.typed_value("_dump_att_weight_def")

  dump_att_weight_out = dump_att_weight_def(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    align_targets=align_targets,
    align_targets_spatial_dim=align_targets_spatial_dim,
    seq_tags=extern_data["seq_tag"],
  )
  # assert len(dump_att_weight_out) == 7
  # assert all(isinstance(x, Tensor) for x in dump_att_weight_out[:5])
  # assert all(isinstance(x, Dim) for x in dump_att_weight_out[5:])
  # (
  #   att_weights,
  #   center_positions,
  #   segment_lens,
  #   segment_starts,
  #   align_targets,
  #   att_num_heads,
  #   non_blank_targets_spatial_dim,
  # ) = dump_att_weight_out

  # rf.get_run_ctx().mark_as_output(
  #   att_weights,
  #   "att_weights",
  #   dims=[batch_dim, non_blank_targets_spatial_dim, align_targets_spatial_dim, att_num_heads]
  # )
  # rf.get_run_ctx().mark_as_output(
  #   center_positions, "center_positions", dims=[batch_dim, non_blank_targets_spatial_dim])
  # rf.get_run_ctx().mark_as_output(
  #   segment_starts, "segment_starts", dims=[batch_dim, non_blank_targets_spatial_dim])
  # rf.get_run_ctx().mark_as_output(
  #   segment_lens, "segment_lens", dims=[batch_dim, non_blank_targets_spatial_dim])
  # rf.get_run_ctx().mark_as_output(
  #   align_targets, "align_targets", dims=[batch_dim, align_targets_spatial_dim])


def _returnn_v2_get_forward_callback():
  from returnn.tensor import TensorDict
  from returnn.forward_iface import ForwardCallbackIface
  from returnn.config import get_global_config

  class _ReturnnRecogV2ForwardCallbackIface(ForwardCallbackIface):
    def __init__(self):
      return
      # self.att_weights_hdf: Optional[SimpleHDFWriter] = None
      # self.center_positions_hdf: Optional[SimpleHDFWriter] = None
      # self.segment_lens_hdf: Optional[SimpleHDFWriter] = None
      # self.segment_starts_hdf: Optional[SimpleHDFWriter] = None
      # self.align_targets_hdf: Optional[SimpleHDFWriter] = None

    def init(self, *, model):
      return
      # self.att_weights_hdf = SimpleHDFWriter(filename="att_weights.hdf", dim=1, ndim=3)
      # self.center_positions_hdf = SimpleHDFWriter(filename="center_positions.hdf", dim=1, ndim=1)
      # self.segment_lens_hdf = SimpleHDFWriter(filename="seg_lens.hdf", dim=1, ndim=1)
      # self.segment_starts_hdf = SimpleHDFWriter(filename="seg_starts.hdf", dim=1, ndim=1)
      # self.align_targets_hdf = SimpleHDFWriter(
      #   filename="targets.hdf", dim=model.align_target_dim.dimension, ndim=1)

    def process_seq(self, *, seq_tag: str, outputs: TensorDict):
      return
      att_weights: rf.Tensor = outputs["att_weights"]

      for hdf_dataset, tensor, dim, ndim in (
              (self.att_weights_hdf, att_weights, 1, 3),
              # ("center_positions", center_positions, 1, 1),
              # ("segment_lens", segment_lens, 1, 1),
              # ("segment_starts", segment_starts, 1, 1),
              # ("targets", align_targets, model.align_target_dim.dimension, 1),
      ):
        hdf.dump_hdf_rf(
          hdf_dataset=hdf_dataset,
          data=tensor,
          batch_dim=None,
          seq_tags=[seq_tag],
        )
        hdf_dataset.close()

    def finish(self):
      return
      self.out_file.write("}\n")
      self.out_file.close()
      if self.out_ext_file:
        self.out_ext_file.write("}\n")
        self.out_ext_file.close()

  return _ReturnnRecogV2ForwardCallbackIface()
