import os.path
from typing import Optional, Dict, List
import sys

import torch
import matplotlib.pyplot as plt

from returnn.tensor import TensorDict
from returnn.frontend.tensor_array import TensorArray
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.encoder.conformer import ConformerEncoder

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import SegmentalAttentionModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.model import (
  SegmentalAttLabelDecoder,
  SegmentalAttEfficientLabelDecoder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import get_alignment_args
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.label_model.train import (
  forward_sequence, forward_sequence_efficient
)
from i6_experiments.users.schmitt import hdf
from i6_experiments.users.schmitt.visualization.visualization import plot_att_weights

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


def analyze_gradients():
  pass


global code_obj_to_func
global captured_tensors  # func -> (list of calls) -> tensor local name -> (list of versions) -> tensor


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


def analyze_att_energy_gradients(
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

  config = get_global_config()

  # start by analyzing the weights
  # analyze_weights(model)

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

  # enable gradients
  with torch.enable_grad():
    batch_dims = data.remaining_dims(data_spatial_dim)
    (
      segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim, non_blank_mask
    ) = get_alignment_args(
      model=model,
      align_targets=align_targets,
      align_targets_spatial_dim=align_targets_spatial_dim,
      batch_dims=batch_dims,
    )

    # ------------------- run encoder and capture encoder input -----------------
    funcs_to_trace_list = [
      model.encoder.encode,
      ConformerEncoder.__call__
    ]
    global code_obj_to_func
    code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
    _layer_mapping = {
      "x_linear": (ConformerEncoder.__call__, 0, "x_linear", -1),
    }

    sys.settrace(_trace_func)
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim)
    sys.settrace(None)

    for tensor_name, var_path in _layer_mapping.items():
      new_out = captured_tensors
      for k in var_path:
        new_out = new_out[k]

      x_linear = new_out  # type: rf.Tensor

    # ----------------------------------------------------------------------------------

    max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()
    max_num_frames = rf.reduce_max(enc_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

    new_slice_dim = rf.Dim(min(model.center_window_size, max_num_frames), name="att_window")
    if model.use_joint_model:
      if type(model.label_decoder) is SegmentalAttLabelDecoder:
        assert model.label_decoder_state == "joint-lstm", "not implemented yet, simple to extend"
        raise NotImplementedError
      else:
        assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
        raise NotImplementedError
    else:
      if type(model.label_decoder) is SegmentalAttLabelDecoder:
        raise NotImplementedError
      else:
        funcs_to_trace_list = [
          forward_sequence_efficient,
          SegmentalAttEfficientLabelDecoder.__call__,
          SegmentalAttLabelDecoder.decode_logits,
        ]
        code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
        _layer_mapping = {
          "att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1),
          "energy": (SegmentalAttEfficientLabelDecoder.__call__, 0, "energy", -1),
          "logits": (SegmentalAttLabelDecoder.decode_logits, 0, "logits", -1),
        }

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

        att_weights = None
        logits = None

        for tensor_name, var_path in _layer_mapping.items():
          new_out = captured_tensors
          for k in var_path:
            new_out = new_out[k]

          if tensor_name == "att_weights":
            old_slice_dim = new_out.remaining_dims(
              batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
            att_weights = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)
          elif tensor_name == "energy":
            old_slice_dim = new_out.remaining_dims(
              batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
            energies = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)
          else:
            assert tensor_name == "logits"
            logits = new_out  # type: rf.Tensor


        # print("att_weights", att_weights)  # [B, T, S, 1]
        # print("logits", logits)  # [B, S, V]
        # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
        # print("x_linear", x_linear)  # [B, T, F]

        att_weights_raw = att_weights.raw_tensor
        energies_raw = energies.raw_tensor
        x_linear_raw = x_linear.raw_tensor

        b = 0
        s = 0
        S = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor[b].item()  # noqa

        energies_raw.retain_grad()
        x_linear_raw.retain_grad()

        t_max = 100
        t_step = 10
        num_steps = t_max // t_step

        for s in range(S):
          fig, axs = plt.subplots(num_steps, 1, figsize=(num_steps, 20))  # Create num_step subplots stacked vertically

          for i, t in enumerate(range(0, t_max, t_step)):
            energies_raw[b, t, s, 0].backward(retain_graph=True)

            x_linear_grad = x_linear_raw.grad[b]
            x_linear_grad = torch.mean(torch.abs(x_linear_grad), dim=1)

            axs[i].plot(x_linear_grad.detach().cpu().numpy())  # Plot on the i-th subplot
            axs[i].set_title(f"Gradient at t={t}")  # Optional: Set title for each subplot

          plt.tight_layout()  # Adjust layout to not overlap
          plt.savefig(f"x_linear_grads_s-{s}_t-{t_max}_step-{t_step}.png")
          plt.close()
        exit()


def analyze_log_prob_gradients(
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

  config = get_global_config()

  # start by analyzing the weights
  # analyze_weights(model)

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

  with torch.enable_grad():
    batch_dims = data.remaining_dims(data_spatial_dim)
    (
      segment_starts, segment_lens, center_positions, non_blank_targets, non_blank_targets_spatial_dim, non_blank_mask
    ) = get_alignment_args(
      model=model,
      align_targets=align_targets,
      align_targets_spatial_dim=align_targets_spatial_dim,
      batch_dims=batch_dims,
    )

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)

    for enc_layer_idx in range(12):
      enc = collected_outputs[str(enc_layer_idx)]
      enc_args = {
        "enc": enc,
        "enc_ctx": model.encoder.enc_ctx(enc),
      }

      max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()
      max_num_frames = rf.reduce_max(enc_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

      new_slice_dim = rf.Dim(min(model.center_window_size, max_num_frames), name="att_window")
      if model.use_joint_model:
        if type(model.label_decoder) is SegmentalAttLabelDecoder:
          assert model.label_decoder_state == "joint-lstm", "not implemented yet, simple to extend"
          raise NotImplementedError
        else:
          assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder
          raise NotImplementedError
      else:
        if type(model.label_decoder) is SegmentalAttLabelDecoder:
          raise NotImplementedError
        else:
          funcs_to_trace_list = [
            forward_sequence_efficient,
            SegmentalAttEfficientLabelDecoder.__call__,
            SegmentalAttLabelDecoder.decode_logits,
          ]
          global code_obj_to_func
          global captured_tensors
          code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
          captured_tensors = {}
          _layer_mapping = {
            "att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1),
            "energy": (SegmentalAttEfficientLabelDecoder.__call__, 0, "energy", -1),
            "logits": (SegmentalAttLabelDecoder.decode_logits, 0, "logits", -1),
          }

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

          att_weights = None
          logits = None

          for tensor_name, var_path in _layer_mapping.items():
            new_out = captured_tensors
            for k in var_path:
              new_out = new_out[k]

            if tensor_name == "att_weights":
              old_slice_dim = new_out.remaining_dims(
                batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
              att_weights = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)
            elif tensor_name == "energy":
              old_slice_dim = new_out.remaining_dims(
                batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
              energies = utils.copy_tensor_replace_dim_tag(new_out, old_slice_dim, new_slice_dim)
            else:
              assert tensor_name == "logits"
              logits = new_out  # type: rf.Tensor

          print("\n\n")

          print("seq_tags", seq_tags.raw_tensor)

          print("\n")

          # print("att_weights", att_weights)  # [B, T, S, 1]
          # print("logits", logits)  # [B, S, V]
          # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

          log_probs = rf.log_softmax(logits, axis=model.target_dim)

          att_weights_raw = att_weights.raw_tensor
          energies_raw = energies.raw_tensor
          log_probs_raw = log_probs.raw_tensor  # type: torch.Tensor

          b = 0
          S = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor[b].item()  # noqa

          # for each target label, compute gradient of log-prob with respect to energies and att weights
          # and append to list, which will be stacked later
          att_weights_grads_list = []
          energies_grads_list = []
          for s in range(S):
            target_idx = non_blank_targets.raw_tensor[b, s]

            att_weights_raw.retain_grad()
            energies_raw.retain_grad()
            log_probs_raw.retain_grad()
            log_probs_raw[b, s, target_idx].backward(retain_graph=True)

            att_weights_grads_list.append(att_weights_raw.grad[b, :, s, 0])
            energies_grads_list.append(energies_raw.grad[b, :, s, 0])

          # stack att weight gradients
          att_weights_grads_raw = torch.stack(att_weights_grads_list, dim=1)[None, :, :, None]
          att_weights_grads = att_weights.copy_template(name="att_weights_grads")
          att_weights_grads.raw_tensor = att_weights_grads_raw

          # stack energies gradients
          energies_grads_raw = torch.stack(energies_grads_list, dim=1)[None, :, :, None]
          energies_grads = att_weights.copy_template(name="energies_grads")
          energies_grads.raw_tensor = energies_grads_raw

          # ------------------ plot gradients -----------------
          for tensor, name in (
                  (att_weights, "att_weights"),
                  (att_weights_grads, "att_weights_grads"),
                  (energies_grads, "energies_grads"),
          ):
            scattered_tensor = scatter_att_weights(
              att_weights=tensor,
              segment_starts=segment_starts,
              new_slice_dim=new_slice_dim,
              align_targets_spatial_dim=align_targets_spatial_dim,
            )

            dirname = f"encoder-layer-{enc_layer_idx + 1}"
            hdf_dirname = os.path.join(dirname, f"{name}_hdfs")
            plot_dirname = os.path.join(dirname, f"{name}_plots")
            # if not os.path.exists(hdf_dirname):
            #   os.makedirs(hdf_dirname)

            dump_hdfs(
              att_weights=scattered_tensor,
              center_positions=center_positions,
              segment_lens=segment_lens,
              segment_starts=segment_starts,
              align_targets=align_targets,
              seq_tags=seq_tags,
              batch_dims=batch_dims,
              align_target_dim=model.align_target_dim.dimension,
              dirname=hdf_dirname
            )

            plot_att_weights(
              att_weight_hdf=Path(os.path.join(hdf_dirname, "att_weights.hdf")),
              targets_hdf=Path(os.path.join(hdf_dirname, "targets.hdf")),
              seg_starts_hdf=Path(os.path.join(hdf_dirname, "seg_starts.hdf")),
              seg_lens_hdf=Path(os.path.join(hdf_dirname, "seg_lens.hdf")),
              center_positions_hdf=Path(os.path.join(hdf_dirname, "center_positions.hdf")),
              target_blank_idx=model.blank_idx,
              ref_alignment_blank_idx=model.blank_idx,
              ref_alignment_hdf=Path(os.path.join(hdf_dirname, "targets.hdf")),
              json_vocab_path=Path(config.typed_value("forward_data")["datasets"]["zip_dataset"]["targets"]["vocab_file"]),
              segment_whitelist=list(seq_tags.raw_tensor),
              plot_name=plot_dirname
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
  analyze_gradients_def = config.typed_value("_analyze_gradients_def")

  analyze_log_prob_gradients(
    model=model,
    data=data,
    data_spatial_dim=data_spatial_dim,
    align_targets=align_targets,
    align_targets_spatial_dim=align_targets_spatial_dim,
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
