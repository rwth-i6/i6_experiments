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


def _plot_log_prob_gradient_wrt_to_input(
        input_: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        input_name: str,
        json_vocab_path: Path,
        ref_alignment_hdf: Path,
        ref_alignment_blank_idx: int,
        ref_alignment_json_vocab_path: Path,
):
  dirname = f"log-prob-grads_wrt_{input_name}_log-space"
  if os.path.exists(dirname):
    return

  input_raw = input_.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa

  batch_gradients = []
  for b in range(B):
    s_gradients = []
    for s in range(S):
      v = targets.raw_tensor[b, s]
      input_raw.retain_grad()
      log_probs_raw.retain_grad()
      log_probs_raw[b, s, v].backward(retain_graph=True)

      x_linear_grad_l2 = torch.linalg.vector_norm(input_raw.grad[b], dim=-1)
      s_gradients.append(x_linear_grad_l2)

      # zero grad before next step
      input_raw.grad.zero_()

    batch_gradients.append(torch.stack(s_gradients, dim=0))
  x_linear_grad_l2_raw = torch.stack(batch_gradients, dim=0)[:, :, :, None]
  x_linear_grad_l2 = rf.convert_to_tensor(
    x_linear_grad_l2_raw,
    dims=batch_dims + [targets_spatial_dim, enc_spatial_dim, rf.Dim(1, name="dummy")],
  )
  x_linear_grad_l2 = rf.log(x_linear_grad_l2)  # use log for better visualization of small values

  dump_hdfs(
    att_weights=x_linear_grad_l2,
    batch_dims=batch_dims,
    dirname=dirname,
    seq_tags=seq_tags,
  )

  plot_att_weights(
    att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
    targets_hdf=Path("targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    ref_alignment_hdf=ref_alignment_hdf,
    ref_alignment_json_vocab_path=ref_alignment_json_vocab_path,
    json_vocab_path=json_vocab_path,
    segment_whitelist=list(seq_tags.raw_tensor),
    plot_name=dirname
  )


def _plot_cosine_sim_matrix(
        x: rf.Tensor,
        input_name: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        extra_name: Optional[str] = None,
):
  assert len(x.dims) == 3

  dirname = f"{input_name}_cosine_sim"
  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])

  feature_axis_int = x.get_axis_from_description(feature_dim)
  x_raw = x.raw_tensor
  x_norm = x_raw / x_raw.norm(dim=feature_axis_int, keepdim=True)

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_norm_b = x_norm[b, :seq_len_b]
    cos_sim = torch.matmul(x_norm_b, x_norm_b.transpose(0, 1))
    # cos_sim = torch.tril(cos_sim, diagonal=0)
    cos_sim = cos_sim.cpu().detach().numpy()

    plt.matshow(cos_sim, cmap="seismic")
    plt.colorbar()
    plt.savefig(os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}.png"))
    plt.close()

    x_norm_b = x_norm_b.cpu().detach().numpy()
    from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
    # kmeans = KMeans(n_clusters=num_non_blank_targets + 1, random_state=0).fit(x_norm_b)
    # clusters = KMeans(n_clusters=2, random_state=0).fit(x_norm_b)
    clusters = SpectralClustering(n_clusters=4, random_state=0).fit(x_norm_b)
    plt.matshow(clusters.labels_[None, :], cmap="tab20c")
    plt.savefig(os.path.join(dirname, f"clusters_{seq_tag.replace('/', '_')}.png"))
    plt.close()
  # exit()


def _blank_model_analysis(
        model: SegmentalAttentionModel,
        x: rf.Tensor,
        input_name: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        non_blank_targets_spatial_dim: rf.Dim,
        enc_spatial_dim: rf.Dim,
        label_model_states: Optional[rf.Tensor] = None
):
  assert len(x.dims) == 3
  assert model.blank_decoder_version in (3, 4, 8)

  dirname = f"{input_name}_blank_model_analysis"
  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa
  spatial_dim = x.remaining_dims([batch_dim, x.feature_dim])[0]
  # spatial_dimension = rf.reduce_max(spatial_dim.dyn_size_ext, axis=batch_dim).raw_tensor.item()
  # feature_dimension = model.blank_decoder.s.out_dim.dimension

  if model.blank_decoder_version == 8:
    blank_model_input = x
  else:
    blank_model_input = rf.concat_features(x, label_model_states, allow_broadcast=False)

  if model.blank_decoder_version == 3:
    x_transformed, _ = model.blank_decoder.loop_step(
      enc=x,
      enc_spatial_dim=enc_spatial_dim,
      label_model_state=label_model_states,
      state=model.blank_decoder.default_initial_state(batch_dims=[batch_dim]),
      spatial_dim=enc_spatial_dim,
    )
    x_transformed = x_transformed["s_blank"]
  else:
    x_transformed = model.blank_decoder.s(blank_model_input)
    x_transformed = rf.reduce_out(x_transformed, mode="max", num_pieces=2, out_dim=model.blank_decoder.emit_prob.in_dim)

  x_transformed = x_transformed.copy_transpose(
    x_transformed.remaining_dims(x_transformed.feature_dim) + [x_transformed.feature_dim]
  )
  x_transformed_raw = x_transformed.raw_tensor

  x_transformed_norm = x_transformed.copy()
  x_transformed_norm.raw_tensor = x_transformed_raw / x_transformed_raw.norm(dim=-1, keepdim=True)

  if len(x_transformed_norm.dims) == 3:
    _plot_cosine_sim_matrix(
      x=x_transformed_norm,
      input_name=dirname,
      seq_tags=seq_tags,
      batch_dim=batch_dim,
      spatial_dim=enc_spatial_dim,
    )
  else:
    assert len(x_transformed_norm.dims) == 4
    x_transformed_norm = x_transformed_norm.copy_transpose(
      [batch_dim, non_blank_targets_spatial_dim, enc_spatial_dim, x_transformed_norm.feature_dim])
    S = rf.reduce_max(
      non_blank_targets_spatial_dim.dyn_size_ext,
      axis=non_blank_targets_spatial_dim.dyn_size_ext.dims
    ).raw_tensor.item()

    for s in range(S):
      _plot_cosine_sim_matrix(
        x=rf.gather(x_transformed_norm, axis=non_blank_targets_spatial_dim, indices=rf.constant(s, dims=[batch_dim])),
        input_name=f"{dirname}/s_{s}",
        seq_tags=seq_tags,
        batch_dim=batch_dim,
        spatial_dim=enc_spatial_dim,
      )
  #   exit()
  #
  #
  # # emit_prob_weight = model.blank_decoder.emit_prob.weight.raw_tensor
  # # emit_prob_bias = model.blank_decoder.emit_prob.bias.raw_tensor
  # # emit_logits = torch.matmul(x_raw, emit_prob_weight) + emit_prob_bias
  #
  # for b in range(B):
  #   seq_tag = seq_tags.raw_tensor[b].item()
  #   seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
  #   x_norm_b = x_transformed_norm[b, :seq_len_b]
  #   cos_sim = torch.matmul(x_norm_b, x_norm_b.transpose(0, 1))
  #   # cos_sim = torch.tril(cos_sim, diagonal=0)
  #   cos_sim = cos_sim.cpu().detach().numpy()
  #
  #   plt.matshow(cos_sim, cmap="seismic")
  #   plt.colorbar()
  #   plt.savefig(os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}.png"))
  #   plt.close()
  #
  #   x_norm_b = x_norm_b.cpu().detach().numpy()
  #   from sklearn.cluster import SpectralClustering
  #   clusters = SpectralClustering(n_clusters=2, random_state=0).fit(x_norm_b)
  #
  #   plt.matshow(clusters.labels_[None, :], cmap="tab20c")
  #   plt.savefig(os.path.join(dirname, f"clusters_{seq_tag.replace('/', '_')}.png"))
  #   plt.close()

    # emit_logits_b = emit_logits[b, :seq_len_b]
    # emit_logits_b = emit_logits_b.cpu().detach().numpy()
    # emit_logits_b = np.squeeze(emit_logits_b)
    # plt.matshow(emit_logits_b[None, :], cmap="seismic")
    # plt.colorbar()
    # plt.savefig(os.path.join(dirname, f"emit_logits_{seq_tag.replace('/', '_')}.png"))
    # plt.close()


def _info_mixing_analysis(
        input_: rf.Tensor,
        representation: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        input_name: str,
        json_vocab_path: Path
):
  """
  Analyze the mixing of the input and the representation in the log probabilities.

  :param input_: The input frames for the Conformer, i.e. x_linear.
  :param representation: The intermediate encoder representation to analyse.
  """
  input_raw = input_.raw_tensor
  representation_raw = representation.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa
  T = input_raw.size(1)  # noqa
  F = input_raw.size(2)  # noqa

  print()
  print("input_raw", input_raw.shape)
  print("log_probs_raw", log_probs_raw.shape)

  batch_gradients = []
  for b in range(B):
    s_gradients = []
    for s in range(S):
      v = targets.raw_tensor[b, s]
      input_raw.retain_grad()
      representation_raw.retain_grad()
      log_probs_raw.retain_grad()
      log_probs_raw[b, s, v].backward(retain_graph=True, inputs=[representation_raw])

      log_prob_grad_wrt_repr = representation_raw.grad[b]
      print("log_prob_grad_wrt_repr", log_prob_grad_wrt_repr.shape)

      gs = []
      for t in range(T):
        g = 0
        for f in range(F):
          representation_raw[b, t, f].backward(
            retain_graph=True,
            inputs=[input_raw]
          )
          repr_grad_wrt_input = input_raw.grad[b, t]

          g += log_prob_grad_wrt_repr[t, f] * repr_grad_wrt_input

        g = torch.linalg.vector_norm(g, dim=-1)

        print("g", g)
        exit()


def _plot_multi_head_enc_self_att(
        att_weights: rf.Tensor,
        energies: rf.Tensor,
        head_dim: rf.Dim,
        batch_dims: List[rf.Dim],
        enc_query_dim: rf.Dim,
        enc_kv_dim: rf.Dim,
        dirname: str,
        seq_tags: rf.Tensor,
):
  for tensor, tensor_name in (
          (att_weights, "att_weights"),
          (energies, "energies"),
  ):
    for h in range(head_dim.dimension):
      tensor_single_head = rf.gather(
        tensor,
        indices=rf.constant(h, dims=batch_dims),
        axis=head_dim,
      )
      dummy_att_head_dim = rf.Dim(1, name="att_head")
      tensor_single_head = rf.expand_dim(tensor_single_head, dim=dummy_att_head_dim)
      tensor_single_head = tensor_single_head.copy_transpose(
        batch_dims + [enc_query_dim, enc_kv_dim, dummy_att_head_dim])

      head_dirname = os.path.join(dirname, f"{tensor_name}_head-{h}")
      if not os.path.exists(head_dirname):
        os.makedirs(head_dirname)

      tensor_single_head_raw = tensor_single_head.raw_tensor

      for b in range(tensor_single_head_raw.size(0)):
        seq_tag = seq_tags.raw_tensor[b].item()
        seq_len_b = enc_query_dim.dyn_size_ext.raw_tensor[b].item()
        tensor_single_head_b_raw = tensor_single_head_raw[b, :seq_len_b, :seq_len_b, 0]
        tensor_single_head_b_raw = tensor_single_head_b_raw.cpu().detach().numpy()

        plt.matshow(tensor_single_head_b_raw, cmap="Blues")
        plt.ylabel("queries")
        plt.xlabel("keys/values")
        plt.savefig(os.path.join(head_dirname, f"{tensor_name}_{seq_tag.replace('/', '_')}.png"))
        plt.close()


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


def get_tensors_trafo_decoder(
        model: TransformerDecoder,
        enc_args: Dict[str, rf.Tensor],
        enc_spatial_dim: rf.Dim,
        non_blank_targets: rf.Tensor,
        non_blank_targets_spatial_dim: rf.Dim,
        batch_dims: List[rf.Dim],
        dec_layer_idx: int,
        get_cross_attentions: bool = True,
):
  funcs_to_trace_list = [
    TransformerDecoder.__call__,
    TransformerDecoderLayer.__call__,
    rf.CrossAttention.__call__,
    rf.CrossAttention.attention,
    rf.dot_attention,
  ]
  global code_obj_to_func
  global captured_tensors
  code_obj_to_func = {func.__code__: func for func in funcs_to_trace_list}
  captured_tensors = {}

  _dec_layer_idx = 2 * dec_layer_idx + int(get_cross_attentions)

  _layer_mapping = {
    # dot_attention is called twice per decoder layer (self-attention and cross-attention)
    # we want the last cross-attention, therefore we use 11 as the second arg here
    "att_weights": (rf.dot_attention, _dec_layer_idx, "att_weights", -1),
    "energy": (rf.dot_attention, _dec_layer_idx, "energy", -1),
    "logits": (TransformerDecoder.__call__, 0, "logits", -1),
  }

  sys.settrace(_trace_func)
  model(
    non_blank_targets,
    spatial_dim=non_blank_targets_spatial_dim,
    encoder=model.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
    state=model.default_initial_state(batch_dims=batch_dims)
  )
  sys.settrace(None)

  att_weights = None
  energies = None
  logits = None

  for tensor_name, var_path in _layer_mapping.items():
    new_out = captured_tensors
    for k in var_path:
      new_out = new_out[k]

    if tensor_name == "att_weights":
      att_weights = new_out
    elif tensor_name == "energy":
      energies = new_out
    else:
      assert tensor_name == "logits"
      logits = new_out  # type: rf.Tensor

  return att_weights, energies, logits


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

  # start by analyzing the weights
  # analyze_weights(model)

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

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

  if isinstance(model.label_decoder, SegmentalAttLabelDecoder):
    target_blank_idx = model.blank_idx
  else:
    target_blank_idx = None

  max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

  ref_alignment_hdf = Path(config.typed_value("ref_alignment_hdf", str))
  ref_alignment_blank_idx = config.typed_value("ref_alignment_blank_idx", int)
  ref_alignment_vocab_path = Path(config.typed_value("ref_alignment_vocab_path", str))
  json_vocab_path = Path(config.typed_value("json_vocab_path", str))

  with torch.enable_grad():
    # ------------------- run encoder and capture encoder input -----------------
    set_trace_variables(
      funcs_to_trace_list=[
        model.encoder.encode,
        ConformerEncoder.__call__,
        ConformerEncoderLayer.__call__,
        RelPosSelfAttention.__call__,
      ]
    )

    collected_outputs = {}
    sys.settrace(_trace_func)
    enc_args, enc_spatial_dim = model.encoder.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    sys.settrace(None)

    x_linear = process_captured_tensors(
      layer_mapping={"x_linear": (ConformerEncoder.__call__, 0, "x_linear", -1)},
    )
    assert isinstance(x_linear, rf.Tensor)

    enc_att_weights = process_captured_tensors(
      layer_mapping={"att_weights": (RelPosSelfAttention.__call__, 11, "att_weights", -1)},
    )
    assert isinstance(enc_att_weights, rf.Tensor)
    enc_att_energies = process_captured_tensors(
      layer_mapping={"energies": (RelPosSelfAttention.__call__, 11, "scores", -1)},
    )
    assert isinstance(enc_att_energies, rf.Tensor)

    enc_att_head_dim = enc_att_weights.remaining_dims(batch_dims + enc_att_weights.get_dyn_size_tags())[0]
    enc_kv_dim = enc_att_weights.remaining_dims(batch_dims + [enc_att_head_dim, enc_spatial_dim])[0]
    _plot_multi_head_enc_self_att(
      att_weights=enc_att_weights,
      energies=enc_att_energies,
      head_dim=enc_att_head_dim,
      batch_dims=batch_dims,
      enc_query_dim=enc_spatial_dim,
      enc_kv_dim=enc_kv_dim,
      dirname="enc_self_att",
      seq_tags=seq_tags,
    )

    # ----------------------------------------------------------------------------------

    for enc_layer_idx in range(11, 12):
      enc = collected_outputs[str(enc_layer_idx)]
      enc_args = {
        "enc": enc,
      }

      if isinstance(model.label_decoder, GlobalAttDecoder) or isinstance(model.label_decoder, TransformerDecoder):
        dump_hdfs(
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          align_targets=non_blank_targets,
          align_target_dim=model.target_dim.dimension
        )

      if not isinstance(model.label_decoder, TransformerDecoder):
        enc_args["enc_ctx"] = model.encoder.enc_ctx(enc)  # does not exist for transformer decoder

        if model.label_decoder.use_weight_feedback:
          enc_args["inv_fertility"] = rf.sigmoid(model.encoder.inv_fertility(enc))  # does not exist for transformer decoder
        else:
          enc_args["inv_fertility"] = None  # dummy value, not used

      if isinstance(model.label_decoder, SegmentalAttLabelDecoder):
        dump_hdfs(
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          center_positions=center_positions,
          segment_starts=segment_starts,
          segment_lens=segment_lens,
          align_targets=non_blank_targets,
          align_target_dim=model.target_dim.dimension
        )
        center_positions_hdf = Path("center_positions.hdf")
        segment_starts_hdf = Path("seg_starts.hdf")
        segment_lens_hdf = Path("seg_lens.hdf")

        max_num_frames = rf.reduce_max(enc_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()
        new_slice_dim = rf.Dim(min(model.center_window_size, max_num_frames), name="att_window")

        if type(model.label_decoder) is SegmentalAttLabelDecoder:
          assert not model.use_joint_model

          set_trace_variables(
            funcs_to_trace_list=[
              forward_sequence_segmental,
              SegmentalAttLabelDecoder.loop_step,
              SegmentalAttLabelDecoder.decode_logits,
            ]
          )

          sys.settrace(_trace_func)
          forward_sequence_segmental(
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

          def _replace_slice_dim(tensor: rf.Tensor):
            old_slice_dim = tensor.remaining_dims(
              batch_dims + [model.label_decoder.att_num_heads])
            assert len(old_slice_dim) == 1
            old_slice_dim = old_slice_dim[0]
            max_slice_size = rf.reduce_max(old_slice_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

            # at the end of the seq, there may be less than center_window_size attention weights
            # in this case, just pad with zeros until we reach center_window_size
            if max_slice_size < new_slice_dim.dimension:
              tensor, _ = rf.pad(
                tensor,
                axes=[old_slice_dim],
                padding=[(0, new_slice_dim.dimension - max_slice_size)],
                out_dims=[new_slice_dim],
                value=0.0,
              )
            else:
              tensor = utils.copy_tensor_replace_dim_tag(tensor, old_slice_dim, new_slice_dim)

            return tensor

          att_weights = process_captured_tensors(
            layer_mapping={
              f"att_weight_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "att_weights", -1) for i in range(max_num_labels)},
            process_func=_replace_slice_dim
          )
          att_weights, _ = rf.stack(att_weights, out_dim=non_blank_targets_spatial_dim)

          if model.center_window_size != 1 and model.label_decoder.gaussian_att_weight_opts is None:
            energies = process_captured_tensors(
              layer_mapping={
                f"energy_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "energy", -1) for i in range(max_num_labels)},
              process_func=_replace_slice_dim
            )
            energies, _ = rf.stack(energies, out_dim=non_blank_targets_spatial_dim)
          else:
            energies = None

          label_model_states = process_captured_tensors(
            layer_mapping={
              f"s_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "s", -1) for i in range(max_num_labels)},
          )
          label_model_states, _ = rf.stack(label_model_states, out_dim=non_blank_targets_spatial_dim)
          label_model_states.feature_dim = label_model_states.remaining_dims(
            batch_dims + [non_blank_targets_spatial_dim])[0]

          logits = process_captured_tensors(
            layer_mapping={"logits": (SegmentalAttLabelDecoder.decode_logits, 0, "logits", -1)}
          )
          assert isinstance(logits, rf.Tensor)
          log_probs = rf.log_softmax(logits, axis=model.target_dim)
          log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

          # shapes
          # print("att_weights", att_weights)  # [B, T, S, 1]
          # print("logits", logits)  # [B, S, V]
          # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
        else:
          assert type(model.label_decoder) is SegmentalAttEfficientLabelDecoder, "need to extend for other models"
          assert not model.use_joint_model

          set_trace_variables(
            funcs_to_trace_list=[
              forward_sequence_efficient_segmental,
              SegmentalAttEfficientLabelDecoder.__call__,
              SegmentalAttEfficientLabelDecoder.decode_logits,
            ]
          )

          sys.settrace(_trace_func)
          forward_sequence_efficient_segmental(
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

          def _replace_slice_dim(tensor: rf.Tensor):
            old_slice_dim = tensor.remaining_dims(
              batch_dims + [model.label_decoder.att_num_heads, non_blank_targets_spatial_dim])
            assert len(old_slice_dim) == 1
            return utils.copy_tensor_replace_dim_tag(tensor, old_slice_dim[0], new_slice_dim)

          att_weights = process_captured_tensors(
            layer_mapping={"att_weights": (SegmentalAttEfficientLabelDecoder.__call__, 0, "att_weights", -1)},
            process_func=_replace_slice_dim
          )
          assert isinstance(att_weights, rf.Tensor)

          if model.center_window_size != 1 or model.label_decoder.gaussian_att_weight_opts is not None:
            energies = process_captured_tensors(
              layer_mapping={"energy": (SegmentalAttEfficientLabelDecoder.__call__, 0, "energy", -1)},
              process_func=_replace_slice_dim
            )
            assert isinstance(energies, rf.Tensor)
          else:
            energies = None  # no energies for center window size 1 (att weights = 1 for current frame)

          label_model_states = process_captured_tensors(
            layer_mapping={f"s": (forward_sequence_efficient_segmental, 0, "s_out", -1)},
          )
          assert isinstance(label_model_states, rf.Tensor)

          logits = process_captured_tensors(
            layer_mapping={"logits": (SegmentalAttEfficientLabelDecoder.decode_logits, 0, "logits", -1)}
          )
          assert isinstance(logits, rf.Tensor)
          log_probs = rf.log_softmax(logits, axis=model.target_dim)
          log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

          # shapes
          # print("att_weights", att_weights)  # [B, T, S, 1]
          # print("logits", logits)  # [B, S, V]
          # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]

        att_weights = scatter_att_weights(
          att_weights=att_weights,
          segment_starts=segment_starts,
          new_slice_dim=new_slice_dim,
          align_targets_spatial_dim=enc_spatial_dim,
          batch_dims=batch_dims,
        )
        if energies is not None:
          energies = scatter_att_weights(
            att_weights=energies,
            segment_starts=segment_starts,
            new_slice_dim=new_slice_dim,
            align_targets_spatial_dim=enc_spatial_dim,
            batch_dims=batch_dims,
          )

      elif type(model.label_decoder) is GlobalAttEfficientDecoder:
        set_trace_variables(
          funcs_to_trace_list=[
            forward_sequence_global,
            get_s_and_att_efficient_global,
            GlobalAttEfficientDecoder.__call__,
            GlobalAttEfficientDecoder.decode_logits,
          ]
        )

        sys.settrace(_trace_func)
        forward_sequence_global(
          model=model.label_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          targets=non_blank_targets,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          center_positions=center_positions,
          batch_dims=batch_dims,
        )
        sys.settrace(None)

        att_weights = process_captured_tensors(
          layer_mapping={"att_weights": (GlobalAttEfficientDecoder.__call__, 0, "att_weights", -1)},
        )
        assert isinstance(att_weights, rf.Tensor)
        energy_in = process_captured_tensors(
          layer_mapping={"energy_in": (GlobalAttEfficientDecoder.__call__, 0, "energy_in", -1)},
        )
        assert isinstance(energy_in, rf.Tensor)
        energies = process_captured_tensors(
          layer_mapping={"energy": (GlobalAttEfficientDecoder.__call__, 0, "energy", -1)},
        )
        assert isinstance(energies, rf.Tensor)
        logits = process_captured_tensors(
          layer_mapping={"logits": (GlobalAttEfficientDecoder.decode_logits, 0, "logits", -1)}
        )
        assert isinstance(logits, rf.Tensor)
        label_model_states = process_captured_tensors(
          layer_mapping={f"s": (get_s_and_att_efficient_global, 0, "s", -1)},
        )
        assert isinstance(label_model_states, rf.Tensor)

        log_probs = rf.log_softmax(logits, axis=model.target_dim)
        log_probs = log_probs.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])

        # shapes
        # print("att_weights", att_weights)  # [B, T, S, 1]
        # print("logits", logits)  # [B, S, V]
        # print("non_blank_targets", non_blank_targets.raw_tensor[0, :])  # [B, S]
      elif type(model.label_decoder) is GlobalAttDecoder:
        set_trace_variables(
          funcs_to_trace_list=[
            forward_sequence_global,
            get_s_and_att_global,
            GlobalAttDecoder.loop_step,
            GlobalAttDecoder.decode_logits,
          ]
        )

        sys.settrace(_trace_func)
        forward_sequence_global(
          model=model.label_decoder,
          enc_args=enc_args,
          enc_spatial_dim=enc_spatial_dim,
          targets=non_blank_targets,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          center_positions=center_positions,
          batch_dims=batch_dims,
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
      else:
        assert isinstance(model.label_decoder, TransformerDecoder)

        for get_cross_attentions in [True, False]:
          for dec_layer_idx in range(5, 6):
            set_trace_variables(
              funcs_to_trace_list=[
                TransformerDecoder.__call__,
                TransformerDecoderLayer.__call__,
                rf.CrossAttention.__call__,
                rf.CrossAttention.attention,
                rf.dot_attention,
              ]
            )

            sys.settrace(_trace_func)
            model.label_decoder(
              non_blank_targets,
              spatial_dim=non_blank_targets_spatial_dim,
              encoder=model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
              state=model.label_decoder.default_initial_state(batch_dims=batch_dims)
            )
            sys.settrace(None)

            _dec_layer_idx = 2 * dec_layer_idx + int(get_cross_attentions)
            att_weights = process_captured_tensors(
              layer_mapping={"att_weights": (rf.dot_attention, _dec_layer_idx, "att_weights", -1)}
            )
            assert isinstance(att_weights, rf.Tensor)
            energies = process_captured_tensors(
              layer_mapping={"energy": (rf.dot_attention, _dec_layer_idx, "energy", -1)}
            )
            assert isinstance(energies, rf.Tensor)
            logits = process_captured_tensors(
              layer_mapping={"logits": (TransformerDecoder.__call__, 0, "logits", -1)}
            )
            log_probs = rf.log_softmax(logits, axis=model.target_dim)

            # shapes for trafo decoder
            # print("att_weights", att_weights)  # [B, H, S, T] -> [B, T, S, 1]
            # print("energies", energies)  # [B, H, S, T]
            # print("logits", logits)  # [B, S, V]

            # print("x_linear", x_linear)  # [B, T, F]
            # print("logits", logits)  # [B, S, V]

            def _plot_attention_weights():
              if get_cross_attentions:
                att_head_dim = att_weights.remaining_dims(batch_dims + [enc_spatial_dim, targets_spatial_dim])
                assert len(att_head_dim) == 1
                att_head_dim = att_head_dim[0]
              else:
                # this is either the att head dim or the kv-dim
                att_head_and_kv_dim = att_weights.remaining_dims(batch_dims + [targets_spatial_dim])
                # att head dim is static, kv-dim is dynamic
                att_head_dim = [dim for dim in att_head_and_kv_dim if dim.is_static()][0]
                kv_dim = [dim for dim in att_head_and_kv_dim if dim.is_dynamic()][0]
                # kv_dim has seq lens (1, 2, ..., S) but needs single len (S) for my hdf_dump script
                kv_dim.dyn_size_ext.raw_tensor = targets_spatial_dim.dyn_size_ext.raw_tensor.clone()

              for tensor, tensor_name in (
                      (att_weights, "att_weights"),
                      (energies, "energies"),
              ):
                for h in range(att_head_dim.dimension):
                  tensor_single_head = rf.gather(
                    tensor,
                    indices=rf.constant(h, dims=batch_dims),
                    axis=att_head_dim,
                  )
                  dummy_att_head_dim = rf.Dim(1, name="att_head")
                  tensor_single_head = rf.expand_dim(tensor_single_head, dim=dummy_att_head_dim)

                  if get_cross_attentions:
                    # [B, S, T, 1]
                    tensor_single_head = tensor_single_head.copy_transpose(
                      batch_dims + [targets_spatial_dim, enc_spatial_dim, dummy_att_head_dim])
                  else:
                    # [B, S, S, 1]
                    tensor_single_head = tensor_single_head.copy_transpose(
                      batch_dims + [targets_spatial_dim, kv_dim, dummy_att_head_dim])
                    # optionally copy lower triangular part to upper triangular part
                    # att_weights_single_head.raw_tensor[0, :, :, 0] = copy_lower_triangular_matrix_to_upper(
                    #   att_weights_single_head.raw_tensor[0, :, :, 0]
                    # )

                  if get_cross_attentions:
                    dirname = f"cross-att/enc-layer-{enc_layer_idx + 1}_dec-layer-{dec_layer_idx + 1}"
                  else:
                    dirname = f"self-att/dec-layer-{dec_layer_idx + 1}"
                  dirname = os.path.join(dirname, f"{tensor_name}_head-{h}")

                  dump_hdfs(
                    att_weights=tensor_single_head,
                    batch_dims=batch_dims,
                    dirname=dirname,
                    seq_tags=seq_tags,
                  )

                  targets_hdf = Path("targets.hdf")
                  if get_cross_attentions:
                    plot_att_weight_opts = {
                      "ref_alignment_blank_idx": ref_alignment_blank_idx,
                      "ref_alignment_hdf": ref_alignment_hdf,
                      "ref_alignment_json_vocab_path": ref_alignment_vocab_path,
                    }
                  else:
                    plot_att_weight_opts = {
                      "ref_alignment_blank_idx": -1,
                      "ref_alignment_hdf": targets_hdf,
                      "ref_alignment_json_vocab_path": json_vocab_path,
                    }

                  plot_att_weights(
                    att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
                    targets_hdf=targets_hdf,
                    seg_starts_hdf=None,
                    seg_lens_hdf=None,
                    center_positions_hdf=None,
                    target_blank_idx=None,
                    json_vocab_path=json_vocab_path,
                    segment_whitelist=list(seq_tags.raw_tensor),
                    plot_name=dirname,
                    **plot_att_weight_opts
                  )

            _plot_attention_weights()

      if isinstance(model.label_decoder, GlobalAttDecoder):
        for s in range(max_num_labels):
          energy_in_s = rf.gather(
            energy_in,
            indices=rf.constant(s, dims=batch_dims),
            axis=non_blank_targets_spatial_dim,
          )
          energy_in_s = energy_in_s.copy_transpose(batch_dims + [enc_spatial_dim, model.label_decoder.enc_key_total_dim])

          _plot_cosine_sim_matrix(
            energy_in_s,
            input_name=f"energy_in_analysis/s-{s}",
            batch_dim=batch_dims[0],
            seq_tags=seq_tags,
            spatial_dim=enc_spatial_dim,
            extra_name=str(s)
          )

      for input_name, input_ in (
              *[(f"enc-{i}", collected_outputs[str(i)]) for i in range(12)],
              (f"enc_ctx-{enc_layer_idx}", enc_args["enc_ctx"]),
              ("x_linear", x_linear),
      ):
        if isinstance(model, SegmentalAttentionModel) and isinstance(model.label_decoder, SegmentalAttLabelDecoder) and (
                model.center_window_size == 1 or model.label_decoder.gaussian_att_weight_opts is not None) and (
          "enc_ctx" in input_name
        ):
          continue  # encoder ctx not used

        _plot_log_prob_gradient_wrt_to_input(
          input_=input_,
          log_probs=log_probs,
          targets=non_blank_targets,
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          enc_spatial_dim=enc_spatial_dim,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          input_name=input_name,
          json_vocab_path=json_vocab_path,
          ref_alignment_hdf=ref_alignment_hdf,
          ref_alignment_blank_idx=ref_alignment_blank_idx,
          ref_alignment_json_vocab_path=ref_alignment_vocab_path,
        )

        _plot_cosine_sim_matrix(
          input_,
          input_name=input_name,
          batch_dim=batch_dims[0],
          seq_tags=seq_tags,
          spatial_dim=enc_spatial_dim
        )

      if isinstance(model, SegmentalAttentionModel) and model.blank_decoder_version in (3, 4, 8,):
        label_model_states_unmasked = utils.get_unmasked(
          label_model_states,
          input_spatial_dim=non_blank_targets_spatial_dim,
          mask=non_blank_mask,
          mask_spatial_dim=targets_spatial_dim,
        )
        label_model_states_unmasked = utils.copy_tensor_replace_dim_tag(
          label_model_states_unmasked, targets_spatial_dim, enc_spatial_dim
        )

        _blank_model_analysis(
          model=model,
          x=collected_outputs["11"],
          input_name=f"enc-{11}",
          batch_dim=batch_dims[0],
          seq_tags=seq_tags,
          non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
          enc_spatial_dim=enc_spatial_dim,
          label_model_states=label_model_states_unmasked,
        )

      # plot attention weights for non-trafo decoders
      # (for trafo decoders, the attention weights are plotted in the loop above)
      if not isinstance(model.label_decoder, TransformerDecoder):
        dirname = f"cross-att/enc-layer-{enc_layer_idx + 1}"
        if os.path.exists(dirname):
          return

        for tensor, tensor_name in [
          (att_weights, "att_weights"),
          (energies, "energies"),
        ]:
          if energies is None:
            continue

          tensor_dirname = os.path.join(dirname, tensor_name)

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
