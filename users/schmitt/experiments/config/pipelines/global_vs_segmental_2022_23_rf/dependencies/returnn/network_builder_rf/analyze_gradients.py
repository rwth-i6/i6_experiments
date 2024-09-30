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


def _plot_log_prob_gradient_wrt_to_input_single_seq(
        input_: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        dirname: str,
        json_vocab_path: Optional[Path],
        ref_alignment_hdf: Optional[Path],
        ref_alignment_blank_idx: Optional[int],
        ref_alignment_json_vocab_path: Optional[Path],
        return_gradients: bool = False,
        print_progress: bool = False,
):
  from returnn.config import get_global_config
  config = get_global_config()

  # if os.path.exists(dirname) and not return_gradients:
  #   return

  input_raw = input_.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa

  batch_gradients = []
  for b in range(B):
    s_gradients = []
    for s in range(targets_spatial_dim.dyn_size_ext.raw_tensor[b].item()):
      if print_progress:
        print(f"Batch {b}/{B}, Step {s}/{S}")
      v = targets.raw_tensor[b, s]
      input_raw.retain_grad()
      log_probs_raw.retain_grad()
      log_probs_raw[b, s, v].backward(retain_graph=True)

      x_linear_grad_l2 = torch.linalg.vector_norm(input_raw.grad[b], dim=-1)
      s_gradients.append(x_linear_grad_l2)

      # zero grad before next step
      input_raw.grad.zero_()

    num_s_gradients = len(s_gradients)
    if num_s_gradients < S:
      s_gradients += [torch.zeros_like(s_gradients[0]) for _ in range(S - num_s_gradients)]
    batch_gradients.append(torch.stack(s_gradients, dim=0))
  x_linear_grad_l2_raw = torch.stack(batch_gradients, dim=0)[:, :, :, None]
  x_linear_grad_l2 = rf.convert_to_tensor(
    x_linear_grad_l2_raw,
    dims=batch_dims + [targets_spatial_dim, enc_spatial_dim, rf.Dim(1, name="dummy")],
  )

  if return_gradients:
    return x_linear_grad_l2
  else:
    dump_hdfs(
      att_weights=rf.log(x_linear_grad_l2),  # use log for better visualization
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
      plot_name=dirname,
      plot_w_color_gradient=config.bool("debug", False)
    )


def _plot_log_prob_gradient_wrt_to_input_batched(
        input_: rf.Tensor,
        log_probs: rf.Tensor,
        targets: rf.Tensor,
        batch_dims: List[rf.Dim],
        seq_tags: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        targets_spatial_dim: rf.Dim,
        dirname: str,
        json_vocab_path: Optional[Path],
        ref_alignment_hdf: Optional[Path],
        ref_alignment_blank_idx: Optional[int],
        ref_alignment_json_vocab_path: Optional[Path],
        return_gradients: bool = False,
        print_progress: bool = False,
        dummy_singleton_dim: rf.Dim = rf.Dim(1, name="dummy"),
):
  print(f"Plotting log prob gradient w.r.t. input for {dirname}")

  from returnn.config import get_global_config
  config = get_global_config()

  if os.path.exists(dirname) and not return_gradients:
    return

  input_raw = input_.raw_tensor
  log_probs_raw = log_probs.raw_tensor
  B = log_probs_raw.size(0)  # noqa
  S = log_probs_raw.size(1)  # noqa

  log_probs_sum = rf.gather(
    log_probs,
    indices=targets,
  )
  log_probs_sum = rf.reduce_sum(log_probs_sum, axis=batch_dims)
  log_probs_sum_raw = log_probs_sum.raw_tensor

  s_gradients = []
  for s in range(S):
    if print_progress:
      print(f"Step {s}/{S}")
    input_raw.retain_grad()
    log_probs_sum_raw.retain_grad()
    log_probs_sum_raw[s].backward(retain_graph=True)

    x_linear_grad_l2 = torch.linalg.vector_norm(input_raw.grad, dim=-1)
    s_gradients.append(x_linear_grad_l2)

    # zero grad before next step
    input_raw.grad.zero_()

  x_linear_grad_l2_raw = torch.stack(s_gradients, dim=1)[:, :, :, None]

  input_spatial_dim = input_.remaining_dims(batch_dims + [input_.feature_dim])[0]
  x_linear_grad_l2 = rf.convert_to_tensor(
    x_linear_grad_l2_raw,
    dims=batch_dims + [targets_spatial_dim, input_spatial_dim, dummy_singleton_dim],
  )

  if return_gradients:
    return x_linear_grad_l2
  else:
    log_x_linear_grad_l2 = rf.log(x_linear_grad_l2)
    dump_hdfs(
      att_weights=log_x_linear_grad_l2,  # use log for better visualization
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
      plot_name=dirname,
      plot_w_color_gradient=config.bool("debug", False)
    )


def _plot_activation_matrix(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        max_values: rf.Tensor,
        min_values: rf.Tensor,
        activation_magnitude_file_path,
        extra_name: Optional[str] = None,
        batch_mask: Optional[rf.Tensor] = None,
        non_blank_positions_dict: Optional[Dict] = None,
        non_blank_targets_dict: Optional[Dict] = None,
        return_axes: bool = False,
):
  print(f"Plotting activation matrix for {input_name}")
  assert len(x.dims) == 3

  plot_file_paths = {}

  if not return_axes:
    if os.path.exists(dirname):
      return plot_file_paths, None, None
    else:
      os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])

  x_raw = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  activation_axes = {}

  for b in range(B):
    if batch_mask is not None and not batch_mask.raw_tensor[b].item():
      continue

    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_raw_b = x_raw[b, :seq_len_b]  # [T, F]
    x_raw_b = x_raw_b.cpu().detach().numpy()
    x_raw_b = x_raw_b.T  # [F, T]

    fig = plt.figure(figsize=(10, 5))
    activation_ax = plt.axes()
    mat = activation_ax.matshow(
      x_raw_b,
      cmap="Reds",
      vmin=min_values.raw_tensor[b].item(),
      vmax=max_values.raw_tensor[b].item(),
      aspect="auto",
    )
    # activation_ax.set_title(f"{input_name} Activation matrix for \n {seq_tag}", fontsize=12, pad=20)

    time_step_size = 1 / 60 * 500
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    activation_ax.set_xticks(time_ticks)
    tick_labels = [(time_tick * 60) / 1000 for time_tick in time_ticks]
    activation_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels], fontsize=fontsize_ticks)
    activation_ax.xaxis.set_ticks_position('bottom')

    yticks = np.arange(0, x_raw_b.shape[0], 100, dtype=int)
    activation_ax.set_yticks(yticks)
    activation_ax.set_yticklabels(yticks, fontsize=fontsize_ticks)

    activation_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    activation_ax.set_ylabel("Feature dimension", fontsize=fontsize_axis, labelpad=4)
    activation_ax.invert_yaxis()

    # show top axis with non-blank positions
    if non_blank_positions_dict is not None:
      non_blank_positions = non_blank_positions_dict[seq_tag]
      non_blank_targets = non_blank_targets_dict[seq_tag]

      ref_align_axis = activation_ax.secondary_xaxis('top')
      ref_align_axis.set_xticks(non_blank_positions)
      ref_align_axis.tick_params("x", length=4)
      ref_align_axis.set_xlabel("Ref. alignment", fontsize=fontsize_axis, labelpad=4)
      ref_align_axis.set_xticklabels(non_blank_targets, rotation=90)

    # fig.tight_layout()

    if not return_axes:
      cax = fig.add_axes([activation_ax.get_position().x1 + 0.01, activation_ax.get_position().y0, 0.02,
                          activation_ax.get_position().height])
      plt.colorbar(mat, cax=cax)

      plot_file_path_png = os.path.join(dirname,
                                        f"{input_name}_acts_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}.png")
      plot_file_path_pdf = os.path.join(dirname,
                                        f"{input_name}_acts_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}.pdf")
      magnitude_file_path = os.path.join(dirname,
                                         f"{input_name}_acts_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}_magnitude.txt")
      plt.savefig(plot_file_path_png)
      plt.savefig(plot_file_path_pdf)
      with open(activation_magnitude_file_path, "a") as f:
        f.write(f"{input_name} mean over norm of abs: {np.mean(np.linalg.norm(x_raw_b, axis=0))}\n")
        f.write(f"{input_name} norm over time and feature: {np.linalg.norm(x_raw_b)}\n\n")

      plot_file_paths[seq_tag] = plot_file_path_png
    else:
      activation_axes[b] = activation_ax

    plt.close()

  return plot_file_paths, activation_axes if return_axes else None


def _plot_neurons(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
):
  assert len(x.dims) == 3

  plot_file_paths = {}

  if os.path.exists(dirname):
    return plot_file_paths, None, None
  else:
    os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])

  x_raw = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_raw_b = x_raw[b, :seq_len_b]  # [T, F]
    x_raw_b = x_raw_b.cpu().detach().numpy()

    n_components = 8
    if n_components != x_raw_b.shape[1]:
      from sklearn.decomposition import PCA
      pca = PCA(n_components=n_components)
      x_raw_b = pca.fit_transform(x_raw_b)  # [T, n_components]

    if n_components == 512:
      n_rows = 16
      n_cols = 32
    elif n_components == 16:
      n_rows = 4
      n_cols = 4
    elif n_components == 8:
      n_rows = 2
      n_cols = 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    for i, neuron in enumerate(range(x_raw_b.shape[1])):
      axes[i // n_cols, i % n_cols].scatter(range(seq_len_b), x_raw_b[:, neuron])
      # axes[i // n_cols, i % n_cols].set_title(f"{input_name} neuron #{neuron} for \n {seq_tag}")
      axes[i // n_cols, i % n_cols].xaxis.set_ticks_position('bottom')
      # axes[i // n_cols, i % n_cols].set_xlabel("Time steps")
      # axes[i // n_cols, i % n_cols].set_ylabel("Activation")

    plot_file_path = os.path.join(
      dirname, f"{input_name}_neurons_{seq_tag.replace('/', '_')}.png")
    plt.savefig(
      plot_file_path,
    )
    plot_file_paths[seq_tag] = plot_file_path

    plt.close()

  return plot_file_paths


def _plot_encoder_gradient_graph(
        layers: List[rf.Tensor],
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
):
  import networkx as nx
  import pickle

  os.makedirs(dirname, exist_ok=True)

  assert all(len(layer.dims) == 3 for layer in layers)
  feature_dim = layers[0].remaining_dims([batch_dim, spatial_dim])[0]
  assert all(layer.dims == (batch_dim, spatial_dim, feature_dim) for layer in layers)

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  num_layers = len(layers)
  layer_indices = list(zip(range(num_layers - 1), range(1, num_layers)))

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    plot_name = os.path.join(dirname, f"encoder_graph_{seq_tag.replace('/', '_')}")
    pickle_name = os.path.join(dirname, f"encoder_graph_{seq_tag.replace('/', '_')}.pkl")
    if not os.path.exists(pickle_name):
      gradients = []
      # first, calculate gradients between each pair of layers (11 wrt 10, 10 wrt 9, ..., 1 wrt 0)
      for i, (input_layer_idx, output_layer_idx) in enumerate(reversed(layer_indices)):
        input_layer = layers[input_layer_idx].raw_tensor
        output_layer = layers[output_layer_idx].raw_tensor

        gradients_b = []
        for t in range(seq_len_b):
          print(f"layer {i + 1}/{num_layers}, time step {t}/{seq_len_b}")
          gradients_t = []

          for f in range(feature_dim.dimension):
            out_feature = output_layer[b, t, f]
            out_feature.retain_grad()
            input_layer.retain_grad()

            gradients_f = torch.autograd.grad(
              outputs=out_feature,
              inputs=input_layer,
              retain_graph=True
            )[0]  # [B, T, F]

            gradients_f_norm = torch.linalg.vector_norm(gradients_f[b], dim=-1)  # [T]
            gradients_t.append(gradients_f_norm)

            # zero gradients
            out_feature.grad.zero_()
            input_layer.grad.zero_()

          gradients_t = torch.stack(gradients_t, dim=0)  # [F, T]
          gradients_t = torch.sum(gradients_t, dim=0)  # [T]
          gradients_b.append(gradients_t)

        gradients_b = torch.stack(gradients_b, dim=0)  # [output, input]
        gradients.append(gradients_b)

      gradients = torch.stack(gradients, dim=0)  # [layer_pair, output, input]
      gradients_min = gradients.min()
      gradients_max = gradients.max()
      gradients = (gradients - gradients_min) / (gradients_max - gradients_min)
      k = 2
      gradients = gradients.cpu().detach().numpy()

      G = nx.DiGraph()
      edges = []
      edge_alphas = {}
      pos = {}
      pos_time_labels = {}

      # create graph and use normalized gradients as edge weights
      for i, (input_layer_idx, output_layer_idx) in enumerate(reversed(layer_indices)):
        input_layer_names = [f"{input_layer_idx}.{t}" for t in range(seq_len_b)]
        output_layer_names = [f"{output_layer_idx}.{t}" for t in range(seq_len_b)]

        G.add_nodes_from(input_layer_names, layer=input_layer_idx)
        if i == 0:
          G.add_nodes_from(output_layer_names, layer=output_layer_idx)

        # add edges from each output neuron to each input neuron
        for t_in, input_neuron in enumerate(input_layer_names):
          for t_out, output_neuron in enumerate(output_layer_names):
            edge_weight = gradients[i, t_out, t_in]
            G.add_edge(output_neuron, input_neuron, weight=edge_weight)
            edges.append((output_neuron, input_neuron))
            edge_alphas[(output_neuron, input_neuron)] = edge_weight

        # define positions of nodes in the plot
        if i == 0:
          pos.update((n, (j, 1)) for j, n in enumerate(output_layer_names))  # Layer 1
          pos_time_labels.update((n, (j, 0)) for j, n in enumerate(output_layer_names))  # Layer 1
        pos.update((n, (j, i + 2)) for j, n in enumerate(input_layer_names))  # Layer 2
    else:
      G = pickle.load(open(pickle_name, "rb"))
      edge_alphas = nx.get_edge_attributes(G, "weight")
      edges = list(G.edges)
      nodes = list(G.nodes)
      pos = {}
      pos_time_labels = {}
      for node in nodes:
        layer_idx, t = map(int, node.split("."))
        pos[node] = (t, num_layers - layer_idx)
        if num_layers - layer_idx == 1:
          pos_time_labels[node] = (t, 0)

    plt.figure(figsize=(40, 20))
    plt.title(f"Encoder graph for {seq_tag}")

    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue')
    # pos_time_labels = {k: v for k, v in pos_time_labels.items() if k[1][0] % 10 == 0}
    # time_labels = [k[1][0] for k in pos_time_labels.keys()]
    # nx.draw_networkx_labels(G, pos_time_labels, labels=time_labels)

    for edge in edges:
      alpha = edge_alphas[edge]
      if alpha > 0.2:
        nx.draw_networkx_edges(G, pos, edgelist=[edge], alpha=alpha, width=2.0, edge_color='black')
    plt.savefig(f"{plot_name}.png")
    plt.savefig(f"{plot_name}.pdf")
    pickle.dump(G, open(os.path.join(dirname, f"encoder_graph_{seq_tag.replace('/', '_')}.pkl"), "wb"))
    plt.close()


def _plot_pca_components(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        batch_mask: Optional[rf.Tensor] = None,
        non_blank_positions_dict: Optional[Dict] = None,
        return_axes: bool = False,
):
  assert len(x.dims) == 3

  plot_file_paths = {}

  if not return_axes:
    if os.path.exists(dirname):
      return plot_file_paths, None, None
    else:
      os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])

  x_raw = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  pca_scatter_axes = {}
  # 2d or 3d pca reduction
  dimensions = 2

  for b in range(B):
    if batch_mask is not None and not batch_mask.raw_tensor[b].item():
      continue

    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_raw_b = x_raw[b, :seq_len_b]  # [T, F]
    x_raw_b = x_raw_b.cpu().detach().numpy()

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dimensions)
    x_pca = pca.fit_transform(x_raw_b)
    fig = plt.figure()
    ax = plt.axes(projection='3d' if dimensions == 3 else None)
    ax.set_title(f"{input_name} PCA for {seq_tag}", fontsize=fontsize_axis, pad=20)
    ax.set_xlabel("PCA component 1", fontsize=fontsize_axis, labelpad=4)
    ax.set_ylabel("PCA component 2", fontsize=fontsize_axis, labelpad=4)
    if dimensions == 3:
      ax.set_zlabel("PCA component 3", fontsize=fontsize_axis, labelpad=4)
    if non_blank_positions_dict is not None:
      non_blank_positions = non_blank_positions_dict[seq_tag]
      for i, pair in enumerate(x_pca):
        # optionally: highlight first 5 positions in red
        # if i in range(5):
        #   scat = ax.scatter(pair[0], pair[1], marker='x' if i in non_blank_positions else 'o', color="red")
        # else:
        scat = ax.scatter(*pair, marker='x' if i in non_blank_positions else 'o', c=i, cmap="summer", vmin=0,
                          vmax=x_pca.shape[0] - 1)
    else:
      scat = ax.scatter(x_pca[:, 0], x_pca[:, 1], c=np.arange(x_pca.shape[0]), cmap="summer")
    fig.tight_layout(pad=5)
    if not return_axes:
      cax = fig.add_axes(
        [ax.get_position().x1 + (0.1 if dimensions == 3 else 0.01),
         ax.get_position().y0, 0.02, ax.get_position().height])
      cbar = plt.colorbar(scat, cax=cax)
      cbar.set_label("Time steps", fontsize=fontsize_axis, labelpad=4)
      plt.savefig(os.path.join(dirname, f"pca_{seq_tag.replace('/', '_')}.png"))
    else:
      pca_scatter_axes[b] = ax

    plt.close()

  return plot_file_paths, pca_scatter_axes if return_axes else None


def _plot_cosine_sim_matrix(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        extra_name: Optional[str] = None,
        batch_mask: Optional[rf.Tensor] = None,
        non_blank_positions_dict: Optional[Dict] = None,
        highlight_position: Optional[int] = None,
        return_axes: bool = False,
):
  """
  Visualizes the cosine similarity matrix of the given tensor `x` for each batch and saves the plots as images.

  Args:
      x (rf.Tensor): A 3-dimensional tensor with dimensions [batch_dim, spatial_dim, feature_dim].
      input_name (str): Name associated with the input data, used for directory naming.
      seq_tags (rf.Tensor): Tensor containing sequence tags for each batch, used in naming the output files.
      batch_dim (rf.Dim): Batch dimension of the input tensor `x`.
      spatial_dim (rf.Dim): Spatial dimension (typically sequence length) of the input tensor `x`.
      extra_name (Optional[str], optional): Additional string to append to filenames. Default is None.
      batch_mask (Optional[rf.Tensor], optional): Mask tensor to exclude specific batches from visualization. Default is None.

  Raises:
      AssertionError: If the input tensor `x` does not have exactly 3 dimensions.

  The function performs the following steps:
  1. Creates a directory named `{input_name}_cosine_sim` if it doesn't already exist.
  2. Transposes the input tensor to have the shape [batch_dim, spatial_dim, feature_dim].
  3. Normalizes the tensor along the feature dimension.
  4. For each batch in the input tensor:
     - Computes the cosine similarity matrix for the sequence data in that batch.
     - Saves a plot of the cosine similarity matrix.
     - Applies spectral clustering to the normalized features and saves a plot of the cluster labels.
  """
  print(f"Plotting cosine similarity matrix for {input_name}")
  assert len(x.dims) == 3

  plot_file_paths = {}

  if not return_axes:
    if os.path.exists(dirname):
      return plot_file_paths, None, None
    else:
      os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])

  feature_axis_int = x.get_axis_from_description(feature_dim)
  x_raw = x.raw_tensor
  x_norm = x_raw / x_raw.norm(dim=feature_axis_int, keepdim=True)

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  cosine_axes = {}

  upsampling_factor = 6

  for b in range(B):
    if batch_mask is not None and not batch_mask.raw_tensor[b].item():
      continue

    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_norm_b = x_norm[b, :seq_len_b]
    cos_sim = torch.matmul(x_norm_b, x_norm_b.transpose(0, 1))
    # cos_sim = torch.tril(cos_sim, diagonal=0)
    cos_sim = cos_sim.cpu().detach().numpy()

    fig = plt.figure()
    cosine_ax = plt.axes()
    mat = cosine_ax.matshow(cos_sim, cmap="seismic", vmin=-1, vmax=1)
    # cosine_ax.set_title(f"{input_name} Cosine similarity matrix for \n {seq_tag}", fontsize=12, pad=20)
    cosine_ax.xaxis.set_ticks_position('bottom')

    time_step_size = 1 / 60 * 500
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    cosine_ax.set_xticks(time_ticks)
    cosine_ax.set_yticks(time_ticks)
    tick_labels = [(time_tick * 60) / 1000 for time_tick in time_ticks]
    cosine_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels])
    cosine_ax.set_yticklabels([f"{label:.1f}" for label in tick_labels])

    cosine_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    cosine_ax.set_ylabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    cosine_ax.invert_yaxis()

    # show top axis with non-blank positions
    if non_blank_positions_dict is not None:
      non_blank_positions = non_blank_positions_dict[seq_tag]
      num_non_blanks = len(non_blank_positions)

      ref_align_axis = cosine_ax.secondary_xaxis('top')
      ref_align_axis.set_xticks(non_blank_positions)
      ref_align_axis.tick_params("x", length=4)
      ref_align_axis.set_xlabel("Ref. non-blank positions", fontsize=fontsize_axis, labelpad=4)

      # the second condition is needed if we are at the EOS token, because this won't be in the ref alignment
      if highlight_position is not None and highlight_position < num_non_blanks:
        # draw red triangle pointing to the highlighted position
        ref_align_axis.set_xticklabels(
          [""] * highlight_position + ["\u25BC"] + [""] * (num_non_blanks - highlight_position - 1),
          color='red',
        )
        ref_align_axis.tick_params("x", pad=1)
      else:
        ref_align_axis.set_xticklabels([])

    fig.tight_layout()

    if not return_axes:
      cax = fig.add_axes(
        [cosine_ax.get_position().x1 + 0.01, cosine_ax.get_position().y0, 0.02, cosine_ax.get_position().height])
      plt.colorbar(mat, cax=cax)

      plot_file_path_png = os.path.join(dirname,
                                        f"cos_sim_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}.png")
      plot_file_path_pdf = os.path.join(dirname,
                                        f"cos_sim_{seq_tag.replace('/', '_')}_{extra_name if extra_name else ''}.pdf")
      plt.savefig(plot_file_path_png)
      plt.savefig(plot_file_path_pdf)
      plot_file_paths[seq_tag] = plot_file_path_png
    else:
      cosine_axes[b] = cosine_ax

    plt.close()

  return plot_file_paths, cosine_axes if return_axes else None


def _plot_multi_cosine_sim_matrix_one_fig(
        xs: List[rf.Tensor],
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        extra_name: Optional[str] = None,
        batch_mask: Optional[rf.Tensor] = None,
        non_blank_positions_dict: Optional[Dict] = None,
        highlight_position: Optional[int] = None
):
  """
  Plots multiple cosine similarity matrices in a single figure, arranging them in a 2x4 grid for each sequence.

  Args:
      xs (List[rf.Tensor]): A list of tensors, each representing a feature set for which cosine similarity
                            matrices will be plotted. The tensors should have dimensions [batch_dim, spatial_dim, feature_dim].
      input_name (str): The base name used for the title and filename of the plots.
      dirname (str): The directory where the plots will be saved.
      seq_tags (rf.Tensor): Tensor containing sequence tags for each batch, used in naming the output files.
      batch_dim (rf.Dim): Batch dimension of the input tensors in `xs`.
      spatial_dim (rf.Dim): Spatial dimension (typically sequence length) of the input tensors in `xs`.
      extra_name (Optional[str], optional): Additional string to append to filenames. Default is None.
      batch_mask (Optional[rf.Tensor], optional): Mask tensor to exclude specific batches from visualization. Default is None.
      non_blank_positions_dict (Optional[Dict], optional): Dictionary indicating non-blank positions for special highlighting. Default is None.
      highlight_position (Optional[int], optional): Position to highlight in the plots. Default is None.

  The function performs the following steps:
  1. Calls `_plot_cosine_sim_matrix` for each tensor in `xs` and stores the resulting axes.
  2. For each sequence tag, creates a figure with a 2x4 grid of subplots.
  3. Plots the cosine similarity matrices from the stored axes into the grid.
  4. Adjusts the layout, titles, and labels of the figure.
  5. Saves the figure to the specified directory.

  The final figure shows the cosine similarity matrices side-by-side for easy comparison across different feature sets.
  """

  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  n_rows = 2
  n_cols = 4

  cosine_axes = {}
  # activation_axes = {}
  for i, x in enumerate(xs):
    _, cosine_axes_ = _plot_cosine_sim_matrix(
      x=x,
      input_name=input_name,
      dirname=f"{input_name}/cosine_sim",
      seq_tags=seq_tags,
      batch_dim=batch_dim,
      spatial_dim=spatial_dim,
      extra_name=extra_name,
      batch_mask=batch_mask,
      non_blank_positions_dict=non_blank_positions_dict,
      highlight_position=highlight_position,
      return_axes=True,
    )
    # _, activation_axes_ = _plot_activation_matrix(
    #   x=x,
    #   input_name=input_name,
    #   dirname=dirname,
    #   seq_tags=seq_tags,
    #   batch_dim=batch_dim,
    #   spatial_dim=spatial_dim,
    #   extra_name=extra_name,
    #   batch_mask=batch_mask,
    #   non_blank_positions_dict=non_blank_positions_dict,
    #   highlight_position=highlight_position,
    #   return_axes=True,
    # )
    cosine_axes[i] = cosine_axes_
    # activation_axes[i] = activation_axes_

  for single_axes, single_axes_name in [
    (cosine_axes, "cosine_sim"),
    # (activation_axes, "activation"),
  ]:
    for b in range(seq_tags.raw_tensor.size):
      seq_tag = seq_tags.raw_tensor[b].item()

      # Create a figure for the 2x4 grid
      fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
      fig.suptitle(f'{input_name}_{single_axes_name} for sequence {seq_tag}')

      for h in range(len(xs)):
        ax = single_axes[h][b]

        row_idx = h // n_cols
        col_idx = h % n_cols

        # Clear the subplot and plot the data from the returned axes
        axes[row_idx, col_idx].clear()
        for line in ax.get_lines():
          axes[row_idx, col_idx].plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
        for image in ax.get_images():
          axes[row_idx, col_idx].imshow(image.get_array(), cmap=image.get_cmap(), norm=image.norm)
        # for collection in ax.collections:
        #   new_collection = collection.__class__(collection.get_paths(), **collection.properties())
        #   axes[row_idx, col_idx].add_collection(new_collection)
        axes[row_idx, col_idx].set_title(ax.get_title())
        axes[row_idx, col_idx].set_xlabel(ax.get_xlabel())
        axes[row_idx, col_idx].set_ylabel(ax.get_ylabel())
        axes.invert_yaxis()

      # Adjust layout and save the figure
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])

      plt.savefig(os.path.join(dirname, f"{input_name}_{single_axes_name}_{seq_tag.replace('/', '_')}.png"))
      plt.close(fig)


def _blank_model_analysis(
        model: SegmentalAttentionModel,
        x: rf.Tensor,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        non_blank_targets_spatial_dim: rf.Dim,
        align_targets_spatial_dim: rf.Dim,
        non_blank_mask: rf.Tensor,
        enc_spatial_dim: rf.Dim,
        label_model_states: Optional[rf.Tensor] = None,
        non_blank_positions_dict: Optional[Dict] = None,
):
  assert len(x.dims) == 3
  assert model.blank_decoder_version in (3, 4, 8)

  dirname = f"{dirname}_blank_model_analysis"

  label_model_states_unmasked = utils.get_unmasked(
    label_model_states,
    input_spatial_dim=non_blank_targets_spatial_dim,
    mask=non_blank_mask,
    mask_spatial_dim=align_targets_spatial_dim,
  )
  label_model_states_unmasked = utils.copy_tensor_replace_dim_tag(
    label_model_states_unmasked,
    align_targets_spatial_dim,
    enc_spatial_dim
  )

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  if model.blank_decoder_version == 8:
    blank_model_input = x
  else:
    blank_model_input = rf.concat_features(x, label_model_states_unmasked, allow_broadcast=False)

  def _apply_blank_model(blank_model_input_: rf.Tensor):
    if model.blank_decoder_version == 3:
      x_transformed_, _ = model.blank_decoder.loop_step(
        enc=x,
        enc_spatial_dim=enc_spatial_dim,
        label_model_state=label_model_states_unmasked,
        state=model.blank_decoder.default_initial_state(batch_dims=[batch_dim]),
        spatial_dim=enc_spatial_dim,
      )
      x_transformed_ = x_transformed_["s_blank"]
    else:
      x_transformed_ = model.blank_decoder.s(blank_model_input_)
      x_transformed_ = rf.reduce_out(
        x_transformed_, mode="max", num_pieces=2, out_dim=model.blank_decoder.emit_prob.in_dim)

    return x_transformed_

  x_transformed = _apply_blank_model(blank_model_input)

  assert len(x_transformed.dims) == 3
  _plot_cosine_sim_matrix(
    x=x_transformed,
    dirname=f"{dirname}_merged/cosine_sim",
    input_name="Blank model features",
    seq_tags=seq_tags,
    batch_dim=batch_dim,
    spatial_dim=enc_spatial_dim,
    non_blank_positions_dict=non_blank_positions_dict,
  )

  # optionally plot per label step
  # if model.blank_decoder_version == 4:
  #   blank_model_input = rf.concat_features(x, label_model_states, allow_broadcast=True)
  #   x_transformed = _apply_blank_model(blank_model_input)
  #
  #   assert len(x_transformed.dims) == 4
  #   S = rf.reduce_max(
  #     non_blank_targets_spatial_dim.dyn_size_ext,
  #     axis=non_blank_targets_spatial_dim.dyn_size_ext.dims
  #   ).raw_tensor.item()
  #
  #   for s in range(S):
  #     _plot_cosine_sim_matrix(
  #       x=rf.gather(x_transformed, axis=non_blank_targets_spatial_dim, indices=rf.constant(s, dims=[batch_dim])),
  #       dirname=f"{dirname}_per_step/s_{s}",
  #       input_name=f"Blank model features step {s}",
  #       seq_tags=seq_tags,
  #       batch_dim=batch_dim,
  #       spatial_dim=enc_spatial_dim,
  #       batch_mask=non_blank_targets_spatial_dim.dyn_size_ext > s,
  #     )


def _plot_multi_head_enc_self_att_one_fig(
        att_weights: rf.Tensor,
        energies: rf.Tensor,
        head_dim: rf.Dim,
        batch_dims: List[rf.Dim],
        enc_query_dim: rf.Dim,
        enc_kv_dim: rf.Dim,
        dirname: str,
        seq_tags: rf.Tensor,
):
  print(f"Plotting multi-head self-attention for {dirname}")

  # Calculate grid size
  num_heads = head_dim.dimension
  grid_rows, grid_cols = 2, 4

  for tensor, tensor_name in (
          # (att_weights, "att_weights"),
          (energies, "energies"),
  ):
    head_dirname = os.path.join(dirname, f"{tensor_name}_grid")
    if not os.path.exists(head_dirname):
      os.makedirs(head_dirname)

    for b in range(tensor.raw_tensor.size(0)):
      seq_tag = seq_tags.raw_tensor[b].item()
      seq_len_b = enc_query_dim.dyn_size_ext.raw_tensor[b].item()

      # Create a figure for the 2x4 grid
      fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(20, 10))
      # fig.suptitle(f'{tensor_name} for sequence {seq_tag}')

      for h in range(num_heads):
        tensor_single_head = rf.gather(
          tensor,
          indices=rf.constant(h, dims=batch_dims),
          axis=head_dim,
        )
        dummy_att_head_dim = rf.Dim(1, name="att_head")
        tensor_single_head = rf.expand_dim(tensor_single_head, dim=dummy_att_head_dim)
        tensor_single_head = tensor_single_head.copy_transpose(
          batch_dims + [enc_query_dim, enc_kv_dim, dummy_att_head_dim])

        tensor_single_head_raw = tensor_single_head.raw_tensor
        tensor_single_head_b_raw = tensor_single_head_raw[b, :seq_len_b, :seq_len_b, 0]
        tensor_single_head_b_raw = tensor_single_head_b_raw.cpu().detach().numpy()

        # Determine the current grid position
        row_idx = h // grid_cols
        col_idx = h % grid_cols

        # Plot in the appropriate grid position
        ax = axes[row_idx, col_idx]
        ax.matshow(tensor_single_head_b_raw, cmap="Blues")
        ax.set_title(f"Head {h}", fontsize=fontsize_axis, pad=10)
        ax.set_ylabel("Queries time (s)", fontsize=fontsize_axis, labelpad=4)
        ax.set_xlabel("Keys/Values time (s)", fontsize=fontsize_axis, labelpad=4)

        time_step_size = 1 / 60 * 500
        time_ticks = np.arange(0, seq_len_b, time_step_size)
        ax.set_xticks(time_ticks)
        ax.set_yticks(time_ticks)
        tick_labels = [(time_tick * 60) / 1000 for time_tick in time_ticks]
        ax.set_xticklabels([f"{label:.1f}" for label in tick_labels], fontsize=fontsize_ticks)
        ax.set_yticklabels([f"{label:.1f}" for label in tick_labels], fontsize=fontsize_ticks)
        ax.xaxis.set_ticks_position('bottom')

        ax.invert_yaxis()

      # Adjust layout and save the figure
      plt.tight_layout(rect=[0, 0.03, 1, 0.95])

      plt.savefig(os.path.join(head_dirname, f"{tensor_name}_{seq_tag.replace('/', '_')}.png"))
      plt.savefig(os.path.join(head_dirname, f"{tensor_name}_{seq_tag.replace('/', '_')}.pdf"))
      plt.close(fig)


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
        plt.savefig(os.path.join(head_dirname, f"{tensor_name}_{seq_tag.replace('/', '_')}.pdf"))
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

  if isinstance(model, CtcModel):
    target_blank_idx = model.blank_idx
  elif isinstance(model.label_decoder, SegmentalAttLabelDecoder):
    target_blank_idx = model.blank_idx
  else:
    target_blank_idx = None

  max_num_labels = rf.reduce_max(non_blank_targets_spatial_dim.dyn_size_ext, axis=batch_dims).raw_tensor.item()

  ref_alignment_hdf = Path(config.typed_value("ref_alignment_hdf", str))
  ref_alignment_blank_idx = config.typed_value("ref_alignment_blank_idx", int)
  ref_alignment_vocab_path = Path(config.typed_value("ref_alignment_vocab_path", str))
  json_vocab_path = Path(config.typed_value("json_vocab_path", str))
  with open(ref_alignment_vocab_path, "r") as f:
    ref_alignment_vocab = ast.literal_eval(f.read())
    ref_alignment_vocab = {v: k for k, v in ref_alignment_vocab.items()}

  ref_alignment_dict = hdf.load_hdf_data(ref_alignment_hdf)
  non_blank_positions_dict = {}
  ref_non_blank_targets_dict = {}
  for seq_tag in ref_alignment_dict:
    non_blank_positions_dict[seq_tag] = np.where(ref_alignment_dict[seq_tag] != ref_alignment_blank_idx)[0]
    ref_non_blank_targets_dict[seq_tag] = [
      ref_alignment_vocab[idx] for idx in ref_alignment_dict[seq_tag][non_blank_positions_dict[seq_tag]]
    ]

  with torch.enable_grad():
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

    x_linear = process_captured_tensors(
      layer_mapping={"x_linear": (type(model.encoder).__call__, 0, "x_linear", -1)},
    )
    assert isinstance(x_linear, rf.Tensor)

    if config.bool("plot_encoder_gradient_graph", False):
      _plot_encoder_gradient_graph(
        layers=[collected_outputs[k] for k in collected_outputs if k != "encoder_input" and k in ["7", "8"]],
        # layers=[collected_outputs[k] for k in collected_outputs if k in ["7", "8", "9"]],
        dirname="encoder_gradient_graph",
        seq_tags=seq_tags,
        batch_dim=batch_dims[0],
        spatial_dim=enc_spatial_dim,
      )

    if isinstance(model.encoder, ConformerEncoder) and config.bool("plot_encoder_layers", False):
      for enc_layer_idx in range(len(model.encoder.layers)):
        print(f"Processing encoder layer {enc_layer_idx}")
        # plot self-attention queries, keys and values
        # for tensor_var_name, tensor_name in [
        #   ("q", f"enc-{enc_layer_idx}_att_query"),
        #   ("k", f"enc-{enc_layer_idx}_att_key"),
        #   ("v", f"enc-{enc_layer_idx}_att_value"),
        # ]:
        #   tensor = process_captured_tensors(
        #     layer_mapping={tensor_var_name: (RelPosSelfAttention.__call__, enc_layer_idx, tensor_var_name, 0)},
        #   )
        #   assert isinstance(tensor, rf.Tensor)
        #
        #   head_dim = [dim for dim in tensor.dims if dim.dimension == 8][0]
        #   tensor_heads = []
        #   for h in range(head_dim.dimension):
        #     enc_att_queries_single_head = rf.gather(
        #       tensor,
        #       indices=rf.constant(h, dims=batch_dims),
        #       axis=head_dim,
        #     )
        #     tensor_heads.append(enc_att_queries_single_head)
        #
        #   _plot_multi_cosine_sim_matrix_one_fig(
        #     xs=tensor_heads,
        #     input_name=f"{tensor_name}_heads",
        #     dirname=f"{tensor_name}_heads_cos_sim_and_pca",
        #     batch_dim=batch_dims[0],
        #     seq_tags=seq_tags,
        #     spatial_dim=enc_spatial_dim,
        #     non_blank_positions_dict=non_blank_positions_dict,
        #   )

        tensor_names = [
          "x_ffn1",
          "x_ffn1_out",
          "x_mhsa",
          "x_mhsa_out",
          # "x_conv",
          # "x_conv_out",
          "x_ffn2",
          "x_ffn2_out"]
        activation_magnitude_file_path = f"enc-{enc_layer_idx}/intermediate_layer_activations/activation_magnitude.txt"
        # if hasattr(enc_layer_cls, "conv_block"):
        #   tensor_names.append("x_conv_out")
        tensors = []
        for tensor_name in tensor_names:
          tensor = process_captured_tensors(
            layer_mapping={tensor_name: (enc_layer_cls.__call__, enc_layer_idx, tensor_name, -1)},
          )
          assert isinstance(tensor, rf.Tensor)
          tensors.append(tensor)
        for tensor, tensor_name in zip(tensors, tensor_names):
          max_values = rf.maximum(*tensors)
          max_values = rf.reduce_max(max_values, axis=enc_spatial_dim)
          max_values = rf.reduce_max(max_values, axis=max_values.remaining_dims(batch_dims))

          min_values = rf.minimum(*tensors)
          min_values = rf.reduce_min(min_values, axis=enc_spatial_dim)
          min_values = rf.reduce_min(min_values, axis=min_values.remaining_dims(batch_dims))

          max_values = rf.maximum(max_values, rf.abs(min_values))
          min_values = rf.zeros_like(min_values)

          _plot_activation_matrix(
            x=rf.abs(tensor),
            input_name=tensor_name,
            dirname=f"enc-{enc_layer_idx}/intermediate_layer_activations/{tensor_name}",
            seq_tags=seq_tags,
            batch_dim=batch_dims[0],
            spatial_dim=enc_spatial_dim,
            extra_name=None,
            batch_mask=None,
            # non_blank_positions_dict=non_blank_positions_dict,
            max_values=max_values,
            min_values=min_values,
            non_blank_targets_dict=ref_non_blank_targets_dict,
            activation_magnitude_file_path=activation_magnitude_file_path,
          )
          # _plot_pca_components(
          #   x=tensor,
          #   input_name=tensor_name,
          #   dirname=f"enc-{enc_layer_idx}/intermediate_layer_pca/{tensor_name}",
          #   seq_tags=seq_tags,
          #   batch_dim=batch_dims[0],
          #   spatial_dim=enc_spatial_dim,
          #   batch_mask=None,
          #   non_blank_positions_dict=non_blank_positions_dict,
          # )
          # _plot_neurons(
          #   x=tensor,
          #   input_name=tensor_name,
          #   dirname=f"enc-{enc_layer_idx}/intermediate_layer_neurons/{tensor_name}",
          #   seq_tags=seq_tags,
          #   batch_dim=batch_dims[0],
          #   spatial_dim=enc_spatial_dim,
          # )

        # plot self-attention weights and energies
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
        _plot_multi_head_enc_self_att_one_fig(
          att_weights=enc_att_weights,
          energies=enc_att_energies,
          head_dim=enc_att_head_dim,
          batch_dims=batch_dims,
          enc_query_dim=enc_spatial_dim,
          enc_kv_dim=enc_kv_dim,
          dirname=f"enc-{enc_layer_idx}/self_att",
          seq_tags=seq_tags,
        )

    if isinstance(model, CtcModel):
      import torchaudio
      # dump targets to hdf file
      dump_hdfs(
        batch_dims=batch_dims,
        seq_tags=seq_tags,
        align_targets=non_blank_targets,
        align_target_dim=model.target_dim.dimension
      )

      layer_idx = len(model.encoder.layers)
      ctc_projection = getattr(model.encoder, f"enc_aux_logits_{layer_idx}")
      aux_logits = ctc_projection(collected_outputs[str(layer_idx - 1)])
      ctc_log_probs = rf.log_softmax(aux_logits, axis=model.align_target_dim)

      non_blank_positions = rf.zeros(dims=batch_dims + [non_blank_targets_spatial_dim], dtype="int32")
      enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
      non_blank_targets_spatial_sizes = rf.copy_to_device(non_blank_targets_spatial_dim.dyn_size_ext)
      for b in range(seq_tags.raw_tensor.size):
        input_len_b = enc_spatial_sizes.raw_tensor[b]
        target_len_b = non_blank_targets_spatial_sizes.raw_tensor[b]
        ctc_alignment, _ = torchaudio.functional.forced_align(
          log_probs=ctc_log_probs.raw_tensor[b, :input_len_b][None],
          targets=non_blank_targets.raw_tensor[b, :target_len_b][None],
          input_lengths=input_len_b[None],
          target_lengths=target_len_b[None],
          blank=model.blank_idx,
        )
        ctc_positions = ctc_align_get_center_positions(ctc_alignment, model.blank_idx)

        non_blank_positions.raw_tensor[b, :len(ctc_positions)] = ctc_positions

      ctc_log_probs = rf.gather(
        ctc_log_probs,
        indices=non_blank_positions,
        axis=enc_spatial_dim,
      )

      tensors = [
        *[(f"enc-{i}", collected_outputs[str(i)]) for i in range(len(model.encoder.layers))],
        ("x_linear", x_linear),
      ]
      for input_name, input_ in tensors:
        _plot_log_prob_gradient_wrt_to_input_single_seq(
          input_=input_,
          log_probs=ctc_log_probs,
          targets=non_blank_targets,
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          enc_spatial_dim=enc_spatial_dim,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          json_vocab_path=json_vocab_path,
          ref_alignment_hdf=ref_alignment_hdf,
          ref_alignment_blank_idx=ref_alignment_blank_idx,
          ref_alignment_json_vocab_path=ref_alignment_vocab_path,
          dirname=f"{input_name}/log-prob-grads_wrt_{input_name}_log-space"
        )
      return

    # ----------------------------------------------------------------------------------

    for enc_layer_idx in range(len(model.encoder.layers) - 1, len(model.encoder.layers)):
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

      if not isinstance(model.label_decoder, TransformerDecoder) and hasattr(model.encoder, "enc_ctx"):
        enc_args["enc_ctx"] = model.encoder.enc_ctx(enc)  # does not exist for transformer decoder

        if model.label_decoder.use_weight_feedback:
          enc_args["inv_fertility"] = rf.sigmoid(
            model.encoder.inv_fertility(enc))  # does not exist for transformer decoder
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
              f"att_weight_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "att_weights", -1) for i in
              range(max_num_labels)},
            process_func=_replace_slice_dim
          )
          att_weights, _ = rf.stack(att_weights, out_dim=non_blank_targets_spatial_dim)

          if model.center_window_size != 1 and model.label_decoder.gaussian_att_weight_opts is None:
            energies = process_captured_tensors(
              layer_mapping={
                f"energy_step{i}": (SegmentalAttLabelDecoder.loop_step, i, "energy", -1) for i in
                range(max_num_labels)},
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

          log_probs_wo_att = None
          log_probs_wo_h_t = None
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

          log_probs_wo_att = None
          log_probs_wo_h_t = None
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

      elif type(model.label_decoder) is GlobalAttEfficientDecoder and not model.label_decoder.trafo_att:
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

        if model.label_decoder.use_current_frame_in_readout or model.label_decoder.use_current_frame_in_readout_w_double_gate:
          logits_wo_att, _ = forward_sequence_global(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=non_blank_targets,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            center_positions=center_positions,
            batch_dims=batch_dims,
            detach_att_before_readout=True,
          )
          log_probs_wo_att = rf.log_softmax(logits_wo_att, axis=model.target_dim)
          log_probs_wo_att = log_probs_wo_att.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])
          logits_wo_h_t, _ = forward_sequence_global(
            model=model.label_decoder,
            enc_args=enc_args,
            enc_spatial_dim=enc_spatial_dim,
            targets=non_blank_targets,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            center_positions=center_positions,
            batch_dims=batch_dims,
            detach_h_t_before_readout=True,
          )
          log_probs_wo_h_t = rf.log_softmax(logits_wo_h_t, axis=model.target_dim)
          log_probs_wo_h_t = log_probs_wo_h_t.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])
        else:
          log_probs_wo_att = None
          log_probs_wo_h_t = None
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

        try:
          energy_in = process_captured_tensors(
            layer_mapping={
              f"energy_step{i}": (GlobalAttDecoder.loop_step, i, "energy_in", -1) for i in range(max_num_labels)},
          )
          energy_in, _ = rf.stack(energy_in, out_dim=non_blank_targets_spatial_dim)
        except KeyError:
          energy_in = None

        try:
          energies = process_captured_tensors(
            layer_mapping={
              f"energy_step{i}": (GlobalAttDecoder.loop_step, i, "energy", -1) for i in range(max_num_labels)},
          )
          energies, _ = rf.stack(energies, out_dim=non_blank_targets_spatial_dim)
        except KeyError:
          energies = None

        label_model_states = process_captured_tensors(
          layer_mapping={
            f"s_step{i}": (GlobalAttDecoder.loop_step, i, "s", -1) for i in range(max_num_labels)},
        )
        label_model_states, _ = rf.stack(label_model_states, out_dim=non_blank_targets_spatial_dim)
        label_model_states.feature_dim = label_model_states.remaining_dims(
          batch_dims + [non_blank_targets_spatial_dim])[0]

        log_probs_wo_att = None
        log_probs_wo_h_t = None

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
        assert isinstance(model.label_decoder, TransformerDecoder) or (
                type(model.label_decoder) is GlobalAttEfficientDecoder and model.label_decoder.trafo_att)

        for get_cross_attentions in [True, False]:
          for dec_layer_idx in range(5, 6):
            funcs_to_trace_list = [
              TransformerDecoder.__call__ if isinstance(
                model.label_decoder, TransformerDecoder) else TrafoAttention.__call__,
              TransformerDecoderLayer.__call__,
              rf.CrossAttention.__call__,
              rf.CrossAttention.attention,
              rf.dot_attention,
            ]
            if not isinstance(model.label_decoder, TransformerDecoder):
              funcs_to_trace_list += [
                forward_sequence_global,
                get_s_and_att_efficient_global,
                GlobalAttEfficientDecoder.__call__,
                GlobalAttEfficientDecoder.decode_logits,
              ]
            set_trace_variables(
              funcs_to_trace_list=funcs_to_trace_list
            )

            sys.settrace(_trace_func)
            if isinstance(model.label_decoder, TransformerDecoder):
              model.label_decoder(
                non_blank_targets,
                spatial_dim=non_blank_targets_spatial_dim,
                encoder=model.label_decoder.transform_encoder(enc_args["enc"], axis=enc_spatial_dim),
                state=model.label_decoder.default_initial_state(batch_dims=batch_dims)
              )
            else:
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

            _dec_layer_idx = 2 * dec_layer_idx + int(get_cross_attentions)
            if model.label_decoder.use_trafo_att_wo_cross_att:
              att_weights = None
              energies = None
            else:
              att_weights = process_captured_tensors(
                layer_mapping={"att_weights": (rf.dot_attention, _dec_layer_idx, "att_weights", -1)}
              )
              assert isinstance(att_weights, rf.Tensor)
              energies = process_captured_tensors(
                layer_mapping={"energy": (rf.dot_attention, _dec_layer_idx, "energy", -1)}
              )
              assert isinstance(energies, rf.Tensor)

            if isinstance(model.label_decoder, TransformerDecoder):
              logits = process_captured_tensors(
                layer_mapping={"logits": (TransformerDecoder.__call__, 0, "logits", -1)}
              )
              log_probs_wo_att = None
              log_probs_wo_h_t = None
            else:
              label_model_states = process_captured_tensors(
                layer_mapping={f"s": (get_s_and_att_efficient_global, 0, "s", -1)},
              )
              assert isinstance(label_model_states, rf.Tensor)

              logits = process_captured_tensors(
                layer_mapping={"logits": (GlobalAttEfficientDecoder.decode_logits, 0, "logits", -1)}
              )

              logits_wo_att, _ = forward_sequence_global(
                model=model.label_decoder,
                enc_args=enc_args,
                enc_spatial_dim=enc_spatial_dim,
                targets=non_blank_targets,
                targets_spatial_dim=non_blank_targets_spatial_dim,
                center_positions=center_positions,
                batch_dims=batch_dims,
                detach_att_before_readout=True,
              )
              log_probs_wo_att = rf.log_softmax(logits_wo_att, axis=model.target_dim)
              log_probs_wo_att = log_probs_wo_att.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])
              logits_wo_h_t, _ = forward_sequence_global(
                model=model.label_decoder,
                enc_args=enc_args,
                enc_spatial_dim=enc_spatial_dim,
                targets=non_blank_targets,
                targets_spatial_dim=non_blank_targets_spatial_dim,
                center_positions=center_positions,
                batch_dims=batch_dims,
                detach_h_t_before_readout=True,
              )
              log_probs_wo_h_t = rf.log_softmax(logits_wo_h_t, axis=model.target_dim)
              log_probs_wo_h_t = log_probs_wo_h_t.copy_transpose(batch_dims + [non_blank_targets_spatial_dim, model.target_dim])
            assert isinstance(logits, rf.Tensor)
            log_probs = rf.log_softmax(logits, axis=model.target_dim)

            # shapes for trafo decoder
            # print("att_weights", att_weights)  # [B, H, S, T] -> [B, T, S, 1]
            # print("energies", energies)  # [B, H, S, T]
            # print("logits", logits)  # [B, S, V]

            # print("x_linear", x_linear)  # [B, T, F]
            # print("logits", logits)  # [B, S, V]

            def _plot_attention_weights():
              if get_cross_attentions:
                att_head_dim = att_weights.remaining_dims(batch_dims + [enc_spatial_dim, non_blank_targets_spatial_dim])
                assert len(att_head_dim) == 1
                att_head_dim = att_head_dim[0]
              else:
                # this is either the att head dim or the kv-dim
                att_head_and_kv_dim = att_weights.remaining_dims(batch_dims + [non_blank_targets_spatial_dim])
                # att head dim is static, kv-dim is dynamic
                att_head_dim = [dim for dim in att_head_and_kv_dim if dim.is_static()][0]
                kv_dim = [dim for dim in att_head_and_kv_dim if dim.is_dynamic()][0]
                # kv_dim has seq lens (1, 2, ..., S) but needs single len (S) for my hdf_dump script
                kv_dim.dyn_size_ext.raw_tensor = non_blank_targets_spatial_dim.dyn_size_ext.raw_tensor.clone()

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
                      batch_dims + [non_blank_targets_spatial_dim, enc_spatial_dim, dummy_att_head_dim])
                  else:
                    # [B, S, S, 1]
                    tensor_single_head = tensor_single_head.copy_transpose(
                      batch_dims + [non_blank_targets_spatial_dim, kv_dim, dummy_att_head_dim])
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

            if not model.label_decoder.use_trafo_att_wo_cross_att:
              _plot_attention_weights()

      # optionally plot energies for every s = 1...S
      # if isinstance(model.label_decoder, GlobalAttDecoder) and not model.label_decoder.trafo_att:
      #   # store the paths to the plots for the energies
      #   energy_plot_paths = {}
      #   for s in range(max_num_labels):
      #     energy_in_s = rf.gather(
      #       energy_in,
      #       indices=rf.constant(s, dims=batch_dims),
      #       axis=non_blank_targets_spatial_dim,
      #     )
      #     energy_in_s = energy_in_s.copy_transpose(batch_dims + [enc_spatial_dim, model.label_decoder.enc_key_total_dim])
      #
      #     for seq_tag, path in _plot_cosine_sim_matrix(
      #         energy_in_s,
      #         dirname=f"energy_in_analysis/s-{s}",
      #         input_name=f"energy_in_s-{s}",
      #         batch_dim=batch_dims[0],
      #         seq_tags=seq_tags,
      #         spatial_dim=enc_spatial_dim,
      #         extra_name=str(s),
      #         batch_mask=non_blank_targets_spatial_dim.dyn_size_ext > s,
      #         non_blank_positions_dict=non_blank_positions_dict,
      #         highlight_position=s,
      #       ).items():
      #       if seq_tag not in energy_plot_paths:
      #         energy_plot_paths[seq_tag] = []
      #       if s < len(non_blank_positions_dict[seq_tag]):
      #         energy_plot_paths[seq_tag].append(path)
      #
      #   # create gifs for the energies going from s=1...S
      #   for tag, paths in energy_plot_paths.items():
      #     images = []
      #     for file_path in paths:
      #       images.append(imageio.imread(file_path))
      #     imageio.mimsave(f'energy_in_analysis/{tag.replace("/", "_")}.gif', images, fps=2, loop=0)

      tensors = [
        ("x_linear", x_linear),
        *[(f"enc-{i}", collected_outputs[str(i)]) for i in range(len(model.encoder.layers))],
      ]
      if "enc_ctx" in enc_args and energies is not None:
        tensors.append((f"enc_ctx-{enc_layer_idx}", enc_args["enc_ctx"]))

      log_gradient_wrt_input_list = []
      log_gradient_wrt_input_dirname_list = []
      dummy_singleton_dim = rf.Dim(1, name="dummy")
      for input_name, input_ in tensors:
        if config.bool("plot_log_gradients", False):
          dirname = f"{input_name}/log-prob-grads_wrt_{input_name}_log-space"
          log_gradient_wrt_input_dirname_list.append(dirname)
          log_gradient_wrt_input_list.append(_plot_log_prob_gradient_wrt_to_input_batched(
            input_=input_,
            log_probs=log_probs,
            targets=non_blank_targets,
            batch_dims=batch_dims,
            seq_tags=seq_tags,
            enc_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=non_blank_targets_spatial_dim,
            json_vocab_path=json_vocab_path,
            ref_alignment_hdf=ref_alignment_hdf,
            ref_alignment_blank_idx=ref_alignment_blank_idx,
            ref_alignment_json_vocab_path=ref_alignment_vocab_path,
            dirname=dirname,
            # return_gradients=True,
            dummy_singleton_dim=dummy_singleton_dim
          ))

        if config.bool("plot_encoder_layers", False):
          _plot_cosine_sim_matrix(
            input_,
            input_name=input_name,
            dirname=f"{input_name}/cosine_sim",
            batch_dim=batch_dims[0],
            seq_tags=seq_tags,
            spatial_dim=enc_spatial_dim,
            # non_blank_positions_dict=non_blank_positions_dict,
          )

      for log_prob_name, log_probs in (
              ("log-probs-wo-h_t", log_probs_wo_h_t),
              ("log-probs-wo-att", log_probs_wo_att),
      ):
        if log_probs is None:
          continue

        dirname = f"enc-11/{log_prob_name}-grads_wrt_enc-11_log-space"
        log_gradient_wrt_input_dirname_list.append(dirname)
        log_gradient_wrt_input_list.append(_plot_log_prob_gradient_wrt_to_input_batched(
          input_=collected_outputs["11"],
          log_probs=log_probs,
          targets=non_blank_targets,
          batch_dims=batch_dims,
          seq_tags=seq_tags,
          enc_spatial_dim=enc_spatial_dim,
          targets_spatial_dim=non_blank_targets_spatial_dim,
          json_vocab_path=json_vocab_path,
          ref_alignment_hdf=ref_alignment_hdf,
          ref_alignment_blank_idx=ref_alignment_blank_idx,
          ref_alignment_json_vocab_path=ref_alignment_vocab_path,
          dirname=dirname,
          # return_gradients=True,
          dummy_singleton_dim=dummy_singleton_dim
        ))

      # for plotting normalized log gradients
      # log_gradient_wrt_input_stacked, _ = rf.stack(log_gradient_wrt_input_list)
      # log_gradient_wrt_input_stacked_wo_zeros = log_gradient_wrt_input_stacked.copy()
      # log_gradient_wrt_input_stacked_wo_zeros.raw_tensor[log_gradient_wrt_input_stacked.raw_tensor == 0.0] = torch.inf
      # log_gradient_wrt_input_stacked = rf.safe_log(log_gradient_wrt_input_stacked)
      #
      # min_log_gradient_wrt_input = log_gradient_wrt_input_stacked_wo_zeros.copy()
      # max_log_gradient_wrt_input = log_gradient_wrt_input_stacked.copy()
      # for dim in log_gradient_wrt_input_stacked.remaining_dims(batch_dims):
      #   max_log_gradient_wrt_input = rf.reduce_max(max_log_gradient_wrt_input, axis=dim)
      #   min_log_gradient_wrt_input = rf.reduce_min(min_log_gradient_wrt_input, axis=dim)
      #
      # for log_gradient_wrt_input, dirname in zip(log_gradient_wrt_input_list, log_gradient_wrt_input_dirname_list):
      #   dump_hdfs(
      #     att_weights=rf.log(log_gradient_wrt_input),  # use log for better visualization
      #     batch_dims=batch_dims,
      #     dirname=dirname,
      #     seq_tags=seq_tags,
      #   )
      #
      #   vmin = {k: v.item() for k, v in zip(seq_tags.raw_tensor, min_log_gradient_wrt_input.raw_tensor)}
      #   vmax = {k: v.item() for k, v in zip(seq_tags.raw_tensor, max_log_gradient_wrt_input.raw_tensor)}
      #
      #   plot_att_weights(
      #     att_weight_hdf=Path(os.path.join(dirname, "att_weights.hdf")),
      #     targets_hdf=Path("targets.hdf"),
      #     seg_starts_hdf=None,
      #     seg_lens_hdf=None,
      #     center_positions_hdf=None,
      #     target_blank_idx=None,
      #     ref_alignment_blank_idx=ref_alignment_blank_idx,
      #     ref_alignment_hdf=ref_alignment_hdf,
      #     ref_alignment_json_vocab_path=ref_alignment_vocab_path,
      #     json_vocab_path=json_vocab_path,
      #     segment_whitelist=list(seq_tags.raw_tensor),
      #     plot_name=dirname,
      #     plot_w_color_gradient=config.bool("debug", False),
      #     # vmin=vmin,
      #     # vmax=vmax,
      #   )

      if isinstance(model, SegmentalAttentionModel) and model.blank_decoder_version in (3, 4, 8,):
        _blank_model_analysis(
          model=model,
          x=collected_outputs["11"],
          dirname=f"enc-{11}",
          batch_dim=batch_dims[0],
          seq_tags=seq_tags,
          non_blank_targets_spatial_dim=non_blank_targets_spatial_dim,
          enc_spatial_dim=enc_spatial_dim,
          label_model_states=label_model_states,
          non_blank_mask=non_blank_mask,
          align_targets_spatial_dim=targets_spatial_dim,
          non_blank_positions_dict=non_blank_positions_dict,
        )

      att_weight_tensor_list = [
        (att_weights, "att_weights"),
        (energies, "energies"),
      ]
      if config.bool("analyze_att_weight_dropout", False):
        att_weights_drop01 = rf.dropout(att_weights, 0.1, on_forward=True, axis=batch_dims + [enc_spatial_dim])
        att_weight_tensor_list.append((att_weights_drop01, "att_weights_drop01_broadcast_to_labels"))

      # plot attention weights for non-trafo decoders
      # (for trafo decoders, the attention weights are plotted in the loop above)
      if not isinstance(model.label_decoder, TransformerDecoder) and not model.label_decoder.trafo_att:
        dirname = f"cross-att/enc-layer-{enc_layer_idx + 1}"

        print(f"Plotting attention weights for {dirname}")

        for tensor, tensor_name in att_weight_tensor_list:
          if tensor is None:
            continue

          tensor_dirname = os.path.join(dirname, tensor_name)
          # if os.path.exists(tensor_dirname):
          #   continue

          assert non_blank_targets_spatial_dim in tensor.dims
          if tensor.dims != tuple(
                  batch_dims + [non_blank_targets_spatial_dim, enc_spatial_dim, model.label_decoder.att_num_heads]):
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
