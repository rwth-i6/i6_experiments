import os.path
from typing import Optional, Dict, List, Callable, Union
import sys
import ast
import contextlib

import numpy as np
import torch
import matplotlib.pyplot as plt

from returnn.tensor import TensorDict
import returnn.frontend as rf
from returnn.datasets.hdf import SimpleHDFWriter
from returnn.config import get_global_config
from returnn.frontend.attention import RelPosSelfAttention, dot_attention

from i6_experiments.users.schmitt import hdf

from sisyphus import Path
from ..model.conformer_tina import (
  TinaConformerModel,
  FramewiseProbModel,
  DiphoneFHModel,
  MonophoneFHModel,
)
from .. import tensor_utils as utils

fontsize_axis = 16
fontsize_ticks = 11


def plot_pca_components(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        ref_alignment_dict: Optional[Dict],
        ref_alignment_blank_idx: Optional[int],
        upsample_factor: int = 1,
        prob_argmax: Optional[Dict] = None,
        blank_sil_idx: int = 0,
):
  print(f"Plotting PCA components for {input_name}")
  assert len(x.dims) == 3

  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  exp_variance_file = os.path.join(dirname, "explained_variance.txt")

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]
  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])
  x_raw = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  # 2d pca reduction
  dimensions = 2

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_raw_b = x_raw[b, :seq_len_b]  # [T, F]
    x_raw_b = x_raw_b.cpu().detach().numpy()

    # normalize before applying PCA
    x_raw_b_mean = x_raw_b.mean(axis=0)
    x_raw_b_std = x_raw_b.std(axis=0)
    x_raw_b = (x_raw_b - x_raw_b_mean) / x_raw_b_std

    from sklearn.decomposition import PCA
    pca = PCA(n_components=dimensions)
    x_pca = pca.fit_transform(x_raw_b)
    fig = plt.figure()
    ax = plt.axes()
    # ax.set_title(f"{input_name} PCA for {seq_tag}", fontsize=fontsize_axis, pad=20)
    ax.set_xlabel("PCA component 1", fontsize=fontsize_axis, labelpad=4)
    ax.set_ylabel("PCA component 2", fontsize=fontsize_axis, labelpad=4)

    if prob_argmax is None:
      silence_indices = get_silence_indices_from_gmm_alignment(
        seq_tag, ref_alignment_dict[seq_tag], ref_alignment_blank_idx)
      upsampled_indices = np.arange(seq_len_b) * upsample_factor
      silence_mask = np.array([idx in silence_indices for idx in upsampled_indices])
    else:
      silence_mask = prob_argmax[seq_tag] == blank_sil_idx

    x_pca_silence = x_pca[silence_mask]
    x_pca_non_silence = x_pca[~silence_mask]
    ax.scatter(
      x_pca_silence[:, 0], x_pca_silence[:, 1], c=np.arange(x_pca_silence.shape[0]), cmap="summer", marker="x")
    scat = ax.scatter(
      x_pca_non_silence[:, 0], x_pca_non_silence[:, 1], c=np.arange(x_pca_non_silence.shape[0]), cmap="summer", marker="o")
    cax = fig.add_axes(
      [ax.get_position().x1 + (0.01),
       ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(scat, cax=cax)
    cbar.set_label("Time steps", fontsize=fontsize_axis, labelpad=4)
    plt.savefig(os.path.join(dirname, f"pca_{seq_tag.replace('/', '_')}.png"), bbox_inches="tight")
    plt.savefig(os.path.join(dirname, f"pca_{seq_tag.replace('/', '_')}.pdf"), bbox_inches="tight")

    plt.close()

    with open(exp_variance_file, "a") as f:
      f.write(f"{seq_tag}: {pca.explained_variance_ratio_}\n\n")


def set_ref_alignment_ticks(
        ax: plt.Axes,
        ref_alignment: np.ndarray,
        ref_alignment_vocab: Dict,
        ref_alignment_blank_idx: int,
):
  # add axis with ref alignment
  ref_labels = ref_alignment[ref_alignment != ref_alignment_blank_idx]
  ref_labels = [ref_alignment_vocab[idx] for idx in ref_labels]
  # positions of reference labels in the reference alignment
  # +1 bc plt starts at 1, not at 0
  ref_label_positions = np.where(ref_alignment != ref_alignment_blank_idx)[-1] + 1
  vertical_lines = [tick - 1.0 for tick in ref_label_positions]
  ref_label_positions = np.concatenate([[0], ref_label_positions])
  ref_segment_sizes = ref_label_positions[1:] - ref_label_positions[:-1]
  xticks = ref_label_positions[:-1] + ref_segment_sizes / 2

  ref_ax = ax.secondary_xaxis('top')
  ref_ax.tick_params(axis='x', direction='inout')

  # ref alignment labels use minor ticks
  ref_ax.set_xticks(xticks, minor=True)
  ref_ax.set_xticklabels(ref_labels, rotation=90, fontsize=fontsize_ticks, minor=True)
  # segment boundaries are indicated by major ticks
  ref_ax.set_xticks(vertical_lines, minor=False)
  # ref ticks and labels are shown on top of the plot
  ref_ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True, which="minor")
  ref_ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=False, which="major")


def get_time_tick_distance(seq_len_in_ms: int):
  if seq_len_in_ms < 5_000:
    return 500
  elif seq_len_in_ms < 10_000:
    return 1_000
  else:
    return 2_000


def plot_cosine_sim_matrix(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        ms_per_frame: int,
        ref_alignment_dict: Optional[Dict] = None,
        ref_alignment_vocab: Optional[Dict] = None,
        ref_alignment_blank_idx: Optional[int] = None,
        ref_alignment_upsampling_factor: Optional[int] = None,
):
  print(f"Plotting cosine similarity matrix for {input_name}")
  assert len(x.dims) == 3

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

    if ref_alignment_dict is not None:
      ref_alignment = ref_alignment_dict[seq_tag]

      seq_len_b = len(ref_alignment)
      # upsample x_norm_b to match ref alignment length
      x_norm_b = x_norm_b.repeat_interleave(ref_alignment_upsampling_factor, axis=0)
      # ref alignment may be shorter because downsampling is not exactly reversible
      x_norm_b = x_norm_b[:seq_len_b]
      ms_per_frame_ = ms_per_frame // ref_alignment_upsampling_factor
    else:
      ms_per_frame_ = ms_per_frame

    cos_sim = torch.matmul(x_norm_b, x_norm_b.transpose(0, 1))
    cos_sim = cos_sim.cpu().detach().numpy()

    fig = plt.figure()
    cosine_ax = plt.axes()
    mat = cosine_ax.matshow(cos_sim, cmap="seismic", vmin=-1, vmax=1)
    # cosine_ax.set_title(f"{input_name} Cosine similarity matrix for \n {seq_tag}", fontsize=12, pad=20)
    cosine_ax.xaxis.set_ticks_position('bottom')

    seq_len_b_ms = seq_len_b * ms_per_frame_
    time_step_size = 1 / ms_per_frame_ * get_time_tick_distance(seq_len_b_ms)
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    cosine_ax.set_xticks(time_ticks)
    cosine_ax.set_yticks(time_ticks)
    tick_labels = [(time_tick * ms_per_frame_) / 1000 for time_tick in time_ticks]
    cosine_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels])
    cosine_ax.set_yticklabels([f"{label:.1f}" for label in tick_labels])

    cosine_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    cosine_ax.set_ylabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    cosine_ax.invert_yaxis()

    if ref_alignment_dict is not None:
      set_ref_alignment_ticks(cosine_ax, ref_alignment, ref_alignment_vocab, ref_alignment_blank_idx)

    fig.tight_layout()

    cax = fig.add_axes(
      [cosine_ax.get_position().x1 + 0.01, cosine_ax.get_position().y0, 0.02, cosine_ax.get_position().height])
    plt.colorbar(mat, cax=cax)

    plot_file_path_png = os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}.png")
    plot_file_path_pdf = os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}.pdf")
    plt.savefig(plot_file_path_png)
    plt.savefig(plot_file_path_pdf)
    plt.close()


def plot_encoder_output(
        x: rf.Tensor,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        ms_per_frame: int,
        ref_alignment_dict: Optional[Dict] = None,
        ref_alignment_vocab: Optional[Dict] = None,
        ref_alignment_blank_idx: Optional[int] = None,
        ref_alignment_upsampling_factor: Optional[int] = None,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
):
  print(f"Plotting encoder output matrix for {dirname}")
  assert len(x.dims) == 3

  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])
  x = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_b = x[b, :seq_len_b].cpu().detach()

    if ref_alignment_dict is not None:
      ref_alignment = ref_alignment_dict[seq_tag]

      seq_len_b = len(ref_alignment)
      # upsample x_norm_b to match ref alignment length
      x_b = x_b.repeat_interleave(ref_alignment_upsampling_factor, axis=0)
      # ref alignment may be shorter because downsampling is not exactly reversible
      x_b = x_b[:seq_len_b]
      ms_per_frame_ = ms_per_frame // ref_alignment_upsampling_factor
    else:
      ms_per_frame_ = ms_per_frame

    fig = plt.figure()
    encoder_ax = plt.axes()
    mat = encoder_ax.matshow(x_b.numpy().T, cmap="seismic", vmin=vmin, vmax=vmax)
    # cosine_ax.set_title(f"{input_name} Cosine similarity matrix for \n {seq_tag}", fontsize=12, pad=20)
    encoder_ax.xaxis.set_ticks_position('bottom')

    seq_len_b_ms = seq_len_b * ms_per_frame_
    time_step_size = 1 / ms_per_frame_ * get_time_tick_distance(seq_len_b_ms)
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    encoder_ax.set_xticks(time_ticks)
    tick_labels = [(time_tick * ms_per_frame_) / 1000 for time_tick in time_ticks]
    encoder_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels])

    encoder_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    encoder_ax.set_ylabel("Features", fontsize=fontsize_axis, labelpad=4)
    encoder_ax.invert_yaxis()

    if ref_alignment_dict is not None:
      set_ref_alignment_ticks(encoder_ax, ref_alignment, ref_alignment_vocab, ref_alignment_blank_idx)

    # fig.tight_layout()

    cax = fig.add_axes(
      [encoder_ax.get_position().x1 + 0.01, encoder_ax.get_position().y0, 0.02, encoder_ax.get_position().height])
    plt.colorbar(mat, cax=cax)

    plot_file_path_png = os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}.png")
    plot_file_path_pdf = os.path.join(dirname, f"cos_sim_{seq_tag.replace('/', '_')}.pdf")
    plt.savefig(plot_file_path_png, bbox_inches="tight")
    plt.savefig(plot_file_path_pdf, bbox_inches="tight")
    plt.close()


def plot_framewise_prob_of_label(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        ms_per_frame: int,
        ref_alignment_dict: Optional[Dict] = None,
        ref_alignment_vocab: Optional[Dict] = None,
        ref_alignment_blank_idx: Optional[int] = None,
        ref_alignment_upsampling_factor: Optional[int] = None,
        label_idx: int = 0,
):
  print(f"Plotting framewise probs for label {label_idx} for {input_name}")
  assert len(x.dims) == 3

  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])
  x = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_b = x[b, :seq_len_b, label_idx].cpu().detach()  # [T, 1]

    if ref_alignment_dict is not None:
      ref_alignment = ref_alignment_dict[seq_tag]

      seq_len_b = len(ref_alignment)
      # upsample x_norm_b to match ref alignment length
      x_b = x_b.repeat_interleave(ref_alignment_upsampling_factor, axis=0)
      # ref alignment may be shorter because downsampling is not exactly reversible
      x_b = x_b[:seq_len_b]
      ms_per_frame_ = ms_per_frame // ref_alignment_upsampling_factor
    else:
      ms_per_frame_ = ms_per_frame

    fig = plt.figure()
    activation_ax = plt.axes()
    activation_ax.plot(x_b)
    # cosine_ax.set_title(f"{input_name} Cosine similarity matrix for \n {seq_tag}", fontsize=12, pad=20)
    activation_ax.xaxis.set_ticks_position('bottom')

    seq_len_b_ms = seq_len_b * ms_per_frame_
    time_step_size = 1 / ms_per_frame_ * get_time_tick_distance(seq_len_b_ms)
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    activation_ax.set_xticks(time_ticks)
    tick_labels = [(time_tick * ms_per_frame_) / 1000 for time_tick in time_ticks]
    activation_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels])

    activation_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    activation_ax.set_aspect("auto")

    if ref_alignment_dict is not None:
      set_ref_alignment_ticks(activation_ax, ref_alignment, ref_alignment_vocab, ref_alignment_blank_idx)

    # fig.tight_layout()

    # cax = fig.add_axes(
    #   [activation_ax.get_position().x1 + 0.01, activation_ax.get_position().y0, 0.02, activation_ax.get_position().height])
    # plt.colorbar(mat, cax=cax)

    plot_file_path_png = os.path.join(dirname, f"ctc_act_{seq_tag.replace('/', '_')}.png")
    plot_file_path_pdf = os.path.join(dirname, f"ctc_act_{seq_tag.replace('/', '_')}.pdf")
    plt.savefig(plot_file_path_png, bbox_inches="tight")
    plt.savefig(plot_file_path_pdf, bbox_inches="tight")
    plt.close()


def get_argmax_of_label_distribution(
        prob_distribution: rf.Tensor,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
):
  """

  :param prob_distribution: tensor of shape [B, T, V]
  :param seq_tags:
  :param batch_dim:
  :param spatial_dim:
  :return:
  """
  assert len(prob_distribution.dims) == 3

  feature_dim = prob_distribution.remaining_dims([batch_dim, spatial_dim])[0]
  prob_distribution = prob_distribution.copy_transpose([batch_dim, spatial_dim, feature_dim])
  prob_distribution = prob_distribution.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  argmaxes = {}

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_b = prob_distribution[b, :seq_len_b].cpu().detach()  # [T, V]

    argmaxes[seq_tag] = torch.argmax(x_b, dim=1).numpy()  # [T]

  return argmaxes


def plot_framewise_prob_of_blank_and_argmax(
        x: rf.Tensor,
        input_name: str,
        dirname: str,
        seq_tags: rf.Tensor,
        batch_dim: rf.Dim,
        spatial_dim: rf.Dim,
        ms_per_frame: int,
        ref_alignment_dict: Optional[Dict] = None,
        ref_alignment_vocab: Optional[Dict] = None,
        ref_alignment_blank_idx: Optional[int] = None,
        ref_alignment_upsampling_factor: Optional[int] = None,
        blank_sil_idx: int = 0,
):
  print(f"Plotting framewise probs for blank/silence and argmax for {input_name}")
  assert len(x.dims) == 3

  if os.path.exists(dirname):
    return
  else:
    os.makedirs(dirname)

  feature_dim = x.remaining_dims([batch_dim, spatial_dim])[0]

  x = x.copy_transpose([batch_dim, spatial_dim, feature_dim])
  x = x.raw_tensor

  B = batch_dim.dyn_size_ext.raw_tensor.item()  # noqa

  for b in range(B):
    seq_tag = seq_tags.raw_tensor[b].item()
    seq_len_b = spatial_dim.dyn_size_ext.raw_tensor[b].item()
    x_b = x[b, :seq_len_b].cpu().detach()  # [T, 1]

    if ref_alignment_dict is not None:
      ref_alignment = ref_alignment_dict[seq_tag]

      seq_len_b = len(ref_alignment)
      # upsample x_norm_b to match ref alignment length
      x_b = x_b.repeat_interleave(ref_alignment_upsampling_factor, axis=0)
      # ref alignment may be shorter because downsampling is not exactly reversible
      x_b = x_b[:seq_len_b]
      ms_per_frame_ = ms_per_frame // ref_alignment_upsampling_factor
    else:
      ms_per_frame_ = ms_per_frame

    fig = plt.figure()
    activation_ax = plt.axes()
    linewidth = 0.5
    # plot values for blank/silence label in black
    activation_ax.plot(x_b[:, blank_sil_idx], label="Blank/Silence", color="darkgray", linestyle="--", linewidth=linewidth)

    # plot values for the argmax except for the blank/silence label in red
    xb_non_blank = torch.cat([x_b[:, :blank_sil_idx], x_b[:, blank_sil_idx + 1:]], dim=1)
    xb_non_blank_max = torch.max(xb_non_blank, dim=1).values
    activation_ax.plot(xb_non_blank_max, label="Other labels", color="red", linewidth=linewidth)
    # put legend on the right side outside the plot
    activation_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend()

    # cosine_ax.set_title(f"{input_name} Cosine similarity matrix for \n {seq_tag}", fontsize=12, pad=20)
    activation_ax.xaxis.set_ticks_position('bottom')

    seq_len_b_ms = seq_len_b * ms_per_frame_
    time_step_size = 1 / ms_per_frame_ * get_time_tick_distance(seq_len_b_ms)
    time_ticks = np.arange(0, seq_len_b, time_step_size)
    activation_ax.set_xticks(time_ticks)
    tick_labels = [(time_tick * ms_per_frame_) / 1000 for time_tick in time_ticks]
    activation_ax.set_xticklabels([f"{label:.1f}" for label in tick_labels])

    activation_ax.set_xlabel("Time (s)", fontsize=fontsize_axis, labelpad=4)
    activation_ax.set_aspect("auto")

    if ref_alignment_dict is not None:
      set_ref_alignment_ticks(activation_ax, ref_alignment, ref_alignment_vocab, ref_alignment_blank_idx)

    # fig.tight_layout()

    # cax = fig.add_axes(
    #   [activation_ax.get_position().x1 + 0.01, activation_ax.get_position().y0, 0.02, activation_ax.get_position().height])
    # plt.colorbar(mat, cax=cax)

    plot_file_path_png = os.path.join(dirname, f"ctc_act_{seq_tag.replace('/', '_')}.png")
    plot_file_path_pdf = os.path.join(dirname, f"ctc_act_{seq_tag.replace('/', '_')}.pdf")
    plt.savefig(plot_file_path_png, bbox_inches="tight")
    plt.savefig(plot_file_path_pdf, bbox_inches="tight")
    plt.close()


silence_indices_cache = {}
def get_silence_indices_from_gmm_alignment(
        seq_tag: str,
        gmm_alignment: np.ndarray,
        blank_idx: int,
        silence_idx: int = 1,
):
  if seq_tag in silence_indices_cache:
    return silence_indices_cache[seq_tag]

  nonzero_indices = np.where(gmm_alignment != blank_idx)[0]
  nonzero_labels = gmm_alignment[nonzero_indices]

  result = np.full_like(gmm_alignment, blank_idx)
  for label, index in zip(nonzero_labels[::-1], nonzero_indices[::-1]):
    result[:index + 1] = label

  silence_indices = np.where(result == silence_idx)[0]
  silence_indices_cache[seq_tag] = silence_indices
  return silence_indices


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


def analyze_encoder(
        model: Union[TinaConformerModel, FramewiseProbModel, DiphoneFHModel, MonophoneFHModel],
        data: rf.Tensor,
        data_spatial_dim: rf.Dim,
        targets: rf.Tensor,
        targets_spatial_dim: rf.Dim,
        seq_tags: rf.Tensor,
):
  batch_dims = data.remaining_dims([data_spatial_dim, data.feature_dim])

  config = get_global_config()

  # set all params trainable, i.e. require gradients
  for name, param in model.named_parameters():
    param.trainable = True
  rf.set_requires_gradient(data)

  enc_numel = 0
  for name, param in model.encoder.named_parameters():
    enc_numel += param.raw_tensor.numel()

  print("NUMBER OF ENCODER PARAMETERS: ", enc_numel)

  ref_alignment_hdf = Path(config.typed_value("ref_alignment_hdf", str))
  ref_alignment_blank_idx = config.typed_value("ref_alignment_blank_idx", int)
  ref_alignment_vocab_path = Path(config.typed_value("ref_alignment_vocab_path", str))
  json_vocab_path = Path(config.typed_value("json_vocab_path", str))
  with open(ref_alignment_vocab_path, "r") as f:
    ref_alignment_vocab = ast.literal_eval(f.read())
    ref_alignment_vocab = {v: k for k, v in ref_alignment_vocab.items()}

  ref_alignment_dict = hdf.load_hdf_data(ref_alignment_hdf)

  blank_or_sil_idx = config.int("blank_or_sil_idx", 0)

  with contextlib.nullcontext():  #  torch.enable_grad():
    # ------------------- run encoder and capture encoder input -----------------
    enc_layer_cls = type(model.encoder.layers[0])
    self_att_cls = type(model.encoder.layers[0].self_att)
    set_trace_variables(
      funcs_to_trace_list=[
        model.encode,
        type(model.encoder).__call__,
        enc_layer_cls.__call__,
        self_att_cls.__call__,
        self_att_cls.attention,
        dot_attention,
        rf.Sequential.__call__,
      ]
    )

    # ----------------------------------- run encoder ---------------------------------------
    sys.settrace(_trace_func)

    collected_outputs = {}
    enc_args, enc_spatial_dim = model.encode(
      data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    sys.settrace(None)

    # get downsampling factor of encoder
    enc_input_layer = model.encoder.input_layer
    input_conv_layers = enc_input_layer.conv_layers
    strides = [layer.strides for layer in input_conv_layers]
    # product over strides in the time dimension
    enc_downsampling_factor = np.prod(list(map(lambda x: x[0], strides)))

    enc = enc_args["enc"].copy_transpose([batch_dims[0], enc_spatial_dim, enc_args["enc"].feature_dim])

    plot_cosine_sim_matrix(
      x=enc,
      input_name="encoder",
      dirname="encoder_cosine_sim",
      seq_tags=seq_tags,
      batch_dim=batch_dims[0],
      spatial_dim=enc_spatial_dim,
      ref_alignment_dict=ref_alignment_dict,
      ref_alignment_vocab=ref_alignment_vocab,
      ref_alignment_upsampling_factor=enc_downsampling_factor,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      ms_per_frame=enc_downsampling_factor * 10,
    )

    if isinstance(model, DiphoneFHModel):
      for i in range(20):
        left_context = rf.range_over_dim(model.left_target_dim)
        left_context = rf.gather(left_context, indices=i, axis=left_context.dims[0])
        center_probs = model.get_center_probs(enc, left_context)
        plot_encoder_output(
          x=center_probs,
          dirname=f"center_framewise_probs_left_context_{i}_matshow",
          seq_tags=seq_tags,
          batch_dim=batch_dims[0],
          spatial_dim=enc_spatial_dim,
          ref_alignment_dict=ref_alignment_dict,
          ref_alignment_vocab=ref_alignment_vocab,
          ref_alignment_upsampling_factor=enc_downsampling_factor,
          ref_alignment_blank_idx=ref_alignment_blank_idx,
          ms_per_frame=enc_downsampling_factor * 10,
          vmin=0,
          vmax=1,
        )
    #   plot_framewise_prob_of_blank_and_argmax(
    #     x=left_framewise_probs,
    #     input_name="left_framewise_blank_and_argmax_probs",
    #     dirname="left_framewise_blank_and_argmax_probs",
    #     seq_tags=seq_tags,
    #     batch_dim=batch_dims[0],
    #     spatial_dim=enc_spatial_dim,
    #     ref_alignment_dict=ref_alignment_dict,
    #     ref_alignment_vocab=ref_alignment_vocab,
    #     ref_alignment_upsampling_factor=enc_downsampling_factor,
    #     ref_alignment_blank_idx=ref_alignment_blank_idx,
    #     ms_per_frame=enc_downsampling_factor * 10,
    #     blank_sil_idx=blank_or_sil_idx,
    #   )
    #   plot_cosine_sim_matrix(
    #     x=model._get_left_enc_rep(enc),
    #     input_name="left_encoder_rep",
    #     dirname="left_encoder_rep_cosine_sim",
    #     seq_tags=seq_tags,
    #     batch_dim=batch_dims[0],
    #     spatial_dim=enc_spatial_dim,
    #     ref_alignment_dict=ref_alignment_dict,
    #     ref_alignment_vocab=ref_alignment_vocab,
    #     ref_alignment_upsampling_factor=enc_downsampling_factor,
    #     ref_alignment_blank_idx=ref_alignment_blank_idx,
    #     ms_per_frame=enc_downsampling_factor * 10,
    #   )

    if isinstance(model, FramewiseProbModel) or isinstance(model, MonophoneFHModel):
      if isinstance(model, MonophoneFHModel):
        framewise_probs = model.get_center_probs(enc)
      else:
        framewise_probs = model(enc)

      plot_framewise_prob_of_label(
        x=framewise_probs,
        input_name="framewise_probs",
        dirname="framewise_blank_probs",
        seq_tags=seq_tags,
        batch_dim=batch_dims[0],
        spatial_dim=enc_spatial_dim,
        ref_alignment_dict=ref_alignment_dict,
        ref_alignment_vocab=ref_alignment_vocab,
        ref_alignment_upsampling_factor=enc_downsampling_factor,
        ref_alignment_blank_idx=ref_alignment_blank_idx,
        ms_per_frame=enc_downsampling_factor * 10,
        label_idx=blank_or_sil_idx,
      )

      plot_framewise_prob_of_blank_and_argmax(
        x=framewise_probs,
        input_name="framewise_blank_and_argmax_probs",
        dirname="framewise_blank_and_argmax_probs",
        seq_tags=seq_tags,
        batch_dim=batch_dims[0],
        spatial_dim=enc_spatial_dim,
        ref_alignment_dict=ref_alignment_dict,
        ref_alignment_vocab=ref_alignment_vocab,
        ref_alignment_upsampling_factor=enc_downsampling_factor,
        ref_alignment_blank_idx=ref_alignment_blank_idx,
        ms_per_frame=enc_downsampling_factor * 10,
        blank_sil_idx=blank_or_sil_idx,
      )

      plot_encoder_output(
        x=framewise_probs,
        dirname="framewise_probs_matshow",
        seq_tags=seq_tags,
        batch_dim=batch_dims[0],
        spatial_dim=enc_spatial_dim,
        ref_alignment_dict=ref_alignment_dict,
        ref_alignment_vocab=ref_alignment_vocab,
        ref_alignment_upsampling_factor=enc_downsampling_factor,
        ref_alignment_blank_idx=ref_alignment_blank_idx,
        ms_per_frame=enc_downsampling_factor * 10,
        vmin=0,
        vmax=1,
      )

      prob_argmax = get_argmax_of_label_distribution(
        prob_distribution=framewise_probs,
        seq_tags=seq_tags,
        batch_dim=batch_dims[0],
        spatial_dim=enc_spatial_dim,
      )
    else:
      prob_argmax = None

    plot_encoder_output(
      x=enc,
      dirname="encoder_output_matshow",
      seq_tags=seq_tags,
      batch_dim=batch_dims[0],
      spatial_dim=enc_spatial_dim,
      ref_alignment_dict=ref_alignment_dict,
      ref_alignment_vocab=ref_alignment_vocab,
      ref_alignment_upsampling_factor=enc_downsampling_factor,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      ms_per_frame=enc_downsampling_factor * 10,
    )

    plot_pca_components(
      x=enc,
      input_name="encoder",
      dirname="encoder_pca",
      seq_tags=seq_tags,
      batch_dim=batch_dims[0],
      spatial_dim=enc_spatial_dim,
      ref_alignment_dict=ref_alignment_dict,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      upsample_factor=enc_downsampling_factor,
      # prob_argmax=prob_argmax,
      # blank_sil_idx=blank_or_sil_idx,
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
  analyze_encoder_def = config.typed_value("_analyze_encoder_def")

  analyze_encoder_def(
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
