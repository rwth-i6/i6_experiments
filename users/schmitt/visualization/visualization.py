import copy

from sisyphus import *

from i6_core.util import MultiOutputPath

import subprocess
import tempfile
import shutil
import os
import json
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import ast
import numpy as np
import h5py
from typing import Optional, Any, Dict, Tuple, List, Union
import scipy

import i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)

from i6_experiments.users.schmitt import hdf


class PlotAttentionWeightsJob(Job):
  def __init__(self, data_path, blank_idx, json_vocab_path, time_red, seq_tag, mem_rqmt=2):
    self.data_path = data_path
    self.blank_idx = blank_idx
    self.time_red = time_red
    self.seq_tag = seq_tag
    self.mem_rqmt = mem_rqmt
    self.json_vocab_path = json_vocab_path
    self.out_plot = self.output_path("out_plot.png")
    self.out_plot_pdf = self.output_path("out_plot.pdf")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": 1, "gpu": 0},
      mini_task=True if self.mem_rqmt <= 2 else False)

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("mem_rqmt")
    return super().hash(kwargs)

  def run(self):
    # seq_tag = "switchboard-1/sw02180A/sw2180A-ms98-a-0002;switchboard-1/sw02180A/sw2180A-ms98-a-0004;switchboard-1/sw02180A/sw2180A-ms98-a-0005;switchboard-1/sw02180A/sw2180A-ms98-a-0006;switchboard-1/sw02180A/sw2180A-ms98-a-0007;switchboard-1/sw02180A/sw2180A-ms98-a-0009;switchboard-1/sw02180A/sw2180A-ms98-a-0011;switchboard-1/sw02180A/sw2180A-ms98-a-0013;switchboard-1/sw02180A/sw2180A-ms98-a-0014;switchboard-1/sw02180A/sw2180A-ms98-a-0015;switchboard-1/sw02180A/sw2180A-ms98-a-0016;switchboard-1/sw02180A/sw2180A-ms98-a-0017;switchboard-1/sw02180A/sw2180A-ms98-a-0018;switchboard-1/sw02180A/sw2180A-ms98-a-0019;switchboard-1/sw02180A/sw2180A-ms98-a-0020;switchboard-1/sw02180A/sw2180A-ms98-a-0021;switchboard-1/sw02180A/sw2180A-ms98-a-0022;switchboard-1/sw02180A/sw2180A-ms98-a-0023;switchboard-1/sw02180A/sw2180A-ms98-a-0024;switchboard-1/sw02180A/sw2180A-ms98-a-0025"
    seq_tags = self.seq_tag.split(";")

    # load hmm alignment with the rasr archiver and parse the console output
    # hmm_align = subprocess.check_output(
    #   ["/u/schmitt/experiments/transducer/config/sprint-executables/archiver", "--mode", "show", "--type", "align",
    #    "--allophone-file", "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/allophones",
    #    "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/tuske__2016_01_28__align.combined.train",
    #    self.seq_tag])
    hmm_aligns = [subprocess.check_output(
      ["/u/schmitt/experiments/transducer/config/sprint-executables/archiver", "--mode", "show", "--type", "align",
       "--allophone-file", "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/allophones",
       "/u/zeyer/setups/switchboard/2016-12-27--tf-crnn/dependencies/tuske__2016_01_28__align.combined.train", tag]) for
      tag in seq_tags]
    # hmm_align = hmm_align.splitlines()
    # hmm_align = [row.decode("utf-8").strip() for row in hmm_align]
    # hmm_align = [row for row in hmm_align if row.startswith("time")]
    # hmm_align_states = [row.split("\t")[-1] for row in hmm_align]
    # print(hmm_align_states)
    # hmm_align = [row.split("\t")[5].split("{")[0] for row in hmm_align]
    # print(hmm_align)
    # hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]
    hmm_aligns = [hmm_align.splitlines() for hmm_align in hmm_aligns]
    hmm_aligns = [[row.decode("utf-8").strip() for row in hmm_align] for hmm_align in hmm_aligns]
    hmm_aligns = [[row for row in hmm_align if row.startswith("time")] for hmm_align in hmm_aligns]
    hmm_align_states = [row.split("\t")[-1] for hmm_align in hmm_aligns for row in hmm_align]
    hmm_align = [row.split("\t")[5].split("{")[0] for hmm_align in hmm_aligns for row in hmm_align]
    hmm_align = [phon if phon != "[SILENCE]" else "[S]" for phon in hmm_align]

    hmm_align_borders = [i + 1 for i, x in enumerate(hmm_align) if
                         i < len(hmm_align) - 1 and (hmm_align[i] != hmm_align[i + 1] or hmm_align_states[i] > hmm_align_states[i+1])]
    if hmm_align_borders[-1] != len(hmm_align):
      hmm_align_borders += [len(hmm_align)]
    hmm_align_major = [hmm_align[i - 1] for i in range(1, len(hmm_align) + 1) if i in hmm_align_borders]
    if hmm_align_borders[0] != 0:
      hmm_align_borders = [0] + hmm_align_borders
    hmm_align_borders_center = [(i + j) / 2 for i, j in zip(hmm_align_borders[:-1], hmm_align_borders[1:])]

    with open(self.json_vocab_path.get_path(), "r") as f:
      label_to_idx = ast.literal_eval(f.read())
      idx_to_label = {int(v): k for k, v in label_to_idx.items()}
    data_file = np.load(self.data_path.get_path())
    weights = data_file["weights"][:, 0, :]
    align = data_file["labels"]
    seg_starts = None
    seg_lens = None
    if self.blank_idx is not None:
      seg_starts = data_file["seg_starts"]
      seg_lens = data_file["seg_lens"]
      label_idxs = align[align != self.blank_idx]
      labels = [idx_to_label[idx] for idx in label_idxs]
      # label_positions = np.where(align != self.blank_idx)[0]
      zeros = np.zeros((weights.shape[0], (int(np.ceil(len(hmm_align) / self.time_red))) - weights.shape[1]))
      weights = np.concatenate([weights, zeros], axis=1)
      weights = np.array([np.roll(row, start) for start, row in zip(seg_starts, weights)])
    else:
      labels = [idx_to_label[idx] for idx in align]

    print("WEIGHTS SHAPE: ", weights.shape)
    print("WEIGHTS: ", weights)
    print("STARTS: ", seg_starts)
    print("LENS: ", seg_lens)

    last_label_rep = len(hmm_align) % self.time_red
    weights = np.concatenate(
      [np.repeat(weights[:, :-1], self.time_red, axis=-1), np.repeat(weights[:, -1:], last_label_rep, axis=-1)], axis=-1)

    font_size = 15
    figsize_factor = 5 * font_size * len(hmm_align_major) / 110
    figsize = (figsize_factor, len(labels) / len(hmm_align_major) * figsize_factor)
    # print(figsize)
    # print("LABELS: ", labels)
    # print("Len LABELS: ", len(labels))
    # print("HMM Align: ", hmm_align_major)
    # print("Len HMM Align: ", len(hmm_align_major))
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    ax.matshow(weights, cmap=plt.cm.get_cmap("Blues"), aspect="auto")
    # time_ax = ax.twiny()

    # for every label, there is a "row" of size 1
    # we want to have a major tick at every row border, i.e. the first is at 1, then 2, then 3 etc
    yticks = [i + 1 for i in range(len(labels))]
    # we want to set the labels between these major ticks, e.g. (i+j)/2
    yticks_minor = [((i + j) / 2) for i, j in zip(([0] + yticks)[:-1], ([0] + yticks)[1:])]
    # set corresponding ticks and labels
    ax.set_yticks([tick - .5 for tick in yticks])
    ax.yaxis.set_major_formatter(ticker.NullFormatter())
    ax.set_yticks([tick - .5 for tick in yticks_minor], minor=True)
    ax.set_yticklabels(labels, fontsize=font_size, minor=True)
    # disable minor ticks and make major ticks longer
    ax.tick_params(axis="y", which="minor", length=0)
    ax.tick_params(axis="y", which="major", length=10)
    # solve the plt bug that cuts off the first and last voxel in matshow
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_ylabel("Output Labels", fontsize=font_size)
    for idx in yticks:
      ax.axhline(y=idx - .5, xmin=0, xmax=1, color="k", linewidth=.5)

    xticks = list(hmm_align_borders)[1:]
    xticks_minor = hmm_align_borders_center
    ax.set_xticks([tick - .5 for tick in xticks])
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.set_xticks([tick - .5 for tick in xticks_minor], minor=True)
    ax.set_xticklabels(hmm_align_major, minor=True, rotation=90)
    ax.tick_params(axis="x", which="minor", length=0, labelsize=font_size)
    ax.tick_params(axis="x", which="major", length=10, width=.5)
    ax.set_xlabel("HMM Alignment", fontsize=font_size, labelpad=20)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    for idx in xticks:
      ax.axvline(x=idx - .5, ymin=0, ymax=1, color="k", linestyle="--", linewidth=.5)

    num_labels = len(labels)
    if seg_starts is not None:
      for i, (start, length) in enumerate(zip(seg_starts, seg_lens)):
        ax.axvline(x=start*self.time_red - 0.5, ymin=(num_labels-i) / num_labels, ymax=(num_labels-i-1) / num_labels, color="r")
        ax.axvline(x=min((start+length) * self.time_red, xticks[-1]) - 0.5, ymin=(num_labels - i) / num_labels, ymax=(num_labels - i - 1) / num_labels, color="r")

    # time_ax = ax.twiny()
    # time_ticks = [x - .5 for x in range(0, len(align), 10)]
    # time_labels = [x for x in range(0, len(align), 10)]
    # time_ax.set_xlabel("Input Time Frames", fontsize=20)
    # time_ax.xaxis.tick_bottom()
    # time_ax.xaxis.set_label_position('bottom')
    # # time_ax.set_xlim(ax.get_xlim())
    # time_ax.set_xticks(time_ticks)
    # time_ax.set_xticklabels(time_labels, fontsize=5)

    plt.savefig(self.out_plot.get_path())
    plt.savefig(self.out_plot_pdf.get_path())


class PlotAttentionWeightsJobV2(Job):
  def __init__(
          self,
          att_weight_hdf: Union[Path, List[Path]],
          targets_hdf: Path,
          seg_starts_hdf: Optional[Path],
          seg_lens_hdf: Optional[Path],
          center_positions_hdf: Optional[Path],
          target_blank_idx: Optional[int],
          ref_alignment_blank_idx: int,
          ref_alignment_hdf: Path,
          json_vocab_path: Path,
          ctc_alignment_hdf: Optional[Path] = None,
          segment_whitelist: Optional[List[str]] = None,
          ref_alignment_json_vocab_path: Optional[Path] = None,
          plot_w_cog: bool = True,
          titles: Optional[List[str]] = None,
  ):
    assert target_blank_idx is None or (seg_lens_hdf is not None and seg_starts_hdf is not None)
    self.seg_lens_hdf = seg_lens_hdf
    self.seg_starts_hdf = seg_starts_hdf
    self.center_positions_hdf = center_positions_hdf
    self.targets_hdf = targets_hdf
    self.titles = titles

    if isinstance(att_weight_hdf, list):
      self.att_weight_hdf = att_weight_hdf[0]
      self.other_att_weights = att_weight_hdf[1:]
    else:
      self.att_weight_hdf = att_weight_hdf
      self.other_att_weights = None
    self.target_blank_idx = target_blank_idx
    self.ref_alignment_blank_idx = ref_alignment_blank_idx
    self.ref_alignment_hdf = ref_alignment_hdf
    self.json_vocab_path = json_vocab_path
    self.ctc_alignment_hdf = ctc_alignment_hdf
    self.segment_whitelist = segment_whitelist  # if segment_whitelist is not None else []

    if ref_alignment_json_vocab_path is None:
      self.ref_alignment_json_vocab_path = json_vocab_path
    else:
      self.ref_alignment_json_vocab_path = ref_alignment_json_vocab_path

    self.plot_w_color_gradient = False

    self.out_plot_dir = self.output_path("plots", True)
    self.out_plot_w_ctc_dir = self.output_path("plots_w_ctc", True)

    self.plot_w_cog = plot_w_cog
    if plot_w_cog:
      self.out_plot_w_cog_dir = self.output_path("plots_w_cog", True)
    self.out_plot_path = MultiOutputPath(self, "plots/plots.$(TASK)", self.out_plot_dir)

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def load_data(self) -> Tuple[Dict[str, np.ndarray], ...]:
    """
      Load data from the hdf files
    """
    if self.att_weight_hdf.get_path().split(".")[-1] == "hdf":
      att_weights_dict = hdf.load_hdf_data(self.att_weight_hdf, num_dims=2, segment_list=self.segment_whitelist)
      targets_dict = hdf.load_hdf_data(self.targets_hdf, segment_list=self.segment_whitelist)
    else:
      assert self.att_weight_hdf.get_path().split(".")[-1] == "npy"

      att_weights_dict = {}
      targets_dict = {}

      numpy_object = np.load(self.att_weight_hdf.get_path(), allow_pickle=True)
      numpy_object = numpy_object[()]

      for key, object_dict in numpy_object.items():
        seq_tag = object_dict["tag"]
        targets = object_dict["classes"]
        att_weights = object_dict["rec_att_weights"][:, 0, :]  # [S, T]

        att_weights_dict[seq_tag] = att_weights
        targets_dict[seq_tag] = targets

    if self.target_blank_idx is not None:
      seg_starts_dict = hdf.load_hdf_data(self.seg_starts_hdf, segment_list=self.segment_whitelist)
      seg_lens_dict = hdf.load_hdf_data(self.seg_lens_hdf, segment_list=self.segment_whitelist)
    else:
      seg_starts_dict = None
      seg_lens_dict = None
    if self.center_positions_hdf is not None:
      center_positions_dict = hdf.load_hdf_data(self.center_positions_hdf, segment_list=self.segment_whitelist)
    else:
      center_positions_dict = None
    if self.ctc_alignment_hdf is not None:
      ctc_alignment_dict = hdf.load_hdf_data(self.ctc_alignment_hdf, segment_list=self.segment_whitelist)
    else:
      ctc_alignment_dict = None
    if self.ref_alignment_hdf is not None:
      ref_alignment_dict = hdf.load_hdf_data(self.ref_alignment_hdf, segment_list=self.segment_whitelist)
      ref_alignment_dict = {k: v for k, v in ref_alignment_dict.items() if k in att_weights_dict}
    else:
      ref_alignment_dict = {}

    return (
      att_weights_dict,
      targets_dict,
      seg_starts_dict,
      seg_lens_dict,
      center_positions_dict,
      ctc_alignment_dict,
      ref_alignment_dict
    )

  @staticmethod
  def _get_fig_ax(att_weights_: List[np.ndarray], upsampling_factor: int = 1):
    """
      Initialize the figure and axis for the plot.
    """
    att_weights = att_weights_[0]

    num_labels = att_weights.shape[0]
    num_frames = att_weights.shape[1] // upsampling_factor
    # change figsize depending on number of frames and labels
    if num_frames == num_labels:
      fig_width = num_frames / 4
      fig_height = fig_width
    else:
      fig_width = 7 * num_frames / 80
      fig_height = att_weights.shape[1] * 0.1 / 7 - 1
      # if num_frames < 32:
      #   width_factor = 4
      #   height_factor = 2
      # else:
      #   width_factor = 8
      #   height_factor = 2
      # fig_width = num_frames / width_factor
      # fig_height = num_labels / height_factor

    if any([size < 2.0 for size in (fig_width, fig_height)]):
      factor = 2
    else:
      factor = 1

    figsize = (fig_width * factor, fig_height * factor)

    figsize = [figsize[0], figsize[1] * len(att_weights_)]
    # if len(att_weights_) > 1:
    # figsize[1] *= 0.6

    fig, axes = plt.subplots(
      len(att_weights_),
      figsize=figsize,
      # constrained_layout=True
    )

    if len(att_weights_) == 1:
      axes = [axes]

    return fig, axes

  @staticmethod
  def set_ticks(
          ax: plt.Axes,
          ref_alignment: Optional[np.ndarray],
          targets: np.ndarray,
          target_vocab: Dict[int, str],
          ref_vocab: Dict[int, str],
          ref_alignment_blank_idx: Optional[int],
          time_len: int,
          target_blank_idx: Optional[int] = None,
          draw_vertical_lines: bool = False,
          use_segment_center_as_label_position: bool = False,
          upsample_factor: int = 1,
          use_time_axis: bool = True,
          use_ref_axis: bool = True,
  ):
    """
      Set the ticks and labels for the x and y axis.
      x-axis: reference alignment
      y-axis: model output
    """

    if ref_alignment is not None:
      ref_label_positions = np.where(ref_alignment != ref_alignment_blank_idx)[-1] + 1
      vertical_lines = [tick - 1.0 for tick in ref_label_positions]

    if use_ref_axis:
      # positions of reference labels in the reference alignment
      # +1 bc plt starts at 1, not at 0
      ref_labels = ref_alignment[ref_alignment != ref_alignment_blank_idx]
      ref_labels = [ref_vocab[idx] for idx in ref_labels]
      # x axis
      if use_segment_center_as_label_position:
        ref_label_positions -= 1
        ref_label_positions = np.concatenate([[0], ref_label_positions])
        ref_segment_sizes = ref_label_positions[1:] - ref_label_positions[:-1]
        xticks = ref_label_positions[:-1] + ref_segment_sizes / 2
      else:
        xticks = vertical_lines
      ax.set_xticks(xticks)
      ax.set_xticklabels(ref_labels, rotation=90, fontsize=16)
      ax.tick_params(axis="x", bottom=False, top=True, labelbottom=False, labeltop=True)

      # ax.set_xlabel("Reference Alignment", fontsize=14)
      # ax.xaxis.set_label_position('top')
    else:
      ax.set_xticks([])

    if use_time_axis:
      # secondary x-axis at the bottom with time stamps
      time_step_size = 50
      time_ticks = range(0, len(ref_alignment) if ref_alignment is not None else time_len, time_step_size)

      # Create the secondary y-axis at the bottom
      time_axis = ax.secondary_xaxis('bottom')

      # Set the ticks and labels for the secondary y-axis
      time_axis.set_xticks(time_ticks)
      xtick_labels = [(time_tick * 60 / upsample_factor) / 1000 for time_tick in time_ticks]
      time_axis.set_xticklabels([f"{label:.1f}" for label in xtick_labels], fontsize=16)
      # time_axis.set_xticks([])  # no time ticks

      time_axis.set_xlabel("Time (s) ($\\rightarrow$)", fontsize=26)
      # ----

    # output labels of the model
    # in case of alignment: filter for non-blank targets. otherwise, just leave the targets array as is
    labels = targets[targets != target_blank_idx] if target_blank_idx is not None else targets
    labels = [target_vocab[idx] for idx in labels]
    # y axis
    yticks = [tick for tick in range(len(labels))]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=13)

    # horizontal lines to separate labels on y axis
    for ytick in yticks:
      ax.axhline(y=ytick - .5, xmin=0, xmax=1, color="k", linewidth=.5)

    if (draw_vertical_lines or True) and ref_alignment is not None:  # always draw
      for xtick in vertical_lines:
        if len(ref_alignment) == len(targets):
          # this is like a square grid
          x = xtick + 0.5
          linestyle = "-"
        else:
          # here, the vertical lines are at the center of the reference labels
          x = xtick
          linestyle = "--"
        ax.axvline(x=x, ymin=0, ymax=1, color="k", linewidth=.5, linestyle=linestyle, alpha=0.8)

    # axis labels
    # ax.set_ylabel("Output Labels ($\\rightarrow$)", fontsize=14)

  @staticmethod
  def _draw_segment_boundaries(
          ax: plt.Axes,
          seg_starts: np.ndarray,
          seg_lens: np.ndarray,
          att_weights: np.ndarray,
  ):
    """
      Draw red delimiters to indicate segment boundaries
    """
    num_labels = att_weights.shape[0]
    for i, (seg_start, seg_len) in enumerate(zip(seg_starts, seg_lens)):
      ymin = (num_labels - i) / num_labels
      ymax = (num_labels - i - 1) / num_labels
      ax.axvline(x=seg_start - 0.5, ymin=ymin, ymax=ymax, color="r")
      ax.axvline(x=min(seg_start + seg_len - .5, att_weights.shape[1] - .5), ymin=ymin, ymax=ymax, color="r")

  @staticmethod
  def _draw_center_positions(
          ax: plt.Axes,
          center_positions: np.ndarray,
  ):
    """
      Draw green delimiters to indicate center positions
    """
    num_labels = center_positions.shape[0]
    for i, center_position in enumerate(center_positions):
      ymin = (num_labels - i) / num_labels
      ymax = (num_labels - i - 1) / num_labels
      ax.axvline(x=center_position - .5, ymin=ymin, ymax=ymax, color="lime")
      ax.axvline(x=center_position + .5, ymin=ymin, ymax=ymax, color="lime")

  @staticmethod
  def plot_ctc_alignment(
          ax: plt.Axes,
          ctc_alignment: np.ndarray,
          num_labels: int,
          ctc_blank_idx: int,
          plot_trailing_blanks: bool = False
  ):
    label_idx = 0
    # store alignment like: 000011112222223333, where the number is the label index (~ height in the plot)
    ctc_alignment_plot_data = []
    for x, ctc_label in enumerate(ctc_alignment):
      ctc_alignment_plot_data.append(label_idx)
      if ctc_label != ctc_blank_idx:
        label_idx += 1
      # stop if we reached the last label, the rest of the ctc alignment are blanks
      if label_idx == num_labels and not plot_trailing_blanks:
        break
    ax.plot(ctc_alignment_plot_data, "o", color="black", alpha=.4)

  @staticmethod
  def plot_center_of_gravity(ax: plt.Axes, att_weights: np.ndarray):
    cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)[:, None]  # [S,1]
    ax.plot(cog, range(cog.shape[0]), "o", color="red", alpha=.4)

  def run(self):
    # load data from hdfs
    att_weights_dict, targets_dict, seg_starts_dict, seg_lens_dict, center_positions_dict, ctc_alignment_dict, ref_alignment_dict = self.load_data()

    if self.other_att_weights is not None:
      other_att_weight_dicts = [hdf.load_hdf_data(att_weight_hdf, num_dims=2, segment_list=self.segment_whitelist) for att_weight_hdf in self.other_att_weights]
    else:
      other_att_weight_dicts = []

    # load vocabulary as dictionary
    with open(self.json_vocab_path.get_path(), "r", encoding="utf-8") as f:
      json_data = f.read()
      target_vocab = ast.literal_eval(json_data)  # label -> idx
      target_vocab = {v: k for k, v in target_vocab.items()}  # idx -> label
      # if we have a target blank idx, we replace the EOS symbol in the vocab with "<b>"
      if self.target_blank_idx is not None:
        target_vocab[self.target_blank_idx] = "<b>"

    # load reference alignment vocabulary as dictionary
    if self.ref_alignment_json_vocab_path != self.json_vocab_path:
      with open(self.ref_alignment_json_vocab_path.get_path(), "r") as f:
        json_data = f.read()
        ref_vocab = ast.literal_eval(json_data)
        ref_vocab = {v: k for k, v in ref_vocab.items()}
    else:
      ref_vocab = target_vocab

    # for each seq tag, plot the corresponding att weights
    for seq_tag in att_weights_dict.keys():
      seg_starts = seg_starts_dict[seq_tag] if self.target_blank_idx is not None else None  # [S]
      seg_lens = seg_lens_dict[seq_tag] if self.target_blank_idx is not None else None  # [S]
      center_positions = center_positions_dict[seq_tag] if self.center_positions_hdf is not None else None  # [S]
      ctc_alignment = ctc_alignment_dict[seq_tag] if self.ctc_alignment_hdf is not None else None  # [T]
      ref_alignment = ref_alignment_dict.get(seq_tag)  # [T]
      targets = targets_dict[seq_tag]  # [S]
      att_weights = att_weights_dict[seq_tag]  # [S,T]

      if ref_alignment is None:
        self.ref_alignment_blank_idx = 0
        ref_alignment = np.zeros((att_weights.shape[1] * 6,), dtype=np.int32)

      att_weights_list = [att_weights]
      for other_att_weight_dict in other_att_weight_dicts:
        other_att_weights = other_att_weight_dict[seq_tag]
        att_weights_list.append(other_att_weights)

      for i in range(len(att_weights_list)):
        att_weights_ = att_weights_list[i]
        if ref_alignment is None:
          upsampling_factor = 1
        else:
          upsampling_factor = ref_alignment.shape[0] // att_weights_.shape[1]
        if upsampling_factor != 1:
          upsampling_factor = 6  # hard code for now
          # repeat each frame by upsample_factor times
          att_weights_ = np.repeat(att_weights_, upsampling_factor, axis=1)
          if att_weights_.shape[1] < ref_alignment.shape[0]:
            assert ref_alignment.shape[0] - att_weights_.shape[1] < upsampling_factor
            att_weights_ = np.concatenate([att_weights_, np.zeros((att_weights_.shape[0], ref_alignment.shape[0] - att_weights_.shape[1])) + np.min(att_weights_)], axis=1)
          # cut off the last frames if necessary
          att_weights_ = att_weights_[:, :ref_alignment.shape[0]]

          att_weights_list[i] = att_weights_

      fig, axes = self._get_fig_ax(att_weights_list, upsampling_factor=(upsampling_factor // 2) if upsampling_factor > 1 else 1)

      for i, ax in enumerate(axes):
        att_weights = att_weights_list[i]
        ax.matshow(
          att_weights,
          cmap=plt.cm.get_cmap("Greys") if self.plot_w_color_gradient else plt.cm.get_cmap("Blues"),
          aspect="auto"
        )

        if self.titles is not None:
          assert len(self.titles) == len(axes)
          ax.set_title(self.titles[i], fontsize=20)

        if self.plot_w_color_gradient:
          gradient = np.linspace(0, 1, att_weights.shape[1])
          gradient = np.repeat(gradient[None, :], att_weights.shape[0], axis=0)
          ax.imshow(gradient, aspect="auto", cmap="hsv", alpha=0.5, interpolation=None)

        # set y ticks and labels
        self.set_ticks(
          ax=ax,
          ref_alignment=ref_alignment,
          targets=targets,
          target_vocab=target_vocab,
          ref_vocab=ref_vocab,
          ref_alignment_blank_idx=self.ref_alignment_blank_idx,
          target_blank_idx=self.target_blank_idx,
          use_segment_center_as_label_position=upsampling_factor > 1,
          upsample_factor=upsampling_factor,
          use_time_axis=i == len(axes) - 1,
          use_ref_axis=i == 0 and ref_alignment is not None,
          time_len=att_weights.shape[1]
        )
        if self.target_blank_idx is not None:
          self._draw_segment_boundaries(ax, seg_starts, seg_lens, att_weights)
        if center_positions is not None:
          self._draw_center_positions(ax, center_positions)

        ax.invert_yaxis()

      fig.text(0.01, 0.5, 'Output Labels ($\\rightarrow$)', va='center', rotation='vertical', fontsize=26)

      dirname = self.out_plot_dir.get_path()
      if seq_tag.startswith("dev-other"):
        concat_num = len(seq_tag.split(";"))
        if concat_num > 1:
          filename = os.path.join(dirname, f"plot.{seq_tag.split(';')[0].replace('/', '_')}.+{concat_num - 1}")
        else:
          filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))
      else:
        filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))

      plt.savefig(filename + ".png", bbox_inches='tight')
      plt.savefig(filename + ".pdf", bbox_inches='tight')

      # plot ctc alignment if available
      if ctc_alignment is not None:
        for i, ax in enumerate(axes):
          self.plot_ctc_alignment(
            ax,
            ctc_alignment,
            num_labels=att_weights.shape[0],
            ctc_blank_idx=self.ref_alignment_blank_idx  # works for us but better to add separate attribute for ctc
          )
        filename = os.path.join(self.out_plot_w_ctc_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")

      # plot center of gravity
      if self.plot_w_cog:
        for i, ax in enumerate(axes):
          ax.lines[-1].remove()  # remove ctc alignment plot
          self.plot_center_of_gravity(ax, att_weights)
        filename = os.path.join(self.out_plot_w_cog_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
        plt.savefig(filename + ".png")
        plt.savefig(filename + ".pdf")

      plt.close()


  @classmethod
  def hash(cls, kwargs: Dict[str, Any]):
    d = copy.deepcopy(kwargs)

    if d["center_positions_hdf"] is None:
      d.pop("center_positions_hdf")
    if d["ctc_alignment_hdf"] is None:
      d.pop("ctc_alignment_hdf")
    if d["ref_alignment_json_vocab_path"] is None:
      d.pop("ref_alignment_json_vocab_path")
    if d["plot_w_cog"] is True:
      d.pop("plot_w_cog")
    if d["titles"] is None:
      d.pop("titles")

    return super().hash(d)


class PlotAveragedSelfAttentionWeightsJob(Job):
  def __init__(
          self,
          att_weight_hdf: Path,
  ):
    self.att_weight_hdf = att_weight_hdf

    self.out_plot_dir = self.output_path("plots", True)

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    att_weights_dict = hdf.load_hdf_data(self.att_weight_hdf, num_dims=2)

    att_weights_resampled_list = []
    num_resampled_frames = 70

    for i, seq_tag in enumerate(att_weights_dict):
      att_weights = att_weights_dict[seq_tag]
      att_weights_resampled = scipy.signal.resample(att_weights, num_resampled_frames, axis=0)
      att_weights_resampled = scipy.signal.resample(att_weights_resampled, num_resampled_frames, axis=1)
      att_weights_resampled_list.append(att_weights_resampled)

    att_weights_resampled = np.stack(att_weights_resampled_list, axis=0)
    num_stacked = att_weights_resampled.shape[0]
    att_weights_resampled = np.mean(att_weights_resampled, axis=0)

    plt.matshow(att_weights_resampled, cmap=plt.cm.get_cmap("Blues"), aspect="auto")
    plt.gca().invert_yaxis()
    plt.savefig("att_weights_resampled.png")
    plt.close()

    print("Stacked: ", num_stacked)


class PlotSelfAttentionWeightsOverEpochsJob(Job):
  def __init__(
          self,
          att_weight_hdfs: List[Path],
          epochs: List[int],
  ):
    self.att_weight_hdfs = att_weight_hdfs
    self.epochs = epochs

    self.out_plot_dir = self.output_path("plots", True)

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    att_weights_dicts = [hdf.load_hdf_data(att_weight_hdf, num_dims=2) for att_weight_hdf in self.att_weight_hdfs]
    num_plots = len(att_weights_dicts)

    if num_plots % 2 == 0:
      num_rows = 2
      num_cols = num_plots // 2
    else:
      num_rows = 1
      num_cols = num_plots

    title_fontsize = 38
    ticklabel_fontsize = 28
    axlabel_fontsize = 34

    for seq_tag in att_weights_dicts[0]:
      # fig, axes = plt.subplots(
      #   nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))
      fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 5 * num_rows))

      for i, att_weights_dict in enumerate(att_weights_dicts):
        row = i // num_cols
        col = i % num_cols

        att_weights = att_weights_dict[seq_tag]  # [query, key]
        epoch = self.epochs[i]
        time_len = att_weights.shape[0]

        ax = axes[row, col]
        ax.matshow(att_weights, cmap=plt.cm.get_cmap("Blues"), aspect="auto")

        ax.set_title(f"Epoch {epoch * 4 / 20}", fontsize=title_fontsize, pad=7)

        time_step_size = 1 / 60 * 1000
        time_ticks = np.arange(0, time_len, time_step_size)
        tick_labels = [(time_tick * 60) / 1000 for time_tick in time_ticks]

        if row == 0:
          ax.set_xticks([])
        else:
          ax.set_xticks(time_ticks)
          ax.set_xticklabels([f"{label:.1f}" for label in tick_labels], fontsize=ticklabel_fontsize)
          ax.xaxis.set_ticks_position('bottom')
          # ax.set_xlabel("Keys/Values time (s)", fontsize=axlabel_fontsize, labelpad=6)

        if col == 0:
          ax.set_yticks(time_ticks)
          ax.set_yticklabels([f"{label:.1f}" for label in tick_labels], fontsize=ticklabel_fontsize)
          # ax.set_ylabel("Queries time (s)", fontsize=axlabel_fontsize, labelpad=6)
        else:
          ax.set_yticks([])

        ax.invert_yaxis()

      fig.text(0.5, 0.01, 'Keys/Values time (s)', ha='center', fontsize=axlabel_fontsize)
      fig.text(0.06, 0.5, 'Queries time (s)', va='center', rotation='vertical', fontsize=axlabel_fontsize)
      # fig.tight_layout()

      plt.savefig(os.path.join(self.out_plot_dir.get_path(), f"plot.{seq_tag.replace('/', '_')}.png"), bbox_inches='tight')
      plt.savefig(os.path.join(self.out_plot_dir.get_path(), f"plot.{seq_tag.replace('/', '_')}.pdf"), bbox_inches='tight')
      plt.close()


class PlotCtcProbsJob(Job):
  def __init__(
          self,
          ctc_probs_hdf: Path,
          targets_hdf: Path,
          ctc_blank_idx: int,
          ctc_alignment_hdf: Path,
          json_vocab_path: Path,
          segment_whitelist: Optional[List[str]] = None,
  ):
    self.targets_hdf = targets_hdf
    self.ctc_probs_hdf = ctc_probs_hdf
    self.ctc_blank_idx = ctc_blank_idx
    self.ctc_alignment_hdf = ctc_alignment_hdf
    self.json_vocab_path = json_vocab_path
    self.segment_whitelist = segment_whitelist if segment_whitelist is not None else []

    self.out_plot_dir = self.output_path("plots", True)
    self.out_plot_w_ctc_dir = self.output_path("plots_w_ctc", True)
    self.out_plot_w_cog_dir = self.output_path("plots_w_cog", True)

  def tasks(self):
    yield Task("run", mini_task=True)

  @staticmethod
  def _get_fig_ax(ctc_probs: np.ndarray, targets: np.ndarray):
    """
      Initialize the figure and axis for the plot.
    """
    num_labels = targets.shape[0]
    num_frames = ctc_probs.shape[1]
    fig_width = num_frames / 8
    fig_height = num_labels / 4
    figsize = (fig_width, fig_height)
    return plt.subplots(figsize=figsize, constrained_layout=True)

  def load_data(self) -> Tuple[Dict[str, np.ndarray], ...]:
    """
      Load data from the hdf files
    """
    ctc_probs_dict = hdf.load_hdf_data(self.ctc_probs_hdf, segment_list=self.segment_whitelist)
    targets_dict = hdf.load_hdf_data(self.targets_hdf, segment_list=self.segment_whitelist)
    ctc_alignment_dict = hdf.load_hdf_data(self.ctc_alignment_hdf, segment_list=self.segment_whitelist)

    return (
      ctc_probs_dict,
      targets_dict,
      ctc_alignment_dict,
    )

  def run(self):
    # load data from hdfs
    ctc_probs_dict, targets_dict, ctc_alignment_dict = self.load_data()

    # load vocabulary as dictionary
    with open(self.json_vocab_path.get_path(), "r") as f:
      json_data = f.read()
      vocab = ast.literal_eval(json_data)  # label -> idx
      vocab = {v: k for k, v in vocab.items()}  # idx -> label

    # for each seq tag, plot the corresponding att weights
    for seq_tag in ctc_probs_dict.keys():
      ctc_alignment = ctc_alignment_dict[seq_tag] if self.ctc_alignment_hdf is not None else None  # [T]
      targets = targets_dict[seq_tag]  # [S]
      ctc_probs = ctc_probs_dict[seq_tag]  # [T, V]
      ctc_probs = ctc_probs[:, targets]  # [T, S]
      ctc_probs = np.transpose(ctc_probs)  # [S, T]
      # normalize ctc probs using softmax
      ctc_probs = np.exp(ctc_probs) / np.sum(np.exp(ctc_probs), axis=1)[:, None]

      fig, ax = self._get_fig_ax(ctc_probs, targets)
      ax.matshow(
        ctc_probs,
        cmap=plt.cm.get_cmap("Blues"),
        aspect="auto"
      )

      # set y ticks and labels
      PlotAttentionWeightsJobV2.set_ticks(
        ax,
        ctc_alignment,
        targets,
        vocab,
        vocab,
        self.ctc_blank_idx,
        ctc_probs.shape[1],
        None
      )

      dirname = self.out_plot_dir.get_path()
      filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))
      plt.savefig(filename + ".png")
      plt.savefig(filename + ".pdf")

      # plot ctc alignment
      PlotAttentionWeightsJobV2.plot_ctc_alignment(
        ax,
        ctc_alignment,
        num_labels=targets.shape[0],
        ctc_blank_idx=self.ctc_blank_idx
      )
      filename = os.path.join(self.out_plot_w_ctc_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
      plt.savefig(filename + ".png")
      plt.savefig(filename + ".pdf")

      # plot center of gravity
      ax.lines[-1].remove()  # remove ctc alignment plot
      PlotAttentionWeightsJobV2.plot_center_of_gravity(ax, ctc_probs)
      filename = os.path.join(self.out_plot_w_cog_dir.get_path(), "plot.%s" % seq_tag.replace("/", "_"))
      plt.savefig(filename + ".png")
      plt.savefig(filename + ".pdf")


class PlotAlignmentJob(Job):
  def __init__(
          self,
          alignment_hdf: Path,
          json_vocab_path: Path,
          target_blank_idx: int,
          segment_list: List[str],
          ref_alignment_hdf: Optional[Path] = None,
          ref_alignment_blank_idx: Optional[int] = None,
          silence_idx: Optional[int] = None,
          ref_alignment_json_vocab_path: Optional[Path] = None,
  ):
    if ref_alignment_hdf is not None:
      assert ref_alignment_blank_idx is not None

    self.alignment_hdf = alignment_hdf
    self.json_vocab_path = json_vocab_path
    self.target_blank_idx = target_blank_idx
    self.segment_list = segment_list
    self.silence_idx = silence_idx
    self.ref_alignment_hdf = ref_alignment_hdf if ref_alignment_hdf is not None else alignment_hdf
    self.ref_alignment_blank_idx = ref_alignment_blank_idx if ref_alignment_blank_idx is not None else target_blank_idx

    if ref_alignment_json_vocab_path is None:
      self.ref_alignment_json_vocab_path = json_vocab_path
    else:
      self.ref_alignment_json_vocab_path = ref_alignment_json_vocab_path

    self.out_plot_dir = self.output_path("plots", True)

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  @staticmethod
  def _get_fig_ax(alignment: np.ndarray):
    """
      Initialize the figure and axis for the plot.
    """
    num_frames = len(alignment)
    fig_width = num_frames / 12
    fig_height = 4.0
    figsize = (fig_width, fig_height)
    return plt.subplots(figsize=figsize, constrained_layout=True)

  @staticmethod
  def _set_ticks(
          ax: plt.Axes,
          alignment: np.ndarray,
          labels: np.ndarray,
          vocab: Dict[int, str],
          blank_idx: int,
          ymin=1.0,
          ymax=0.0,
          color="r"
  ):
    """
      Set the ticks and labels for the x and y axis.
      x-axis: reference alignment
      y-axis: model output
    """
    # positions of reference labels in the reference alignment
    # +1 bc plt starts at 1, not at 0
    label_positions = np.where(alignment != blank_idx)[-1] + 1
    labels = [vocab[idx] for idx in labels]
    # x axis
    ax.set_xticks([tick - 1.0 for tick in label_positions])
    ax.set_xticklabels(labels, rotation=90)

    for i, position in enumerate(label_positions):
      ax.axvline(x=position - 1.0, ymin=ymin, ymax=ymax, color=color)

  @staticmethod
  def _set_ticks_alt(
          ax: plt.Axes,
          alignment: np.ndarray,
          ref_alignment: np.ndarray,
          target_vocab: Dict[int, str],
          ref_vocab: Dict[int, str],
          blank_idx: int,
          ref_alignment_blank_idx: int,
  ):
    """
      Set the ticks and labels for the x and y axis.
      x-axis: reference alignment
      y-axis: model output
    """
    PlotAttentionWeightsJobV2.set_ticks(
      ax,
      ref_alignment=ref_alignment,
      targets=alignment,
      target_vocab=target_vocab,
      ref_vocab=ref_vocab,
      ref_alignment_blank_idx=ref_alignment_blank_idx,
      target_blank_idx=blank_idx,
      draw_vertical_lines=True,
      time_len=alignment.shape[1],
    )

    # draw last horizontal line for trailing blanks
    labels = alignment[alignment != blank_idx]
    ax.axhline(y=len(labels) - .5, xmin=0, xmax=1, color="k", linewidth=.5)

  def run(self):
    # load data from hdf
    alignment_dict = hdf.load_hdf_data(self.alignment_hdf, segment_list=self.segment_list)
    ref_alignment_dict = hdf.load_hdf_data(self.ref_alignment_hdf, segment_list=self.segment_list)

    # load target vocabulary as dictionary
    with open(self.json_vocab_path.get_path(), "r") as f:
      json_data = f.read()
      target_vocab = ast.literal_eval(json_data)  # label -> idx
      target_vocab = {v: k for k, v in target_vocab.items()}  # idx -> label
      if self.silence_idx is not None:
        target_vocab[self.silence_idx] = "[Silence]"

    if self.ref_alignment_json_vocab_path != self.json_vocab_path:
      with open(self.ref_alignment_json_vocab_path.get_path(), "r") as f:
        json_data = f.read()
        ref_vocab = ast.literal_eval(json_data)
        ref_vocab = {v: k for k, v in ref_vocab.items()}
    else:
      ref_vocab = target_vocab

    # for each seq tag, plot the corresponding att weights
    for seq_tag in alignment_dict.keys():
      alignment = alignment_dict[seq_tag]
      ref_alignment = ref_alignment_dict[seq_tag]
      labels = alignment[alignment != self.target_blank_idx]

      dummy_matrix = np.zeros((len(labels) + 1, len(alignment)))
      fig, ax = PlotAttentionWeightsJobV2._get_fig_ax(att_weights=dummy_matrix)
      ax.matshow(dummy_matrix, aspect="auto", alpha=0.0)
      # set y ticks and labels
      self._set_ticks_alt(
        ax,
        alignment,
        ref_alignment,
        target_vocab,
        ref_vocab,
        self.target_blank_idx,
        ref_alignment_blank_idx=self.ref_alignment_blank_idx
      )
      PlotAttentionWeightsJobV2.plot_ctc_alignment(
        ax,
        alignment,
        num_labels=labels.shape[0],
        ctc_blank_idx=self.target_blank_idx,
        plot_trailing_blanks=True
      )
      # plt.gca().invert_yaxis()

      dirname = self.out_plot_dir.get_path()
      filename = os.path.join(dirname, "plot.%s" % seq_tag.replace("/", "_"))
      plt.savefig(filename + ".png")
      plt.savefig(filename + ".pdf")

  @classmethod
  def hash(cls, kwargs):
    if kwargs["ref_alignment_hdf"] is None:
      kwargs.pop("ref_alignment_hdf")
      kwargs.pop("ref_alignment_blank_idx")

    if kwargs["ref_alignment_json_vocab_path"] is None:
      kwargs.pop("ref_alignment_json_vocab_path")

    return super().hash(kwargs)


def plot_att_weights(
        att_weight_hdf: Path,
        targets_hdf: Path,
        seg_starts_hdf: Optional[Path],
        seg_lens_hdf: Optional[Path],
        center_positions_hdf: Optional[Path],
        target_blank_idx: Optional[int],
        ref_alignment_blank_idx: int,
        ref_alignment_hdf: Path,
        json_vocab_path: Path,
        ctc_alignment_hdf: Optional[Path] = None,
        segment_whitelist: Optional[List[str]] = None,
        ref_alignment_json_vocab_path: Optional[Path] = None,
        plot_name: str = "plots",
        plot_w_color_gradient: bool = False,
):
  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=att_weight_hdf,
    targets_hdf=targets_hdf,
    seg_starts_hdf=seg_starts_hdf,
    seg_lens_hdf=seg_lens_hdf,
    center_positions_hdf=center_positions_hdf,
    target_blank_idx=target_blank_idx,
    ref_alignment_blank_idx=ref_alignment_blank_idx,
    ref_alignment_hdf=ref_alignment_hdf,
    json_vocab_path=json_vocab_path,
    ctc_alignment_hdf=ctc_alignment_hdf,
    segment_whitelist=segment_whitelist,
    ref_alignment_json_vocab_path=ref_alignment_json_vocab_path,
    plot_w_cog=False,
  )

  if plot_w_color_gradient:
    plot_att_weights_job.plot_w_color_gradient = True

  # overwrite the output paths to paths in the cwd
  plot_att_weights_job.out_plot_dir = Path(f"{plot_name}/plots", )
  if not os.path.exists(plot_att_weights_job.out_plot_dir.get_path()):
    os.makedirs(plot_att_weights_job.out_plot_dir.get_path())

  plot_att_weights_job.run()
