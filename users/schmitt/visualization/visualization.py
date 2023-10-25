import copy

from sisyphus import *

from i6_core.util import MultiOutputPath

import subprocess
import tempfile
import shutil
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import ticker
import ast
import numpy as np
import h5py
from typing import Optional, Any, Dict

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)


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
          att_weight_hdf: Path,
          targets_hdf: Path,
          seg_starts_hdf: Optional[Path],
          seg_lens_hdf: Optional[Path],
          center_positions_hdf: Optional[Path],
          target_blank_idx: Optional[int],
          ref_alignment_blank_idx: int,
          ref_alignment_hdf: Path,
          json_vocab_path: Path,
  ):
    assert target_blank_idx is None or (seg_lens_hdf is not None and seg_starts_hdf is not None)
    self.seg_lens_hdf = seg_lens_hdf
    self.seg_starts_hdf = seg_starts_hdf
    self.center_positions_hdf = center_positions_hdf
    self.targets_hdf = targets_hdf
    self.att_weight_hdf = att_weight_hdf
    self.target_blank_idx = target_blank_idx
    self.ref_alignment_blank_idx = ref_alignment_blank_idx
    self.ref_alignment_hdf = ref_alignment_hdf
    self.json_vocab_path = json_vocab_path

    self.out_plot_dir = self.output_path("plots", True)
    self.out_plot_path = MultiOutputPath(self, "plots/plots.$(TASK)", self.out_plot_dir)

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    att_weight_hdf = self.att_weight_hdf.get_path()
    targets_hdf = self.targets_hdf.get_path()
    seg_starts_hdf = self.seg_starts_hdf.get_path() if self.target_blank_idx is not None else None
    seg_lens_hdf = self.seg_lens_hdf.get_path() if self.target_blank_idx is not None else None
    center_positions_hdf = self.center_positions_hdf.get_path() if self.center_positions_hdf is not None else None
    ref_alignment_hdf = self.ref_alignment_hdf.get_path()
    json_vocab_path = self.json_vocab_path.get_path()

    # first, we load all the necessary data (att_weights, targets, segment starts/lens (if provided)) from the hdf files
    with h5py.File(att_weight_hdf, "r") as f:
      # load tags once here and reuse for remaining hdfs
      seq_tags = f["seqTags"][()]
      seq_tags = [str(tag) for tag in seq_tags]  # convert from byte to string

      seq_lens = f["seqLengths"][()][:, 0]  # total num frames (num_seqs * num_labels * num_frames)
      shape_data = f["targets"]["data"]["sizes"][
        ()]  # shape of att weights for each seq, stored alternating (num_labels, num_frames, ..., num_labels, num_frames)
      weights_data = f["inputs"][()]  # all att weights as 1D array
      weights_dict = {}
      for i, seq_len in enumerate(seq_lens):
        weights_dict[seq_tags[i]] = weights_data[:seq_len]  # store weights in dict under corresponding tag
        weights_data = weights_data[seq_len:]  # cut the stored weights from the remaining weights
        shape = shape_data[i * 2: i * 2 + 2]  # get shape for current weight matrix
        weights_dict[seq_tags[i]] = weights_dict[seq_tags[i]].reshape(shape)  # reshape corresponding entry in dict

    # follows same principle as the loading of att weights
    with h5py.File(targets_hdf, "r") as f:
      targets_data = f["inputs"][()]
      seq_lens = f["seqLengths"][()][:, 0]
      targets_dict = {}
      for i, seq_len in enumerate(seq_lens):
        targets_dict[seq_tags[i]] = targets_data[:seq_len]
        targets_data = targets_data[seq_len:]

    if self.target_blank_idx is not None:
      # follows same principle as the loading of att weights
      with h5py.File(seg_starts_hdf, "r") as f:
        seg_starts_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        seg_starts_dict = {}
        for i, seq_len in enumerate(seq_lens):
          seg_starts_dict[seq_tags[i]] = seg_starts_data[:seq_len]
          seg_starts_data = seg_starts_data[seq_len:]

      # follows same principle as the loading of att weights
      with h5py.File(seg_lens_hdf, "r") as f:
        seg_lens_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        seg_lens_dict = {}
        for i, seq_len in enumerate(seq_lens):
          seg_lens_dict[seq_tags[i]] = seg_lens_data[:seq_len]
          seg_lens_data = seg_lens_data[seq_len:]

    if center_positions_hdf is not None:
      # follows same principle as the loading of att weights
      with h5py.File(center_positions_hdf, "r") as f:
        center_positions_data = f["inputs"][()]
        seq_lens = f["seqLengths"][()][:, 0]
        center_positions_dict = {}
        for i, seq_len in enumerate(seq_lens):
          center_positions_dict[seq_tags[i]] = center_positions_data[:seq_len]
          center_positions_data = center_positions_data[seq_len:]

    # follows same principle as the loading of att weights
    with h5py.File(ref_alignment_hdf, "r") as f:
      # here we need to load the seq tags again because the ref alignment might contain more seqs than what we dumped
      # into the hdfs
      seq_tags_ref_alignment = f["seqTags"][()]
      seq_tags_ref_alignment = [str(tag) for tag in seq_tags_ref_alignment]  # convert from byte to string

      ref_alignment_data = f["inputs"][()]
      seq_lens = f["seqLengths"][()][:, 0]
      ref_alignment_dict = {}
      for i, seq_len in enumerate(seq_lens):
        # only store the ref alignment if the corresponding seq tag is among the ones for which we dumped att weights
        if seq_tags_ref_alignment[i] in seq_tags:
          ref_alignment_dict[seq_tags_ref_alignment[i]] = ref_alignment_data[:seq_len]
        ref_alignment_data = ref_alignment_data[seq_len:]

    # load vocabulary as dictionary
    with open(json_vocab_path, "r") as f:
      json_data = f.read()
      vocab = ast.literal_eval(json_data)
      vocab = {v: k for k, v in vocab.items()}

    # for each seq tag, plot the corresponding att weights
    for seq_tag in seq_tags:
      seg_starts = seg_starts_dict[seq_tag] if self.target_blank_idx is not None else None
      seg_lens = seg_lens_dict[seq_tag] if self.target_blank_idx is not None else None
      center_positions = center_positions_dict[seq_tag] if center_positions_hdf is not None else None
      ref_alignment = ref_alignment_dict[seq_tag]
      targets = targets_dict[seq_tag]
      weights = weights_dict[seq_tag]

      num_labels = weights.shape[0]
      num_frames = weights.shape[1]
      frame_label_ratio = num_frames / num_labels  # used for the fig size

      font_size = 15
      figsize_factor = 2
      fig_width = num_frames / 8
      fig_height = num_labels / 4
      figsize = (fig_width, fig_height)
      fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
      ax.matshow(weights, cmap=plt.cm.get_cmap("Blues"), aspect="auto")

      ref_label_positions = np.where(ref_alignment != self.ref_alignment_blank_idx)[-1] + 1  # +1 bc plt starts at 1, not at 0
      ref_labels = ref_alignment[ref_alignment != self.ref_alignment_blank_idx]
      ref_labels = [vocab[idx] for idx in ref_labels]  # the corresponding bpe label
      labels = targets[targets != self.target_blank_idx] if self.target_blank_idx is not None else targets  # the alignment labels which are not blank (in case of global att model, just use `targets`)
      labels = [vocab[idx] for idx in labels]  # the corresponding bpe label
      # x axis
      ax.set_xticks([tick - .5 for tick in ref_label_positions])
      ax.set_xticklabels(ref_labels, rotation=90)
      # y axis
      yticks = [tick for tick in range(num_labels)]
      ax.set_yticks(yticks)
      ax.set_yticklabels(labels)
      for ytick in yticks:
        ax.axhline(y=ytick - .5, xmin=0, xmax=1, color="k", linewidth=.5)

      if self.target_blank_idx is not None:
        # red delimiters to indicate segment boundaries
        for i, (seg_start, seg_len) in enumerate(zip(seg_starts, seg_lens)):
          ymin = (num_labels - i) / num_labels
          ymax = (num_labels - i - 1) / num_labels
          ax.axvline(x=seg_start - .5, ymin=ymin, ymax=ymax, color="r")
          ax.axvline(x=seg_start + seg_len - .5, ymin=ymin, ymax=ymax, color="r")

      if center_positions is not None:
        # yellow markers to indicate center positions
        for i, center_position in enumerate(center_positions):
          ymin = (num_labels - i) / num_labels
          ymax = (num_labels - i - 1) / num_labels
          ax.axvline(x=center_position - .5, ymin=ymin, ymax=ymax, color="lime")

      dirname = self.out_plot_dir.get_path()
      filename = os.path.join(dirname, "plot.%s.png" % seq_tag.replace("/", "_"))
      plt.savefig(filename)
      # plt.savefig(self.out_plot_pdf.get_path())


  @classmethod
  def hash(cls, kwargs: Dict[str, Any]):
    d = copy.deepcopy(kwargs)

    if d["center_positions_hdf"] is None:
      d.pop("center_positions_hdf")

    return super().hash(d)
