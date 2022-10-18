from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil
import os
import json
import matplotlib.pyplot as plt
from matplotlib import ticker
import ast
import numpy as np

from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import config as config_mod
tools_dir = os.path.join(os.path.dirname(os.path.abspath(config_mod.__file__)), "tools")


class PlotAttentionWeightsJob(Job):
  def __init__(self, data_path, blank_idx, json_vocab_path, time_red, seq_tag):
    self.data_path = data_path
    self.blank_idx = blank_idx
    self.time_red = time_red
    self.seq_tag = seq_tag
    self.json_vocab_path = json_vocab_path
    self.out_plot = self.output_path("out_plot.png")
    self.out_plot_pdf = self.output_path("out_plot.pdf")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 2, "time": 1, "gpu": 0}, mini_task=True)

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

    last_label_rep = len(hmm_align) % self.time_red
    weights = np.concatenate(
      [np.repeat(weights[:, :-1], self.time_red, axis=-1), np.repeat(weights[:, -1:], last_label_rep, axis=-1)], axis=-1)

    font_size = 15
    figsize_factor = font_size * len(hmm_align_major) / 95
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