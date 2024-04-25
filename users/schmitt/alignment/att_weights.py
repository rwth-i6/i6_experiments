from sisyphus import *
from sisyphus import Path

import os
from typing import Optional
import numpy as np
import heapq
from collections import Counter
from matplotlib import pyplot as plt

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)
from recipe.i6_experiments.users.schmitt import hdf


class CenterOfGravityStatistics:
  def __init__(self):
    self.total_error = np.array([])
    self.top_k_abs_error_seq_tags = []
    self.k = 10
    self.max_abs_err_per_seq_counter = Counter()

  def update(self, att_weights, ctc_alignment, ctc_blank_idx, seq_tag):
    cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)  # [S]
    ctc_positions = np.where(ctc_alignment != ctc_blank_idx)[0]  # [S]
    error = cog - ctc_positions
    self.total_error = np.concatenate((self.total_error, error))
    abs_error = np.abs(error)
    if len(self.top_k_abs_error_seq_tags) < self.k:
      heapq.heappush(self.top_k_abs_error_seq_tags, (np.max(abs_error), seq_tag))
    else:
      heapq.heappushpop(self.top_k_abs_error_seq_tags, (np.max(abs_error), seq_tag))

    self.max_abs_err_per_seq_counter[int(np.max(abs_error))] += 1

  def write_statistics(self, statistics_path: Path):
    total_abs_error = np.abs(self.total_error)
    with open(statistics_path.get_path(), "a") as f:
      f.write("Center of gravity (CoG) of attention weights - CTC label positions\n")
      f.write(f"\tMean absolute error: {np.mean(total_abs_error):.1f}\n")
      f.write(f"\tMean error: {np.mean(self.total_error):.1f}\n")
      f.write(f"\tMean squared error: {np.mean(total_abs_error ** 2):.1f}\n")
      f.write(f"\tMedian absolute error: {np.median(total_abs_error):.1f}\n")
      f.write(f"\tMedian error: {np.median(self.total_error):.1f}\n")
      f.write(f"\tMin absolute error: {np.min(total_abs_error):.1f}\n")
      f.write(f"\tMin error: {np.min(self.total_error):.1f}\n")
      f.write(f"\tMax absolute error: {np.max(total_abs_error):.1f}\n")
      f.write(f"\tMax error: {np.max(self.total_error):.1f}\n")
      f.write(f"\tTop {self.k} seq tags with highest absolute error:\n")
      for diff, seq_tag in sorted(self.top_k_abs_error_seq_tags, reverse=True):
        f.write(f"\t\t{seq_tag}: {diff:.1f}\n")

  def generate_plots(self):
    # generate histogram of max_abs_err_per_seq_counter
    max_abs_err = list(self.max_abs_err_per_seq_counter.keys())
    counts = list(self.max_abs_err_per_seq_counter.values())
    plt.bar(max_abs_err, counts)
    plt.title("#Sequences with Maximum Absolute Error\n between CoG and CTC label position")
    plt.xlabel("Maximum Absolute Error")
    plt.ylabel("#Sequences")
    plt.savefig("max_abs_err_per_seq.png")
    plt.close()


class AttentionWeightStatistics:
  def __init__(self):
    self.total_num_label_transitions = 0
    self.total_non_monotonic_cog_diffs = np.array([])
    self.top_k_non_monotonic_cog_seq_tags = []
    self.total_non_monotonic_argmax_diffs = np.array([])
    self.top_k_non_monotonic_argmax_seq_tags = []
    # compare seq lens for sequences with at least one non-monotonic CoG and sequences with only monotonic CoGs
    self.seq_len_statistics = {
      ">0_non_monotonic_cog": [],
      "all_monotonic_cog": []
    }
    # compare att weight entropies for labels with non-monotonic CoGs and labels with monotonic CoGs
    self.entropy_statistics = {
      "non_monotonic_cog": [],
      "monotonic_cog": []
    }
    # compare initial att weights for labels with non-monotonic CoGs and labels with monotonic CoGs
    self.initial_att_weights_statistics = {
      "non_monotonic_cog": [],
      "monotonic_cog": []
    }
    self.seqs_w_non_monotonic_cog_counter = Counter()
    self.seq_tags_w_non_monotonic_cog = {10: [], 20: []}
    self.k = 10

  def _update_cog_statistics(self, att_weights, seq_tag):
    cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)  # [S]
    self.total_num_label_transitions += len(cog) - 1
    cog_diff = np.diff(cog)
    non_monotonic_cog_mask = cog_diff < 0
    non_monotonic_cog_diffs = np.abs(cog_diff[non_monotonic_cog_mask])
    max_non_monotonic_cog_diff = np.max(non_monotonic_cog_diffs) if non_monotonic_cog_diffs.size > 0 else 0
    # maintain top-k seq_tags with highest non_monotonic_cog_diffs
    if len(self.top_k_non_monotonic_cog_seq_tags) < self.k:
      heapq.heappush(self.top_k_non_monotonic_cog_seq_tags, (max_non_monotonic_cog_diff, seq_tag))
    else:
      heapq.heappushpop(self.top_k_non_monotonic_cog_seq_tags, (max_non_monotonic_cog_diff, seq_tag))
    self.total_non_monotonic_cog_diffs = np.concatenate((self.total_non_monotonic_cog_diffs, non_monotonic_cog_diffs))

    if np.any(non_monotonic_cog_mask):
      self.seq_len_statistics[">0_non_monotonic_cog"].append(att_weights.shape[1])
    else:
      self.seq_len_statistics["all_monotonic_cog"].append(att_weights.shape[1])

    num_non_monotonic_cogs = int(np.sum(non_monotonic_cog_mask))
    self.seqs_w_non_monotonic_cog_counter[num_non_monotonic_cogs] += 1
    if num_non_monotonic_cogs in self.seq_tags_w_non_monotonic_cog:
      self.seq_tags_w_non_monotonic_cog[num_non_monotonic_cogs].append(seq_tag)

    # entropy normalized by log(T) -> [0, 1]
    entropy_norm = -np.sum(att_weights * np.log(att_weights + 1e-10), axis=1) / np.log(att_weights.shape[1])  # [S]
    self.entropy_statistics["non_monotonic_cog"] += entropy_norm[1:][non_monotonic_cog_mask].tolist()
    self.entropy_statistics["monotonic_cog"] += entropy_norm[1:][~non_monotonic_cog_mask].tolist()

    initial_att_weights_mean = np.mean(att_weights[:, :8], axis=1)  # [S]
    self.initial_att_weights_statistics["non_monotonic_cog"] += initial_att_weights_mean[1:][non_monotonic_cog_mask].tolist()
    self.initial_att_weights_statistics["monotonic_cog"] += initial_att_weights_mean[1:][~non_monotonic_cog_mask].tolist()

  def _update_argmax_statistics(self, att_weights, seq_tag):
    argmax = np.argmax(att_weights, axis=1)  # [S]
    argmax_diff = np.diff(argmax)
    non_monotonic_argmax_mask = argmax_diff < 0
    non_monotonic_argmax_diffs = np.abs(argmax_diff[non_monotonic_argmax_mask])
    max_non_monotonic_argmax_diff = np.max(non_monotonic_argmax_diffs) if non_monotonic_argmax_diffs.size > 0 else 0
    # maintain top-k seq_tags with highest non_monotonic_argmax_diffs
    if len(self.top_k_non_monotonic_argmax_seq_tags) < self.k:
      heapq.heappush(self.top_k_non_monotonic_argmax_seq_tags, (max_non_monotonic_argmax_diff, seq_tag))
    else:
      heapq.heappushpop(self.top_k_non_monotonic_argmax_seq_tags, (max_non_monotonic_argmax_diff, seq_tag))
    self.total_non_monotonic_argmax_diffs = np.concatenate((self.total_non_monotonic_argmax_diffs, non_monotonic_argmax_diffs))

  def update(self, att_weights, seq_tag):
    self._update_cog_statistics(att_weights, seq_tag)
    self._update_argmax_statistics(att_weights, seq_tag)

  def write_statistics(self, statistics_path: Path):
    # mean and median sequence lengths for non-monotonic and monotonic CoGs
    non_mono_mean_seq_len = np.mean(self.seq_len_statistics[">0_non_monotonic_cog"])
    non_mono_median_seq_len = np.median(self.seq_len_statistics[">0_non_monotonic_cog"])
    mono_mean_seq_len = np.mean(self.seq_len_statistics["all_monotonic_cog"])
    mono_median_seq_len = np.median(self.seq_len_statistics["all_monotonic_cog"])
    # mean and median entropy for non-monotonic and monotonic CoGs
    non_mono_mean_entropy = np.mean(self.entropy_statistics["non_monotonic_cog"])
    non_mono_median_entropy = np.median(self.entropy_statistics["non_monotonic_cog"])
    mono_mean_entropy = np.mean(self.entropy_statistics["monotonic_cog"])
    mono_median_entropy = np.median(self.entropy_statistics["monotonic_cog"])
    # mean and median mean initial att weights for non-monotonic and monotonic CoGs
    non_mono_mean_init_att_weights = np.mean(self.initial_att_weights_statistics["non_monotonic_cog"])
    non_mono_median_init_att_weights = np.median(self.initial_att_weights_statistics["non_monotonic_cog"])
    mono_mean_init_att_weights = np.mean(self.initial_att_weights_statistics["monotonic_cog"])
    mono_median_init_att_weights = np.median(self.initial_att_weights_statistics["monotonic_cog"])

    with open(statistics_path.get_path(), "a") as f:
      # CoG statistics
      f.write(f"\nCenter of Gravity (CoG) - Monotonicity\n")
      f.write(f"\tNon-monotonic CoGs: {len(self.total_non_monotonic_cog_diffs) / self.total_num_label_transitions * 100:.1f}%\n")
      f.write(f"\tMean non-monotonic CoG diff: {np.mean(self.total_non_monotonic_cog_diffs):.1f}\n")
      f.write(f"\tMedian non-monotonic CoG diff: {np.median(self.total_non_monotonic_cog_diffs):.1f}\n")
      f.write(f"\tMax non-monotonic CoG diff: {np.max(self.total_non_monotonic_cog_diffs):.1f}\n")
      f.write(f"\tMean/median sequence lengths:\n")
      f.write(f"\t\tAt least 1 non-monotonic CoGs: {non_mono_mean_seq_len:.1f}/{non_mono_median_seq_len:.1f}\n")
      f.write(f"\t\tOnly monotonic CoGs: {mono_mean_seq_len:.1f}/{mono_median_seq_len:.1f}\n")
      f.write(f"\tMean/median normalized attention weight entropy:\n")
      f.write(f"\t\tNon-monotonic CoGs: {non_mono_mean_entropy:.4f}/{non_mono_median_entropy:.4f}\n")
      f.write(f"\t\tMonotonic CoGs: {mono_mean_entropy:.4f}/{mono_median_entropy:.4f}\n")
      f.write(f"\tMean/median mean initial attention weights:\n")
      f.write(f"\t\tNon-monotonic CoGs: {non_mono_mean_init_att_weights:.4f}/{non_mono_median_init_att_weights:.4f}\n")
      f.write(f"\t\tMonotonic CoGs: {mono_mean_init_att_weights:.4f}/{mono_median_init_att_weights:.4f}\n")
      f.write(f"\tTop {self.k} seq tags with highest non-monotonic CoG diffs:\n")
      for diff, seq_tag in sorted(self.top_k_non_monotonic_cog_seq_tags, reverse=True):
        f.write(f"\t\t{seq_tag}: {diff:.1f}\n")
      f.write(f"\tSequences with X non-monotonic CoGs:\n")
      for num_non_monotonic_cogs in sorted(self.seq_tags_w_non_monotonic_cog.keys()):
        f.write(f"\t\t{num_non_monotonic_cogs}:\n")
        for seq_tag in self.seq_tags_w_non_monotonic_cog[num_non_monotonic_cogs][:5]:
          f.write(f"\t\t\t{seq_tag}\n")

      # argmax statistics
      f.write(f"\nArgmax - Monotonicity\n")
      f.write(f"\tNon-monotonic argmax: {len(self.total_non_monotonic_argmax_diffs) / self.total_num_label_transitions * 100:.1f}%\n")
      f.write(f"\tMean non-monotonic argmax diff: {np.mean(self.total_non_monotonic_argmax_diffs):.1f}\n")
      f.write(f"\tMedian non-monotonic argmax diff: {np.median(self.total_non_monotonic_argmax_diffs):.1f}\n")
      f.write(f"\tTop {self.k} seq tags with highest non-monotonic argmax diffs:\n")
      for diff, seq_tag in sorted(self.top_k_non_monotonic_argmax_seq_tags, reverse=True):
        f.write(f"\t\t{seq_tag}: {diff:.1f}\n")

  def generate_plots(self, filename: str):
    # generate histogram of seqs_w_non_monotonic_cog_counter
    num_non_monotonic_cogs = list(self.seqs_w_non_monotonic_cog_counter.keys())
    counts = list(self.seqs_w_non_monotonic_cog_counter.values())
    plt.bar(num_non_monotonic_cogs, counts)
    plt.title("#Sequences with non-monotonic CoGs")
    plt.xlabel("#Non-monotonic CoGs")
    plt.ylabel("#Sequences")
    plt.savefig(filename)
    plt.close()


class AttentionWeightStatisticsJob(Job):
  def __init__(
          self,
          att_weights_hdf: Path,
          bpe_vocab: Path,
          ctc_alignment_hdf: Path,
          segment_file: Optional[Path],
          ctc_blank_idx: int,
          time_rqtm=1,
          mem_rqmt=2,
          returnn_python_exe=None,
          returnn_root=None
  ):
    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.att_weights_hdf = att_weights_hdf
    self.bpe_vocab = bpe_vocab
    self.ctc_alignment_hdf = ctc_alignment_hdf
    self.segment_file = segment_file
    self.ctc_blank_idx = ctc_blank_idx

    self.time_rqmt = time_rqtm
    self.mem_rqtm = mem_rqmt

    self.out_statistics = self.output_path("statistics")

  def tasks(self):
    yield Task("run", mini_task=True)

  def get_segment_list(self):
    if self.segment_file is None:
      return None
    segment_list = []
    with open(self.segment_file.get_path(), "r") as f:
      for line in f:
        segment_list.append(line.strip())
    return segment_list

  def run(self):
    segment_list = self.get_segment_list()
    center_of_gravity_statistics = CenterOfGravityStatistics()
    att_weight_statistics = AttentionWeightStatistics()
    att_weights_dict = hdf.load_hdf_data(self.att_weights_hdf, num_dims=2, segment_list=segment_list)
    ctc_alignments_dict = hdf.load_hdf_data(self.ctc_alignment_hdf, segment_list=segment_list)
    for seq_tag in att_weights_dict:
      att_weights = att_weights_dict[seq_tag]  # [S, T]
      ctc_alignment = ctc_alignments_dict[seq_tag]  # [T]
      center_of_gravity_statistics.update(att_weights, ctc_alignment, self.ctc_blank_idx, seq_tag)
      att_weight_statistics.update(att_weights, seq_tag)
      att_weights_clipped = att_weights.copy()
      att_weights_clipped[att_weights_clipped < 0.3] = 0.

    if os.path.exists(self.out_statistics.get_path()):
      os.remove(self.out_statistics.get_path())
    center_of_gravity_statistics.write_statistics(self.out_statistics)
    center_of_gravity_statistics.generate_plots()
    att_weight_statistics.write_statistics(self.out_statistics)
    att_weight_statistics.generate_plots("seqs_w_non_monotonic_cog.png")
