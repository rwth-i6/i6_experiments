from sisyphus import *
from sisyphus import Path

import os
from typing import Optional
import numpy as np
import heapq
from collections import Counter
from matplotlib import pyplot as plt
import json, ast

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)
from recipe.i6_experiments.users.schmitt import hdf


class CenterOfGravityStatistics:
  def __init__(self):
    self.stats = {}
    self.k = 10

  def init_stats(self, name: str):
    self.stats[name] = {
      "total_error": np.array([]),
      "top_k_abs_error_seq_tags": [],
      "max_abs_err_per_seq_counter": Counter()
    }

  def update(self, att_weight_summary, att_weight_summary_name, ctc_positions, seq_tag):
    if att_weight_summary_name not in self.stats:
      self.init_stats(att_weight_summary_name)

    error = att_weight_summary - ctc_positions
    self.stats[att_weight_summary_name]["total_error"] = np.concatenate((self.stats[att_weight_summary_name]["total_error"], error))
    abs_error = np.abs(error)
    if len(self.stats[att_weight_summary_name]["top_k_abs_error_seq_tags"]) < self.k:
      heapq.heappush(self.stats[att_weight_summary_name]["top_k_abs_error_seq_tags"], (np.max(abs_error), seq_tag))
    else:
      heapq.heappushpop(self.stats[att_weight_summary_name]["top_k_abs_error_seq_tags"], (np.max(abs_error), seq_tag))

    self.stats[att_weight_summary_name]["max_abs_err_per_seq_counter"][int(np.max(abs_error))] += 1

  def write_statistics(self, statistics_path: Path):
    for name in self.stats:
      total_abs_error = np.abs(self.stats[name]["total_error"])
      with open(statistics_path.get_path(), "a") as f:
        f.write(f"{name} of attention weights - CTC label positions\n")
        f.write(f"\tMean absolute error: {np.mean(total_abs_error):.1f}\n")
        f.write(f"\tMean error: {np.mean(self.stats[name]['total_error']):.1f}\n")
        f.write(f"\tMean squared error: {np.mean(total_abs_error ** 2):.1f}\n")
        f.write(f"\tMedian absolute error: {np.median(total_abs_error):.1f}\n")
        f.write(f"\tMedian error: {np.median(self.stats[name]['total_error']):.1f}\n")
        f.write(f"\tMin absolute error: {np.min(total_abs_error):.1f}\n")
        f.write(f"\tMin error: {np.min(self.stats[name]['total_error']):.1f}\n")
        f.write(f"\tMax absolute error: {np.max(total_abs_error):.1f}\n")
        f.write(f"\tMax error: {np.max(self.stats[name]['total_error']):.1f}\n")
        f.write(f"\tTop {self.k} seq tags with highest absolute error:\n")
        for diff, seq_tag in sorted(self.stats[name]["top_k_abs_error_seq_tags"], reverse=True):
          f.write(f"\t\t{seq_tag}: {diff:.1f}\n")

  def generate_plots(self):
    # generate histogram of max_abs_err_per_seq_counter
    for name in self.stats:
      max_abs_err = list(self.stats[name]["max_abs_err_per_seq_counter"].keys())
      counts = list(self.stats[name]["max_abs_err_per_seq_counter"].values())
      plt.bar(max_abs_err, counts)
      plt.title("#Sequences with Maximum Absolute Error\n between CoG and CTC label position")
      plt.xlabel("Maximum Absolute Error")
      plt.ylabel("#Sequences")
      plt.savefig(f"{name}_max_abs_err_per_seq.png")
      plt.close()


class AttentionWeightStatistics:
  def __init__(self):
    self.total_num_label_transitions = 0
    self.total_non_monotonic_cog_diffs = np.array([])
    self.top_k_non_monotonic_cog_seq_tags = []
    self.total_non_monotonic_argmax_diffs = np.array([])
    self.top_k_non_monotonic_argmax_seq_tags = []
    self.total_non_monotonic_pos_diffs = np.array([])
    self.top_k_non_monotonic_pos_seq_tags = []
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
    self.k = 50

    self.map_seq_tag_to_non_monotonic_argmax = {}
    self.map_seq_tag_to_non_monotonic_cog = {}
    
  # def _update_statistics(self, att_weight_positions, seq_tag):
  #   self.total_num_label_transitions += len(att_weight_positions) - 1
  #   att_weight_position_diff = np.diff(att_weight_positions)
  #   non_monotonic_pos_mask = att_weight_position_diff < 0
  #   non_monotonic_pos_diffs = np.abs(att_weight_position_diff[non_monotonic_pos_mask])
  #   max_non_monotonic_pos_diff = np.max(non_monotonic_pos_diffs) if non_monotonic_pos_diffs.size > 0 else 0
  #   # maintain top-k seq_tags with highest non_monotonic_cog_diffs
  #   if len(self.top_k_non_monotonic_cog_seq_tags) < self.k:
  #     heapq.heappush(self.top_k_non_monotonic_cog_seq_tags, (max_non_monotonic_pos_diff, seq_tag))
  #   else:
  #     heapq.heappushpop(self.top_k_non_monotonic_cog_seq_tags, (max_non_monotonic_pos_diff, seq_tag))
  #   self.total_non_monotonic_cog_diffs = np.concatenate((self.total_non_monotonic_cog_diffs, non_monotonic_pos_diffs))

  #   if np.any(non_monotonic_pos_mask):
  #     self.seq_len_statistics[">0_non_monotonic_cog"].append(att_weights.shape[1])
  #   else:
  #     self.seq_len_statistics["all_monotonic_cog"].append(att_weights.shape[1])

  #   num_non_monotonic_cogs = int(np.sum(non_monotonic_pos_mask))
  #   self.seqs_w_non_monotonic_cog_counter[num_non_monotonic_cogs] += 1
  #   if num_non_monotonic_cogs in self.seq_tags_w_non_monotonic_cog:
  #     self.seq_tags_w_non_monotonic_cog[num_non_monotonic_cogs].append(seq_tag)

  #   # entropy normalized by log(T) -> [0, 1]
  #   entropy_norm = -np.sum(att_weights * np.log(att_weights + 1e-10), axis=1) / np.log(att_weights.shape[1])  # [S]
  #   self.entropy_statistics["non_monotonic_cog"] += entropy_norm[1:][non_monotonic_pos_mask].tolist()
  #   self.entropy_statistics["monotonic_cog"] += entropy_norm[1:][~non_monotonic_pos_mask].tolist()

  #   initial_att_weights_mean = np.mean(att_weights[:, :8], axis=1)  # [S]
  #   self.initial_att_weights_statistics["non_monotonic_cog"] += initial_att_weights_mean[1:][non_monotonic_pos_mask].tolist()
  #   self.initial_att_weights_statistics["monotonic_cog"] += initial_att_weights_mean[1:][~non_monotonic_pos_mask].tolist()

  def _update_cog_statistics(self, att_weights, seq_tag):
    cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)  # [S]
    num_labels = len(cog)
    self.total_num_label_transitions += num_labels - 1
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

    self.map_seq_tag_to_non_monotonic_cog[seq_tag] = np.sum(non_monotonic_cog_mask) / (num_labels - 1) * 100

  def _update_argmax_statistics(self, att_weights, seq_tag):
    argmax = np.argmax(att_weights, axis=1)  # [S]
    num_labels = len(argmax)
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

    self.map_seq_tag_to_non_monotonic_argmax[seq_tag] = np.sum(non_monotonic_argmax_mask) / (num_labels - 1) * 100

  def update(self, att_weights, seq_tag):
    self._update_cog_statistics(att_weights, seq_tag)
    self._update_argmax_statistics(att_weights, seq_tag)

  def write_statistics(
          self,
          statistics_path: Path,
          seq_tag_to_non_monotonic_cog_file: Path,
          seq_tag_to_non_monotonic_argmax_file: Path
  ):
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

    with open(seq_tag_to_non_monotonic_cog_file.get_path(), "w") as f:
      f.write(json.dumps(self.map_seq_tag_to_non_monotonic_cog))

    with open(seq_tag_to_non_monotonic_argmax_file.get_path(), "w") as f:
      f.write(json.dumps(self.map_seq_tag_to_non_monotonic_argmax))

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
          ctc_alignment_hdf: Optional[Path],
          segment_file: Optional[Path],
          ctc_blank_idx: Optional[int],
          remove_last_label: bool = False
  ):
    if ctc_alignment_hdf is not None:
      assert ctc_blank_idx is not None

    self.att_weights_hdf = att_weights_hdf
    self.bpe_vocab = bpe_vocab
    self.ctc_alignment_hdf = ctc_alignment_hdf
    self.segment_file = segment_file
    self.ctc_blank_idx = ctc_blank_idx
    self.remove_last_label = remove_last_label

    self.out_statistics = self.output_path("statistics")
    self.out_seq_tag_to_non_monotonic_cog = self.output_path("seq_tag_to_non_monotonic_cog.json")
    self.out_seq_tag_to_non_monotonic_argmax = self.output_path("seq_tag_to_non_monotonic_argmax.json")

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
    center_of_gravity_statistics = CenterOfGravityStatistics() if self.ctc_alignment_hdf is not None else None
    att_weight_statistics = AttentionWeightStatistics()
    att_weights_dict = hdf.load_hdf_data(self.att_weights_hdf, num_dims=2, segment_list=segment_list)
    ctc_alignments_dict = hdf.load_hdf_data(
      self.ctc_alignment_hdf, segment_list=segment_list) if self.ctc_alignment_hdf is not None else None
    for seq_tag in att_weights_dict:
      att_weights = att_weights_dict[seq_tag]  # [S, T]
      if self.remove_last_label:
        att_weights = att_weights[:-1, :]
      att_weight_statistics.update(att_weights, seq_tag)

      if center_of_gravity_statistics is not None:
        att_weights_cog = np.sum(att_weights * np.arange(att_weights.shape[1])[None, :], axis=1)  # [S]
        att_weights_argmax = np.argmax(att_weights, axis=1)
        ctc_alignment = ctc_alignments_dict[seq_tag]  # [T]
        ctc_positions = np.where(ctc_alignment != self.ctc_blank_idx)[0]  # [S]
        if self.remove_last_label:
          ctc_positions = ctc_positions[:-1]
        center_of_gravity_statistics.update(att_weights_cog, "CoG", ctc_positions, seq_tag)
        center_of_gravity_statistics.update(att_weights_argmax, "Argmax", ctc_positions, seq_tag)

    if os.path.exists(self.out_statistics.get_path()):
      os.remove(self.out_statistics.get_path())
    if center_of_gravity_statistics is not None:
      center_of_gravity_statistics.write_statistics(self.out_statistics)
      center_of_gravity_statistics.generate_plots()
    att_weight_statistics.write_statistics(
      self.out_statistics,
      seq_tag_to_non_monotonic_cog_file=self.out_seq_tag_to_non_monotonic_cog,
      seq_tag_to_non_monotonic_argmax_file=self.out_seq_tag_to_non_monotonic_argmax
    )
    att_weight_statistics.generate_plots("seqs_w_non_monotonic_cog.png")
    
    
class ScatterAttentionWeightMonotonicityAgainstWERJob(Job):
  name_map = {
    "algore": "AlGore",
    "barryschwartz": "BarrySchwartz",
    "blaiseaguerayarcas": "BlaiseAguerayArcas",
    "briancox": "BrianCox",
    "craigventer": "CraigVenter",
    "davidmerrill": "DavidMerrill",
    "elizabethgilbert": "ElizabethGilbert",
    "wadedavis": "WadeDavis",
  }

  def __init__(
          self,
          seq_tag_to_non_monotonicity_map_file: Path,
          sclite_report_dir: Path,
          dataset: str
  ):
    assert dataset in ("tedlium2",)

    self.seq_tag_to_non_monotonicity_map_file = seq_tag_to_non_monotonicity_map_file
    self.sclite_report_dir = sclite_report_dir
    self.dataset = dataset

    self.out_scatter = self.output_path("scatter")

  def tasks(self):
    yield Task("run", mini_task=True)

  @staticmethod
  def _get_seq_tag(id_: str, file_name: str, dataset: str):
    if dataset == "tedlium2":
      _, seq_idx = id_.split("-")
      name, year = file_name.split("_")
      name = ScatterAttentionWeightMonotonicityAgainstWERJob.name_map[name]
      return f"TED-LIUM-realease2/{name}_{year.upper()}/{int(seq_idx) + 1}"

  @staticmethod
  def parse_pra_file(sclite_pra_file: str, dataset: str):
    seq_tag_to_wer = {}
    
    id = None
    file_name = None
    
    with open(sclite_pra_file, "r") as f:
      for line in f:
        if line.startswith("id:"):
          # e.g. 'id: (barry_schwartz-090)\n'
          id = line.split("(")[1][:-2]
          continue
        if line.startswith("File:"):
          # e.g. 'File: barryschwartz_2005g\n'
          file_name = line.split(" ")[1][:-1]
          continue
        if line.startswith("Scores:"):
          scores = line.split(" ")[-4:]
          scores = [int(score) for score in scores]
          num_words = sum(scores)
          num_errors = sum(scores[1:])
          seq_tag = ScatterAttentionWeightMonotonicityAgainstWERJob._get_seq_tag(id, file_name, dataset)

          seq_tag_to_wer[seq_tag] = num_errors / num_words * 100

    return seq_tag_to_wer

  def run(self):
    sclite_pra_file = f"{self.sclite_report_dir.get_path()}/sclite.pra"
    seq_tag_to_wer = self.parse_pra_file(sclite_pra_file, self.dataset)
    with open(self.seq_tag_to_non_monotonicity_map_file.get_path(), "r") as f:
      seq_tag_to_monotonicity = json.load(f)

    # for completeness, we should check whether the seq tags are the same
    # however, the pra file has some weird seq ids such as "s38-000" instead of "david_merrill-044"
    # and i don't know how to map this correctly so i am just going to ignore this for now
    # this leads to 27 ignored seqs (480 vs 507)
    # assert set(seq_tag_to_wer.keys()) == set(seq_tag_to_monotonicity.keys())

    wers = []
    monotonicities = []
    for seq_tag in seq_tag_to_monotonicity:
      if seq_tag not in seq_tag_to_wer:
        # see comment above
        continue
      wers.append(seq_tag_to_wer[seq_tag])
      monotonicities.append(seq_tag_to_monotonicity[seq_tag])

    plt.scatter(monotonicities, wers)
    plt.xlabel("Non Monotonic Att Weights (%)")
    plt.ylabel("WER (%)")
    plt.title("Non-Monotonicity vs. WER")
    plt.savefig(self.out_scatter.get_path())
