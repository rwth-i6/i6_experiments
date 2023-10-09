from sisyphus import *

import ast
import numpy as np
import h5py
from typing import Optional


class CalcSearchErrorJobV2(Job):
  def __init__(
          self,
          label_model_scores_ground_truth_hdf: Path,
          length_model_scores_ground_truth_hdf: Optional[Path],
          targets_ground_truth_hdf: Path,
          label_model_scores_search_hdf: Path,
          length_model_scores_search_hdf: Optional[Path],
          targets_search_hdf: Path,
          blank_idx: Optional[int],
          json_vocab_path: Path,
  ):
    assert blank_idx is None or (
            length_model_scores_ground_truth_hdf is not None and length_model_scores_search_hdf is not None)

    self.targets_search_hdf = targets_search_hdf
    self.length_model_scores_search_hdf = length_model_scores_search_hdf
    self.label_model_scores_search_hdf = label_model_scores_search_hdf
    self.targets_ground_truth_hdf = targets_ground_truth_hdf
    self.length_model_scores_ground_truth_hdf = length_model_scores_ground_truth_hdf
    self.label_model_scores_ground_truth_hdf = label_model_scores_ground_truth_hdf
    self.blank_idx = blank_idx
    self.json_vocab_path = json_vocab_path

    self.out_search_errors = self.output_path("search_errors")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    json_vocab_path = self.json_vocab_path.get_path()
    blank_idx = self.blank_idx

    if self.blank_idx is None:
      length_model_ground_truth_hdf_path = None
      length_model_search_hdf_path = None
    else:
      length_model_ground_truth_hdf_path = self.length_model_scores_ground_truth_hdf.get_path()
      length_model_search_hdf_path = self.length_model_scores_search_hdf.get_path()

    # keys: seq_tags; values: data (targets and model_scores for ground truth and search)
    data_dict = {}

    # first, we load all the necessary data (targets, model scores) from the hdf files
    for data_name, data_hdf in (
            ("label_model_scores_ground_truth", self.label_model_scores_ground_truth_hdf.get_path()),
            ("length_model_scores_ground_truth", length_model_ground_truth_hdf_path),
            ("targets_ground_truth", self.targets_ground_truth_hdf.get_path()),
            ("label_model_scores_search", self.label_model_scores_search_hdf.get_path()),
            ("length_model_scores_search", length_model_search_hdf_path),
            ("targets_search", self.targets_search_hdf.get_path()),
    ):
      if data_hdf is not None:
        with h5py.File(data_hdf, "r") as f:
          seq_tags = f["seqTags"][()]
          seq_tags = [str(tag) for tag in seq_tags]  # convert from byte to string
          # initialize the keys of the data_dict once
          if len(data_dict) == 0:
            for seq_tag in seq_tags:
              data_dict[seq_tag] = {}

          # the data and seq lens
          hdf_data = f["inputs"][()]
          seq_lens = f["seqLengths"][()][:, 0]
          # cut out each seq from the flattened 1d tensor according to its seq len and store it in the dict
          # indexed by its seq tag
          for i, seq_len in enumerate(seq_lens):
            data_dict[seq_tags[i]][data_name] = hdf_data[:seq_len]
            hdf_data = hdf_data[seq_len:]

    # load vocabulary as dictionary
    with open(json_vocab_path, "r") as f:
      json_data = f.read()
      vocab = ast.literal_eval(json_data)
      vocab = {v: k for k, v in vocab.items()}

    num_seqs = 0
    num_search_errors = 0

    # for each seq tag, calculate whether we have a search error
    for seq_tag in data_dict:
      num_seqs += 1

      label_model_scores_ground_truth = data_dict[seq_tag]["label_model_scores_ground_truth"]
      if "length_model_scores_ground_truth" in data_dict[seq_tag]:
        length_model_scores_ground_truth = data_dict[seq_tag]["length_model_scores_ground_truth"]
      else:
        length_model_scores_ground_truth = 0
      targets_ground_truth = data_dict[seq_tag]["targets_ground_truth"]
      label_model_scores_search = data_dict[seq_tag]["label_model_scores_search"]
      if "length_model_scores_search" in data_dict[seq_tag]:
        length_model_scores_search = data_dict[seq_tag]["length_model_scores_search"]
      else:
        length_model_scores_search = 0
      targets_search = data_dict[seq_tag]["targets_search"]

      non_blank_targets_ground_truth = targets_ground_truth[targets_ground_truth != blank_idx]
      non_blank_targets_search = targets_search[targets_search != blank_idx]

      # # quick dirty fix: remove the trailing SOS label and corresponding score
      # # this only exists in the ground-truth because it is removed by RETURNN in the search
      # # we only can compare the seq scores when either non or both of ground-truth and found seq end with SOS
      # non_blank_targets_ground_truth = non_blank_targets_ground_truth[:-1]
      # label_model_scores_ground_truth = label_model_scores_ground_truth[:-1]

      scores_dict = {}
      label_model_scores_dict = {}
      length_model_scores_dict = {}

      for alias, label_model_scores, length_model_scores, targets in (
              ("ground_truth", label_model_scores_ground_truth, length_model_scores_ground_truth, targets_ground_truth),
              ("search", label_model_scores_search, length_model_scores_search, targets_search),
      ):
        label_model_scores_sum = np.sum(label_model_scores)
        length_model_scores_sum = np.sum(length_model_scores)
        output_log_score = label_model_scores_sum + length_model_scores_sum

        scores_dict[alias] = output_log_score
        label_model_scores_dict[alias] = label_model_scores
        length_model_scores_dict[alias] = length_model_scores

      # we count as search error if the label seqs differ and the search score is worse than the ground truth score
      is_search_error = False
      if list(non_blank_targets_ground_truth) == list(non_blank_targets_search):
        equal_label_seq = True
      else:
        equal_label_seq = False
        if scores_dict["ground_truth"] > scores_dict["search"]:
          is_search_error = True
          num_search_errors += 1

      with open("search_errors_log", "a") as f:
        log_txt = "Seq Tag: %s\n\tGround-truth score: %f\n\tSearch score: %f" % (
          seq_tag,
          scores_dict["ground_truth"],
          scores_dict["search"]
        )
        log_txt += "\n\tGround-truth seq: %s\n\tSearch seq: %s" % (
          str(non_blank_targets_ground_truth),
          str(non_blank_targets_search)
        )
        log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
          str(equal_label_seq),
          "Search error!" if is_search_error else "No search error!"
        )
        f.write(log_txt)

      if seq_tag in [
        b"dev-other/7697-105815-0015/7697-105815-0015"
      ]:
        with open("detailed_examples", "a") as f:
          log_txt = "Seq Tag: %s\n\tGround-truth label model scores: %s\n\tGround truth length model scores: %s" % (
            seq_tag,
            np.array_str(label_model_scores_dict["ground_truth"]),
            np.array_str(length_model_scores_dict["ground_truth"]),
          )
          log_txt += "\n\tGround-truth seq: %s" % (
            str(non_blank_targets_ground_truth),
          )
          log_txt += "\n\tSearch label model scores: %s\n\tSearch length model scores: %s" % (
            np.array_str(label_model_scores_dict["search"]),
            np.array_str(length_model_scores_dict["search"]),
          )
          log_txt += "\n\tSearch seq: %s" % (
            str(non_blank_targets_ground_truth),
          )
          log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
            str(equal_label_seq),
            "Search error!" if is_search_error else "No search error!"
          )
          f.write(log_txt)



    with open(self.out_search_errors.get_path(), "w+") as f:
      f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100))
