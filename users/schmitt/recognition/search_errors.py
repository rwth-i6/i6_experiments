from sisyphus import *

import ast
import numpy as np
import h5py
from typing import Optional, Dict

from i6_core import util

from i6_experiments.users.schmitt import hdf


class CalcSearchErrorJobV2(Job):
  def __init__(
          self,
          label_sync_scores_ground_truth_hdf: Dict[str, Path],
          frame_sync_scores_ground_truth_hdf: Dict[str, Path],
          targets_ground_truth_hdf: Path,
          label_sync_scores_search_hdf: Dict[str, Path],
          frame_sync_scores_search_hdf: Dict[str, Path],
          targets_search_hdf: Path,
          blank_idx: Optional[int],
          json_vocab_path: Path,
  ):
    assert blank_idx is None or (
            label_sync_scores_ground_truth_hdf is not None and frame_sync_scores_search_hdf is not None)

    self.targets_search_hdf = targets_search_hdf
    self.frame_sync_scores_search_hdf = frame_sync_scores_search_hdf
    self.label_sync_scores_search_hdf = label_sync_scores_search_hdf
    self.targets_ground_truth_hdf = targets_ground_truth_hdf
    self.frame_sync_scores_ground_truth_hdf = frame_sync_scores_ground_truth_hdf
    self.label_sync_scores_ground_truth_hdf = label_sync_scores_ground_truth_hdf
    self.blank_idx = blank_idx
    self.json_vocab_path = json_vocab_path

    self.out_search_errors = self.output_path("search_errors")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  @staticmethod
  def load_hdf_data(hdf_path: Path):
    """
    Load data from an hdf file. Returns dict of form {seq_tag: data}
    :param hdf_path:
    :return:
    """
    with h5py.File(hdf_path.get_path(), "r") as f:
      seq_tags = f["seqTags"][()]
      seq_tags = [tag.decode("utf8") if isinstance(tag, bytes) else tag for tag in seq_tags]  # convert from byte to string
      data_dict = {}

      # the data and seq lens
      hdf_data = f["inputs"][()]
      seq_lens = f["seqLengths"][()][:, 0]
      # cut out each seq from the flattened 1d tensor according to its seq len and store it in the dict
      # indexed by its seq tag
      for i, seq_len in enumerate(seq_lens):
        data_dict[seq_tags[i]] = hdf_data[:seq_len]
        hdf_data = hdf_data[seq_len:]
    return data_dict

  def run(self):
    json_vocab_path = self.json_vocab_path.get_path()
    blank_idx = self.blank_idx

    # keys: seq_tags; values: data (targets and model_scores for ground truth and search)
    data_dict = {}

    # first, we load all the necessary data (targets, model scores) from the hdf files
    data_dict["label_sync_scores_ground_truth"] = {
      data_name: self.load_hdf_data(
        hdf_path=hdf_path
      ) for data_name, hdf_path in self.label_sync_scores_ground_truth_hdf.items()
    }
    data_dict["frame_sync_scores_ground_truth"] = {
      data_name: self.load_hdf_data(
        hdf_path=hdf_path
      ) for data_name, hdf_path in self.frame_sync_scores_ground_truth_hdf.items()
    }
    data_dict["label_sync_scores_search"] = {
      data_name: self.load_hdf_data(
        hdf_path=hdf_path
      ) for data_name, hdf_path in self.label_sync_scores_search_hdf.items()
    }
    data_dict["frame_sync_scores_search"] = {
      data_name: self.load_hdf_data(
        hdf_path=hdf_path
      ) for data_name, hdf_path in self.frame_sync_scores_search_hdf.items()
    }
    data_dict["targets_ground_truth"] = self.load_hdf_data(hdf_path=self.targets_ground_truth_hdf)
    data_dict["targets_search"] = self.load_hdf_data(hdf_path=self.targets_search_hdf)

    # load vocabulary as dictionary
    with open(json_vocab_path, "r") as f:
      json_data = f.read()
      vocab = ast.literal_eval(json_data)
      vocab = {v: k for k, v in vocab.items()}

    num_seqs = 0
    num_search_errors = 0

    # for each seq tag, calculate whether we have a search error
    for seq_tag in data_dict["targets_search"]:
      num_seqs += 1

      # get the scores for the current seq tag
      label_sync_scores_ground_truth = {
        score_name: data_dict["label_sync_scores_ground_truth"][score_name][seq_tag] for score_name in data_dict["label_sync_scores_ground_truth"]
      }
      frame_sync_scores_ground_truth = {
        score_name: data_dict["frame_sync_scores_ground_truth"][score_name][seq_tag] for score_name in data_dict["frame_sync_scores_ground_truth"]
      }
      targets_ground_truth = data_dict["targets_ground_truth"][seq_tag]
      label_sync_scores_search = {
        score_name: data_dict["label_sync_scores_search"][score_name][seq_tag] for score_name in data_dict["label_sync_scores_search"]
      }
      frame_sync_scores_search = {
        score_name: data_dict["frame_sync_scores_search"][score_name][seq_tag] for score_name in data_dict["frame_sync_scores_search"]
      }
      targets_search = data_dict["targets_search"][seq_tag]

      non_blank_targets_ground_truth = targets_ground_truth[targets_ground_truth != blank_idx]
      non_blank_targets_search = targets_search[targets_search != blank_idx]

      # # quick dirty fix: remove the trailing SOS label and corresponding score
      # # this only exists in the ground-truth because it is removed by RETURNN in the search
      # # we only can compare the seq scores when either non or both of ground-truth and found seq end with SOS
      # non_blank_targets_ground_truth = non_blank_targets_ground_truth[:-1]
      # label_model_scores_ground_truth = label_model_scores_ground_truth[:-1]

      scores_dict = {}
      label_sync_scores_dict = {}
      frame_sync_scores_dict = {}
      frame_sync_scores_accum_dict = {}

      for alias, label_sync_scores, frame_sync_scores, targets in (
              ("ground_truth", label_sync_scores_ground_truth, frame_sync_scores_ground_truth, targets_ground_truth),
              ("search", label_sync_scores_search, frame_sync_scores_search, targets_search),
      ):
        frame_sync_scores_accum_dict[alias] = {}
        label_sync_scores_dict[alias] = {}
        frame_sync_scores_dict[alias] = {}

        # sum over all kinds of label sync scores
        label_sync_scores_sum = 0
        for label_sync_score_name in label_sync_scores:
          label_sync_score = label_sync_scores[label_sync_score_name]
          label_sync_scores_dict[alias][label_sync_score_name] = label_sync_score
          label_sync_scores_sum += np.sum(label_sync_score)

        # sum over all kinds of frame sync scores
        frame_sync_scores_sum = 0
        for frame_sync_score_name in frame_sync_scores:
          frame_sync_score = frame_sync_scores[frame_sync_score_name]
          frame_sync_scores_dict[alias][frame_sync_score_name] = frame_sync_score
          frame_sync_scores_sum += np.sum(frame_sync_score)

        # sum of all scores
        output_log_score = label_sync_scores_sum + frame_sync_scores_sum
        scores_dict[alias] = output_log_score

        # for frame-sync scores, calculate the accumulated score
        # i.e. for each label, add the frame-sync scores of the blank frames preceding the label frame and the score
        # of the label frame itself
        accum_frame_sync_scores = []
        curr_accum_frame_score = 0
        for frame_sync_score_name in frame_sync_scores:
          for (target, frame_sync_score) in zip(targets, frame_sync_scores[frame_sync_score_name]):
            curr_accum_frame_score += frame_sync_score
            if target != blank_idx:
              accum_frame_sync_scores.append(curr_accum_frame_score)
              curr_accum_frame_score = 0

          frame_sync_scores_accum_dict[alias][frame_sync_score_name] = accum_frame_sync_scores

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
        log_txt += "\n\tGround-truth non-blank label seq: %s\n\tSearch non-blank label seq: %s" % (
          str(non_blank_targets_ground_truth),
          str(non_blank_targets_search)
        )
        log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
          str(equal_label_seq),
          "Search error!" if is_search_error else "No search error!"
        )

        if seq_tag in [
          "dev-other/7697-105815-0015/7697-105815-0015"
        ]:
          # dump the different scores for each label
          log_txt += "Detailed score breakdown:\n\n"
          log_txt += "\n\tGround-truth seq: %s\n\tSearch seq: %s" % (
            str(targets_ground_truth),
            str(targets_search)
          )
          for frame_sync_score_name in frame_sync_scores_dict["search"]:
            log_txt += "\n\tGround-truth %s scores:\n %s\n\tSearch %s scores:\n %s" % (
              frame_sync_score_name,
              "\n".join([str(score) for score in np.cumsum(frame_sync_scores_dict["ground_truth"][frame_sync_score_name])]),
              frame_sync_score_name,
              "\n".join([str(score) for score in np.cumsum(frame_sync_scores_dict["search"][frame_sync_score_name])]),
            )
          for frame_sync_score_name in frame_sync_scores_dict["search"]:
            log_txt += "\n\tGround-truth accum %s scores:\n %s\n\tSearch accum %s scores:\n %s" % (
              frame_sync_score_name,
              "\n".join([str(score) for score in frame_sync_scores_accum_dict["ground_truth"][frame_sync_score_name]]),
              frame_sync_score_name,
              "\n".join([str(score) for score in frame_sync_scores_accum_dict["search"][frame_sync_score_name]]),
            )
          for label_sync_score_name in label_sync_scores_dict["search"]:
            log_txt += "\n\tGround-truth %s scores:\n %s\n\tSearch %s scores:\n %s\n\n" % (
              label_sync_score_name,
              "\n".join([str(score) for score in label_sync_scores_dict["ground_truth"][label_sync_score_name]]),
              label_sync_score_name,
              "\n".join([str(score) for score in label_sync_scores_dict["search"][label_sync_score_name]]),
            )

        log_txt += "-----------------------------------------------------------------------------------\n"

        f.write(log_txt)

      # if seq_tag in [
      #   "dev-other/7697-105815-0015/7697-105815-0015"
      # ]:
      #   with open("detailed_examples", "a") as f:
      #     log_txt = "Seq Tag: %s\n\tGround-truth label model scores: %s\n\tGround truth length model scores: %s" % (
      #       seq_tag,
      #       np.array_str(label_model_scores_dict["ground_truth"]),
      #       np.array_str(length_model_scores_dict["ground_truth"]),
      #     )
      #     log_txt += "\n\tGround-truth seq: %s" % (
      #       str(non_blank_targets_ground_truth),
      #     )
      #     log_txt += "\n\tSearch label model scores: %s\n\tSearch length model scores: %s" % (
      #       np.array_str(label_model_scores_dict["search"]),
      #       np.array_str(length_model_scores_dict["search"]),
      #     )
      #     log_txt += "\n\tSearch seq: %s" % (
      #       str(non_blank_targets_ground_truth),
      #     )
      #     log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
      #       str(equal_label_seq),
      #       "Search error!" if is_search_error else "No search error!"
      #     )
      #     f.write(log_txt)

    with open(self.out_search_errors.get_path(), "w+") as f:
      f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100))


class CalcSearchErrorJobRF(Job):
  def __init__(
          self,
          ground_truth_scores_file: Path,
          ground_truth_hdf: Path,
          search_hyps_file: Path,
          search_seqs_hdf: Path,
  ):
    self.ground_truth_scores_file = ground_truth_scores_file
    self.ground_truth_hdf = ground_truth_hdf
    self.search_hyps_file = search_hyps_file
    self.search_seqs_hdf = search_seqs_hdf

    self.out_search_errors = self.output_path("search_errors")

  def tasks(self):
    yield Task(
      "run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

  def run(self):
    with util.uopen(self.ground_truth_scores_file.get_path(), "r") as f:
      ground_truth_scores = eval(f.read())  # Dict[tag, score]

    with util.uopen(self.search_hyps_file.get_path(), "r") as f:
      search_hyps = eval(f.read())  # Dict[tag, List[Tuple[score, hyp]]]
      best_search_scores = {
        tag: max(hyp, key=lambda x: x[0])[0] for tag, hyp in search_hyps.items()
      }

    ground_truth_data_dict = hdf.load_hdf_data(hdf_path=self.ground_truth_hdf)  # type: Dict[str, np.ndarray]
    search_data_dict = hdf.load_hdf_data(hdf_path=self.search_seqs_hdf)  # type: Dict[str, np.ndarray]

    num_seqs = len(ground_truth_scores)
    num_search_errors = 0
    for tag in ground_truth_scores:
      if list(ground_truth_data_dict[tag]) != list(search_data_dict[tag]):

        print("Output mismatch for tag %s" % tag)
        print("Ground truth: %s" % ground_truth_data_dict[tag])
        print("Ground truth score: %s" % ground_truth_scores[tag])
        print("Search: %s" % search_data_dict[tag])
        print("Search score: %s" % best_search_scores[tag])

        if ground_truth_scores[tag] > best_search_scores[tag]:
          print("Search error!")
          num_search_errors += 1
        else:
          print("No search error!")

        print("\n ------------------------------------ \n")

    with open(self.out_search_errors.get_path(), "w+") as f:
      f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100))
