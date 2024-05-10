from sisyphus import *
import os
import numpy as np
import h5py
from typing import Optional

import i6_core.util as util


class ComputeSearchErrorsJob(Job):
    def __init__(
        self,
        ground_truth_out: Optional[Path],
        recog_out: Optional[Path],
    ):
        self.ground_truth_out = ground_truth_out
        self.recog_out = recog_out

        self.out_search_errors = self.output_path("search_errors")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):

        d_gt = eval(util.uopen(self.ground_truth_out, "rt").read())
        d_rec = eval(util.uopen(self.recog_out, "rt").read())
        assert isinstance(d_gt, dict)  # seq_tag -> bpe string
        assert isinstance(d_rec, dict)  # seq_tag -> bpe string

        num_seqs = 0
        num_search_errors = 0

        # for each seq tag, calculate whether we have a search error
        for seq_tag in d_gt.keys():
            num_seqs += 1

            score_ground_truth, targets_ground_truth = d_gt[seq_tag][0]
            score_search, targets_search = d_rec[seq_tag][0]

            # we count as search error if the label seqs differ and the search score is worse than the ground truth score
            is_search_error = False
            if list(targets_ground_truth) == list(targets_search):
                equal_label_seq = True
            else:
                equal_label_seq = False
                if score_ground_truth > score_search:
                    is_search_error = True
                    num_search_errors += 1

            with open("search_errors_log", "a") as f:
                log_txt = "Seq Tag: %s\n\tGround-truth score: %f\n\tSearch score: %f" % (
                    seq_tag,
                    score_ground_truth,
                    score_search,
                )
                log_txt += "\n\tGround-truth seq: %s\n\tSearch seq:       %s" % (
                    str(targets_ground_truth),
                    str(targets_search),
                )
                log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
                    str(equal_label_seq),
                    "Search error!" if is_search_error else "No search error!",
                )
                f.write(log_txt)

        with open(self.out_search_errors.get_path(), "w+") as f:
            f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100))
#
# class ComputeSearchErrorsJob(Job):
#     def __init__(
#         self,
#         ground_truth_scores_hdf: Optional[Path],
#         search_scores_hdf: Optional[Path],
#         ground_truth_targets_hdf: Path,
#         search_targets_hdf: Path,
#         blank_idx: int,
#     ):
#         self.ground_truth_scores_hdf = ground_truth_scores_hdf
#         self.search_scores_hdf = search_scores_hdf
#         self.ground_truth_targets_hdf = ground_truth_targets_hdf
#         self.search_targets_hdf = search_targets_hdf
#         self.blank_idx = blank_idx
#
#         self.out_search_errors = self.output_path("search_errors")
#
#     def tasks(self):
#         yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)
#
#     def run(self):
#         blank_idx = self.blank_idx
#
#         # keys: seq_tags; values: data (targets and model_scores for ground truth and search)
#         data_dict = {}
#
#         # first, we load all the necessary data (targets, model scores) from the hdf files
#         for data_name, data_hdf in (
#             ("ground_truth_scores", self.ground_truth_scores_hdf.get_path()),
#             ("search_scores", self.search_scores_hdf.get_path()),
#             ("ground_truth_targets", self.ground_truth_targets_hdf.get_path()),
#             ("search_targets", self.search_targets_hdf.get_path()),
#         ):
#             with h5py.File(data_hdf, "r") as f:
#                 seq_tags = f["seqTags"][()]
#                 seq_tags = [str(tag) for tag in seq_tags]  # convert from byte to string
#                 # initialize the keys of the data_dict once
#                 if len(data_dict) == 0:
#                     for seq_tag in seq_tags:
#                         data_dict[seq_tag] = {}
#
#                 # the data and seq lens
#                 hdf_data = f["inputs"][()]
#                 seq_lens = f["seqLengths"][()][:, 0]
#                 # cut out each seq from the flattened 1d tensor according to its seq len and store it in the dict
#                 # indexed by its seq tag
#                 for i, seq_len in enumerate(seq_lens):
#                     data_dict[seq_tags[i]][data_name] = hdf_data[:seq_len]
#                     hdf_data = hdf_data[seq_len:]
#
#         num_seqs = 0
#         num_search_errors = 0
#
#         # for each seq tag, calculate whether we have a search error
#         for seq_tag in data_dict:
#             num_seqs += 1
#
#             label_model_scores_ground_truth = data_dict[seq_tag]["ground_truth_scores"]
#             targets_ground_truth = data_dict[seq_tag]["ground_truth_targets"]
#             label_model_scores_search = data_dict[seq_tag]["search_scores"]
#             targets_search = data_dict[seq_tag]["search_targets"]
#
#             non_blank_targets_ground_truth = targets_ground_truth[targets_ground_truth != blank_idx]
#             non_blank_targets_search = targets_search[targets_search != blank_idx]
#
#             scores_dict = {}  # typing: Dict[str, float]
#
#             for alias, label_model_scores in [
#                 ("ground_truth", label_model_scores_ground_truth),
#                 ("search", label_model_scores_search),
#             ]:
#                 scores_dict[alias] = np.sum(label_model_scores)
#
#             # we count as search error if the label seqs differ and the search score is worse than the ground truth score
#             is_search_error = False
#             if list(non_blank_targets_ground_truth) == list(non_blank_targets_search):
#                 equal_label_seq = True
#             else:
#                 equal_label_seq = False
#                 if scores_dict["ground_truth"] > scores_dict["search"]:
#                     is_search_error = True
#                     num_search_errors += 1
#
#             with open("search_errors_log", "a") as f:
#                 log_txt = "Seq Tag: %s\n\tGround-truth score: %f\n\tSearch score: %f" % (
#                     seq_tag,
#                     scores_dict["ground_truth"],
#                     scores_dict["search"],
#                 )
#                 log_txt += "\n\tGround-truth seq: %s\n\tSearch seq: %s" % (
#                     str(non_blank_targets_ground_truth),
#                     str(non_blank_targets_search),
#                 )
#                 log_txt += "\n\tEqual label sequences: %s\n\t-> %s\n\n" % (
#                     str(equal_label_seq),
#                     "Search error!" if is_search_error else "No search error!",
#                 )
#                 f.write(log_txt)
#
#         with open(self.out_search_errors.get_path(), "w+") as f:
#             f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100))
