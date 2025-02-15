from sisyphus import *
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
            f.write("Search errors: %f%%" % ((num_search_errors / num_seqs) * 100) + "\n")