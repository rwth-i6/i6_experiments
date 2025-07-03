from sisyphus import *
from typing import Optional

import i6_core.util as util


class ComputeSearchErrorsJob(Job):
    def __init__(
        self,
        ground_truth_out: Optional[Path],
        recog_out: Optional[Path],
        verision: Optional[int] = 4,
    ):
        self.ground_truth_out = ground_truth_out
        self.recog_out = recog_out

        self.out_search_errors = self.output_path("search_errors")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):
        #d_gt = eval(util.uopen(self.ground_truth_out, "rt").read().replace('-inf', 'float("-inf")'),{"float":float})
        d_gt = eval(util.uopen(self.ground_truth_out, "rt").read())
        d_rec = eval(util.uopen(self.recog_out, "rt").read())
        assert isinstance(d_gt, dict)  # seq_tag -> bpe string
        assert isinstance(d_rec, dict)  # seq_tag -> bpe string

        num_seqs = 0
        num_search_errors = 0
        num_unequal = 0
        sent_oov = 0
        num_oov = 0
        num_words = 0

        # for each seq tag, calculate whether we have a search error
        for seq_tag in d_gt.keys():
            num_seqs += 1

            score_ground_truth, targets_ground_truth, oov = d_gt[seq_tag][0]
            num_oov += oov
            try:
                score_search, targets_search = d_rec[seq_tag][0]
            except ValueError:
                score_search, targets_search = d_rec[seq_tag]
            num_words += len(targets_ground_truth.split())

            # we count as search error if the label seqs differ and the search score is worse than the ground truth score
            is_search_error = False
            targets_search = targets_search.replace("@@ ", "")
            targets_search = targets_search.replace("<blank>", "")
            targets_search = " ".join(targets_search.split())
            if list(targets_ground_truth) == list(targets_search):
                assert oov == 0, "Search reached a sequence with OOV?"
                equal_label_seq = True
            else:
                num_unequal += 1
                equal_label_seq = False
                if oov > 0: #Implicitly ignore OOV sentence for search error, the score comparision between an OOV gt and hyp makes no sense after all
                    sent_oov += 1

                elif score_ground_truth > score_search:# TODO add threshold
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
                log_txt += "\n\tEqual label sequences: %s\n\t-> %s" % (
                    str(equal_label_seq),
                    "Search error!" if is_search_error else "No search error!",
                )
                log_txt += f"\n\tNum_oov: {oov}\n\n"
                f.write(log_txt)

        with open(self.out_search_errors.get_path(), "w+") as f:
            f.write("Search errors: %.2f%%" % ((num_search_errors / num_seqs) * 100) + "\n" +
                    "Search errors/total errors: %.2f%%" % ((num_search_errors / num_unequal) * 100) + "\n" +
                    "Sent_ER: %.2f%%" % ((num_unequal / num_seqs) * 100) + "\n" +
                    "Sent_OOV: %.2f%%" % ((sent_oov / num_seqs) * 100) + "\n" +
                    "OOV: %.2f%%" % ((num_oov / num_words) * 100) + "\n")
