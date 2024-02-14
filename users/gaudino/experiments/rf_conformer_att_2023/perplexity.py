from sisyphus import *
import os
import numpy as np
import h5py
from typing import Optional

import i6_core.util as util


class ComputePerplexityJob(Job):
    def __init__(
        self,
        ground_truth_out: Optional[Path],
    ):
        self.ground_truth_out = ground_truth_out

        self.out_ppl = self.output_path("ppl")

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0}, mini_task=True)

    def run(self):

        d_gt = eval(util.uopen(self.ground_truth_out, "rt").read())
        assert isinstance(d_gt, dict)  # seq_tag -> bpe string

        scores = []
        lens = []
        for seq_tag in d_gt.keys():
            score_ground_truth, seq_len, _ = d_gt[seq_tag][0]
            scores.append(score_ground_truth)
            lens.append(seq_len)

        scores = np.array(scores)
        lens = np.array(lens)

        ppl = np.exp(-np.sum(scores) / np.sum(lens))

        with open(self.out_ppl.get_path(), "w+") as f:
            f.write("Perplexity: %f" % ppl)
