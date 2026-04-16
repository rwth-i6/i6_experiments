import os
import shutil
import subprocess as sp
import tempfile
import collections
import math
import re
from typing import List, Optional, Dict, Tuple

from sisyphus import tk, Job, Task, gs
from sisyphus.delayed_ops import DelayedBase

# from i6_core.lib.corpus import *
from i6_core.util import uopen
import i6_core.util as util


class RoverJob(Job):
    """
    Wroom wroom!
    """

    __sis_hash_exclude__ = {"sctk_binary_path": None}

    def __init__(
        self,
        hyps: list[tk.Path],
        *,
        sort_files: bool = False,
        method: str = "avgconf",
        alpha: float | DelayedBase = 1.0,
        null_confidence: float | DelayedBase = 0.0,
        additional_args: Optional[list[str]] = None,
        sctk_binary_path: Optional[tk.Path] = None,
    ):
        """
        :param hyps: hypothesis ctm text files
        :param sort_files: sort ctm and stm before scoring
        :param additional_args: additional command line arguments passed to the Rover binary call
        :param sctk_binary_path: set an explicit binary path.
        :param method: rover combination method, e.g. "avgconf"
        :param alpha: Alpha is the tradeoff between using word occurrence counts and confidence scores.
            Alpha = 1 means only word occurence counts are used
        :param null_confidence: Set confidence score associated with NULL transition arcs

        See SCTK/doc/rover.htm
        """
        self.set_vis_name("Rover")

        assert len(hyps) > 1  # otherwise doesnt make sense
        self.hyps = hyps
        self.method = method
        self.sort_files = sort_files
        self.additional_args = additional_args
        self.sctk_binary_path = sctk_binary_path
        self.alpha = alpha
        self.null_confidence = null_confidence

        self.out = self.output_path("outfile")
        self.rqmt = {"cpu": 1, "mem": 12, "time": 1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

    def run(self):
        if not os.environ.get("I_RECOMPILED_SCTK_I_KNOW_WHAT_I_AM_DOING", False):
            # rover in SCTK has a hard limit of max 50 hyps, hardcoded in its source code
            # it preallocates a fixed size array, and doesn't ever check if more than 50 hyps are given
            # leading to memory corruption
            # so we check it here
            assert len(self.hyps) < 50, (
                "See https://github.com/usnistgov/SCTK/blob/9688a26882a688132a5e414cadcb4c19b6fffaba/src/sclite/rover.c#L53"
            )

        sorted_files = []
        if self.sort_files:
            for hyp in self.hyps:
                sort_ctm_args = ["sort", "-k1,1", "-k3,3n", hyp.get_path()]
                (fd_ctm, tmp_ctm_file) = tempfile.mkstemp(suffix=".ctm")
                res = sp.run(sort_ctm_args, stdout=sp.PIPE)
                os.write(fd_ctm, res.stdout)
                os.close(fd_ctm)
                sorted_files.append(tmp_ctm_file)

        if self.sctk_binary_path:
            rover_path = os.path.join(self.sctk_binary_path.get_path(), "rover")
        else:
            rover_path = os.path.join(gs.SCTK_PATH, "rover") if hasattr(gs, "SCTK_PATH") else "rover"
        ctm_files = sorted_files if self.sort_files else [hyp.get_path() for hyp in self.hyps]

        alpha = self.alpha
        if isinstance(alpha, DelayedBase):
            alpha = alpha.get()
        null_confidence = self.null_confidence
        if isinstance(null_confidence, DelayedBase):
            null_confidence = null_confidence.get()

        print(f"Alpha: {alpha}, Null confidence: {null_confidence}")

        args = [
            rover_path,
            "-o",
            self.out.get_path(),
            "-m",
            self.method,
            "-a",
            str(alpha),
            "-c",
            str(null_confidence),
        ]
        for ctm_file in ctm_files:
            args += ["-h", ctm_file, "ctm"]

        if self.additional_args is not None:
            args += self.additional_args

        sp.check_call(args)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 3
        return super().hash(d)


class SearchToRoverCtms(Job):
    """
    From RETURNN beam search results, extract the hyps for rover
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(
        self,
        search_py_output: tk.Path,
        num_ctms: int,
        *,
        convert_log_probs_to_probs: bool = False,
        length_norm_words: bool = False,
        seq_order_file: Optional[tk.Path] = None,
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        """
        self.search_py_output = search_py_output
        self.out_ctms = [self.output_path(f"rover_hyp_{i}.ctm") for i in range(num_ctms)]
        self.convert_log_probs_to_probs = convert_log_probs_to_probs
        self.length_norm_words = length_norm_words
        self.seq_order_file = seq_order_file

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})

        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_ctms[0].get_path())

        if self.seq_order_file is not None:
            seq_order = eval(util.uopen(self.seq_order_file, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            assert isinstance(seq_order, (dict, list, tuple))
        else:
            seq_order = d.keys()

        file_handles = [util.uopen(out_ctm, "wt") for out_ctm in self.out_ctms]
        try:
            # write ctm
            for out in file_handles:
                out.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
            for seq_tag in seq_order:
                entry = d[seq_tag]
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                assert isinstance(seq_tag, str), f"invalid seq_order entry {seq_tag!r} (type {type(seq_tag).__name__})"

                # sort entry, highest score first
                entry = sorted(entry, key=lambda x: x[0], reverse=True)
                if len(entry) > len(self.out_ctms):
                    entry = entry[: len(self.out_ctms)]

                seg_start = 0.0
                seg_end = 1000.0
                for i, (score, text) in enumerate(entry):
                    out = file_handles[i]
                    out.write(f";; {seq_tag} ({seg_start:f}-{seg_end:f})\n")
                    words = text.split()
                    time_step_per_word = (seg_end - seg_start) / max(len(words), 1)
                    avg_dur = time_step_per_word * 0.9
                    count = 0

                    real_score = score
                    if self.length_norm_words:
                        real_score = score / max(len(words), 1)
                    if self.convert_log_probs_to_probs:
                        real_score = math.exp(real_score)

                    for i in range(len(words)):
                        out.write(
                            f"{seq_tag} 1 {seg_start + time_step_per_word * i} {avg_dur} {words[i]} {real_score}\n"
                        )
                        count += 1
                    if count == 0:
                        # some logic to prevent empty sequences
                        # this is just copied from sclitejob
                        out.write(f"{seq_tag} 1 {seg_start} {avg_dur} <empty-sequence> {real_score}\n")
        finally:
            for fh in file_handles:
                fh.close()
