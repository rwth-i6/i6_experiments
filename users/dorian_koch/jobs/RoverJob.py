import os
import shutil
import subprocess as sp
import tempfile
import collections
import re
from typing import List, Optional, Dict, Tuple

from sisyphus import *
from i6_core.lib.corpus import *
from i6_core.util import uopen
import i6_core.util as util


class RoverJob(Job):
    """
    Wroom wroom!
    """

    __sis_hash_exclude__ = {"sctk_binary_path": None, "precision_ndigit": 1}

    def __init__(
        self,
        hyps: List[tk.Path],
        *,
        sort_files: bool = False,
        method: str = "meth1",
        additional_args: Optional[List[str]] = None,
        sctk_binary_path: Optional[tk.Path] = None,
        precision_ndigit: Optional[int] = 1,
    ):
        """
        :param hyps: hypothesis ctm text files
        :param sort_files: sort ctm and stm before scoring
        :param additional_args: additional command line arguments passed to the Rover binary call
        :param sctk_binary_path: set an explicit binary path.

        """
        self.set_vis_name("Rover")

        assert len(hyps) > 2  # otherwise doesnt make sense
        self.hyps = hyps
        self.method = method
        self.sort_files = sort_files
        self.additional_args = additional_args
        self.sctk_binary_path = sctk_binary_path
        self.precision_ndigit = precision_ndigit

        self.out = self.output_path("outfile")
        self.rqmt = {"cpu": 1, "mem": 24, "time": 1}

    def tasks(self):
        yield Task("run", resume="run", rqmt=self.rqmt)

    def run(self):
        # print env
        print(dict(os.environ))
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

        args = [rover_path, "-o", self.out.get_path(), "-m", self.method]
        for ctm_file in ctm_files:
            args += ["-h", ctm_file, "ctm"]

        if self.additional_args is not None:
            args += self.additional_args

        sp.check_call(args)

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 2
        return super().hash(d)


class SearchToRoverCtms(Job):
    """
    From RETURNN beam search results, extract the hyps for rover
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, num_ctms: int, *, top_k: Optional[int] = None):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        """
        self.search_py_output = search_py_output
        self.out_ctms = [self.output_path(f"rover_hyp_{i}.ctm") for i in range(num_ctms)]
        self.top_k = top_k

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_ctms[0].get_path())
        file_handles = [util.uopen(out_ctm, "wt") for out_ctm in self.out_ctms]
        try:
            # write ctm
            for out in file_handles:
                out.write(";; <name> <track> <start> <duration> <word> <confidence>\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                assert isinstance(seq_tag, str), f"invalid seq_order entry {seq_tag!r} (type {type(seq_tag).__name__})"

                # sort entry, highest score first
                entry = sorted(entry, key=lambda x: x[0], reverse=True)
                if self.top_k is not None:
                    entry = entry[: self.top_k]

                seg_start = 0.0
                seg_end = 1000.0
                for i, (score, text) in enumerate(entry):
                    out = file_handles[i]
                    out.write(f";; {seq_tag} ({seg_start:f}-{seg_end:f})\n")
                    words = text.split()
                    time_step_per_word = (seg_end - seg_start) / max(len(words), 1)
                    avg_dur = time_step_per_word * 0.9
                    count = 0
                    for i in range(len(words)):
                        out.write(f"{seq_tag} 1 {seg_start + time_step_per_word * i} {avg_dur} {words[i]} {score}\n")
                        count += 1
                    if count == 0:
                        # sclite cannot handle empty sequences, and would stop with an error like:
                        #   hyp file '4515-11057-0054' and ref file '4515-11057-0053' not synchronized
                        #   sclite: Alignment failed.  Exiting
                        # So we make sure it is never empty.
                        # For the WER, it should not matter, assuming the reference sequence is non-empty,
                        # you will anyway get a WER of 100% for this sequence.
                        out.write(f"{seq_tag} 1 {seg_start} {avg_dur} <empty-sequence> {score}\n")
        finally:
            for fh in file_handles:
                fh.close()
