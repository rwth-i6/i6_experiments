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
        additional_args: Optional[List[str]] = None,
        sctk_binary_path: Optional[tk.Path] = None,
        precision_ndigit: Optional[int] = 1,
    ):
        """
        :param hyps: hypothesis ctm text files
        :param sort_files: sort ctm and stm before scoring
        :param additional_args: additional command line arguments passed to the Sclite binary call
        :param sctk_binary_path: set an explicit binary path.

        """
        self.set_vis_name("Rover - %s" % ("CER" if cer else "WER"))

        assert len(hyps) > 2  # otherwise doesnt make sense
        self.hyps = hyps
        self.sort_files = sort_files
        self.additional_args = additional_args
        self.sctk_binary_path = sctk_binary_path
        self.precision_ndigit = precision_ndigit

        self.out = self.output_path("outfile")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
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

        args = [
            rover_path,
            "-o",
            self.out,
        ]
        for ctm_file in ctm_files:
            args += ["-h", ctm_file, "ctm"]

        if self.additional_args is not None:
            args += self.additional_args

        sp.check_call(args)
