__all__ = ["SetupFairseqJob"]

import logging
import os
import shutil
import subprocess as sp
from typing import Optional

from sisyphus import *
from i6_core.util import get_executable_path


class SetupFairseqJob(Job):
    """
    Set up a fairseq repository. Needed to build Cython components.
    """

    def __init__(self, fairseq_root: Path, python_exe: Optional[Path] = None):
        self.fairseq_root = fairseq_root
        self.python_exe = python_exe
        self.out_fairseq_root = self.output_path("fairseq", directory=True)

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        shutil.copytree(self.fairseq_root.get(), self.out_fairseq_root.get(), dirs_exist_ok=True, symlinks=True)
        python_exe = get_executable_path(self.python_exe, "FAIRSEQ_PYTHON_EXE")
        assert python_exe is not None
        args = [python_exe, "setup.py", "build_ext", "--inplace"]
        sp.run(args, cwd=self.out_fairseq_root.get(), check=True)
