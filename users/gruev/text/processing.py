__all__ = ["PasteTextJob"]

from typing import List
from i6_core.util import uopen
from sisyphus import *


class PasteTextJob(Job):
    """Merges the lines of text files, similar to the 'paste' command"""

    def __init__(self, text_files: List[tk.Path]):
        """
        :param text_files:
        """
        self.text_files = text_files

        self.out_txt = self.output_path("pasted.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        file_handles = [uopen(text_file, "rt") for text_file in self.text_files]

        with uopen(self.out_txt.get_path(), "wt") as f:
            while True:
                lines = [fh.readline().strip() for fh in file_handles]
                if any(line == "" for line in lines):
                    break

                f.write(" ".join(lines))
                if not all(line == "" for line in lines):
                    f.write("\n")

        for fh in file_handles:
            fh.close()
