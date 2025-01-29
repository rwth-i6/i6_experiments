"""
Concatenate N-best lists
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set
from sisyphus import Job, Task, tk

import i6_core.util as util


class SearchConcatHypsJob(Job):
    """
    Takes a number of files, each with some N-best list of hyps, and concatenates them,
    such that you get a single M-best list, M=sum(each N).
    """

    def __init__(self, search_py_output: List[tk.Path], *, output_gzip: bool = True):
        """
        :param search_py_output: search output file from RETURNN in python format (n-best list)
        :param output_gzip: gzip the output
        """
        assert len(search_py_output) > 0
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        data: List[Dict[str, List[Tuple[float, str]]]] = [
            eval(util.uopen(fn, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
            for fn in self.search_py_output
        ]
        seq_tags: List[str] = list(data[0].keys())
        seq_tags_set: Set[str] = set(seq_tags)
        assert len(seq_tags) == len(seq_tags_set), "duplicate seq tags"
        for _, d in data:
            assert set(d.keys()) == seq_tags_set, "inconsistent seq tags"

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag in seq_tags:
                data_: List[Tuple[float, str]] = []
                for d in data:
                    assert seq_tag in d
                    d_: List[Tuple[float, str]] = d[seq_tag]
                    assert isinstance(d_, list)
                    data_ += d_
                # n-best list as [(score, text), ...]
                out.write(f"{seq_tag!r}: [\n")
                for d__ in data_:
                    out.write(f"{d__!r},\n")
                out.write("],\n")
            out.write("}\n")
