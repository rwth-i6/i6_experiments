"""
Concatenate N-best lists
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set
from sisyphus import Job, Task, tk

import i6_core.util as util

from i6_experiments.users.zeyer.datasets.score_results import RecogOutput


def concat_hyps(search_py_output: List[tk.Path]) -> tk.Path:
    """
    Concatenate N-best lists

    :param search_py_output: search output file from RETURNN in python format (n-best list)
    :param output_gzip: gzip the output
    :return: path to concatenated output
    """
    job = SearchConcatHypsJob(search_py_output)
    return job.out_search_results


def concat_hyps_recog_out(search_py_output: List[RecogOutput]) -> RecogOutput:
    """
    Concatenate N-best lists
    """
    paths = [r.output for r in search_py_output]
    out_path = concat_hyps(paths)
    return RecogOutput(output=out_path)


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
        for d in data:
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


class ExtendSingleRefToHypsJob(Job):
    """
    Takes a number of files, each with some N-best list of hyps, and concatenates them,
    such that you get a single M-best list, M=sum(each N).
    """

    __sis_version__ = 2

    def __init__(self, ref_py_output: tk.Path, *, output_gzip: bool = True, score: float = 0.0):
        """
        :param ref_py_output: search output file from RETURNN in python format (single hyp per seq)
        :param output_gzip: gzip the output
        """
        self.ref_py_output = ref_py_output
        self.score = score
        self.out_hyps = self.output_path("hyps.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        data: Dict[str, str] = eval(
            util.uopen(self.ref_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")}
        )

        with util.uopen(self.out_hyps, "wt") as out:
            out.write("{\n")
            for seq_tag, text in data.items():
                assert isinstance(text, str)
                # n-best list as [(score, text), ...]
                out.write(f"{seq_tag!r}: [({self.score}, {text})],\n")
            out.write("}\n")
