"""
Search jobs
"""

import os
from sisyphus import tk, Job, Task
from i6_core import util


class SearchTakeBest(Job):
    """
    From RETURNN beam search results, extract the best result for each sequence.
    """

    def __init__(self, search_py_output: tk.Path):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        """
        self.search_py_output = search_py_output
        self.out_best_search_results = self.output_path("best_search_results.py")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = eval(util.uopen(self.search_py_output, "r").read())
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_best_search_results.get_path())
        with util.uopen(self.out_best_search_results, "w") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                best_score, best_entry = max(entry)
                out.write("%r: %r,\n" % (seq_tag, best_entry))
            out.write("}\n")


class SearchRemoveLabel(Job):
    """
    Remove some labels from the search output, e.g. "<blank>".
    """

    def __init__(self, search_py_output: tk.Path, *, remove_label: str):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param remove_label: label to remove from the output, e.g. "<blank>"
        """
        self.search_py_output = search_py_output
        self.remove_label = remove_label
        self.out_search_results = self.output_path("search_results.py")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d = eval(util.uopen(self.search_py_output, "r").read())
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "w") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, self._filter(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, self._filter(entry)))
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t for t in tokens if t != self.remove_label]
        return " ".join(tokens)


class SearchBeamJoinScores(Job):
    """
    Expects a beam of hypotheses.
    If there are multiple hyps which are the same (e.g. this can happen after "<blank>" removal),
    it will collapse them into a single hyp with the logsumexp of the scores.
    """

    def __init__(self, search_py_output: tk.Path):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param remove_label: label to remove from the output, e.g. "<blank>"
        """
        self.search_py_output = search_py_output
        self.out_search_results = self.output_path("search_results.py")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import numpy
        neg_inf = -float("inf")

        def logsumexp(*args):
            """
            Stable log sum exp.
            """
            if all(a == neg_inf for a in args):
                return neg_inf
            a_max = max(args)
            lsp = numpy.log(sum(numpy.exp(a - a_max) for a in args))
            return a_max + lsp

        d = eval(util.uopen(self.search_py_output, "r").read())
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "w") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                # n-best list as [(score, text), ...]
                assert isinstance(entry, list)
                hyps = {}  # text -> score
                for score, text in entry:
                    if text not in hyps:
                        hyps[text] = score
                    else:
                        hyps[text] = logsumexp(hyps[text], score)
                out.write("%r: [\n" % (seq_tag,))
                for score, text in sorted([(score, text) for text, score in hyps.items()], reverse=True):
                    out.write("(%f, %r),\n" % (score, text))
                out.write("],\n")
            out.write("}\n")
