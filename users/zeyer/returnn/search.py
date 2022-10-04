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
            for seq_tag, entry in sorted(d.items()):
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
