import os
from typing import Sequence, Tuple
from sisyphus import Job, Task, Path
import i6_core.util as util


class SearchOutputRawReplaceJob(Job):
    """
    converts via replacement list
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(
        self, search_py_output: Path, replacement_list: Sequence[Tuple[str, str]], *, output_gzip: bool = False
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param replacement_list: list/sequence of (old, new) pairs to perform ``s.replace(old,new)`` on the raw text
        :param output_gzip: if True, gzip the output
        """
        self.search_py_output = search_py_output
        self.replacement_list = replacement_list
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string
        assert not os.path.exists(self.out_search_results.get_path())

        def _transform_text(s: str):
            for in_, out_ in self.replacement_list:
                s = s.replace(in_, out_)
            return s

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in sorted(d.items()):
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, _transform_text(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, _transform_text(entry)))
            out.write("}\n")
