import os
from typing import Sequence, Tuple
from sisyphus import Job, Task, Path
import i6_core.util as util


class SearchOutputRawReplaceJob(Job):
    """
    converts via replacement list

    TODO: remove this, this should be merged into i6_core: https://github.com/rwth-i6/i6_core/pull/499
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


class TextDictToTextLinesJob(Job):
    """
    Operates on RETURNN search output (or :class:`CorpusToTextDictJob` output) and prints the values line-by-line.
    The ordering from the dict is preserved.

    TODO move this to i6_core: https://github.com/rwth-i6/i6_core/pull/501
    """

    def __init__(self, text_dict: Path, *, gzip: bool = False):
        """
        :param text_dict: a text file with a dict in python format, {seq_tag: text}
        :param gzip: if True, gzip the output
        """
        self.text_dict = text_dict
        self.out_text_lines = self.output_path("text_lines.txt" + (".gz" if gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        # nan/inf should not be needed, but avoids errors at this point and will print an error below,
        # that we don't expect an N-best list here.
        d = eval(util.uopen(self.text_dict, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> text

        with util.uopen(self.out_text_lines, "wt") as out:
            for seq_tag, entry in sorted(d.items()):
                assert isinstance(entry, str), f"expected str, got {entry!r} (type {type(entry).__name__})"
                out.write(entry + "\n")
