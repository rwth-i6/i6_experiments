__all__ = [
    "SearchOutputRawReplaceJob",
    "SearchRemoveLabelJob",
    "SearchCollapseRepeatedLabelsJob",
    "SearchTakeBestJob",
]

import os
from typing import Any, Optional, Union, Sequence, Set, Dict, Tuple, Iterator, List
from sisyphus import *
import i6_core.util as util
from i6_experiments.users.zhang.experiments.decoding.combine_scores import ConvertPyLiteralToNDJSONJob, iter_ndjson

def safe_next(iter):
    try:
        return next(iter)
    except StopIteration:
        return None

class SearchOutputRawReplaceJob(Job):
    """
    Converts via replacement list.

    Generalizes over :class:`SearchBPEtoWordsJob`.
    BPE-to-words::

        words = SearchOutputRawReplaceJob(bpe, [("@@ ", "")], output_gzip=True).out_search_results

    SentencePiece-to-words::

        words = SearchOutputRawReplaceJob(spm, [(" ", ""), ("▁", " ")], output_gzip=True).out_search_results

    """

    def __init__(
        self, search_py_output: Path, replacement_list: Sequence[Tuple[str, str]], *, output_gzip: bool = False
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param replacement_list: list/sequence of (old, new) pairs to perform ``s.replace(old,new)`` on the raw text
        :param output_gzip: if True, gzip the output
        """
        self.search_py_output = ConvertPyLiteralToNDJSONJob(in_file=search_py_output).out
        self.replacement_list = replacement_list
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        assert not os.path.exists(self.out_search_results.get_path())
        def _transform_text(s: str):
            for in_, out_ in self.replacement_list:
                s = s.replace(in_, out_)
            return s

        with util.uopen(self.out_search_results, "wt") as out:
            gen: Iterator[Tuple[str, List[Tuple[float, str]] | str]] = iter_ndjson(self.search_py_output)
            out.write("{\n")
            while True:
                pair = safe_next(gen)
                if pair is None:
                    break  # all exhausted
                seq_tag, entry = pair
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, _transform_text(text)))
                    out.write("],\n")
                else:
                    out.write("%r: %r,\n" % (seq_tag, _transform_text(entry)))
            out.write("}\n")


class SearchTakeBestJob(Job):
    """
    From RETURNN beam search results, extract the best result for each sequence.
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (n-best)
        :param output_gzip: if True, the output will be gzipped
        """
        self.search_py_output = ConvertPyLiteralToNDJSONJob(in_file=search_py_output).out
        self.out_best_search_results = self.output_path("best_search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        assert not os.path.exists(self.out_best_search_results.get_path())
        with util.uopen(self.out_best_search_results, "wt") as out:
            gen: Iterator[Tuple[str, List[Tuple[float, str]] | str]] = iter_ndjson(self.search_py_output)
            out.write("{\n")
            while True:
                pair = safe_next(gen)
                if pair is None:
                    break  # all exhausted
                seq_tag, entry = pair
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                best_score, best_entry = max(entry)
                out.write("%r: %r,\n" % (seq_tag, best_entry))
            out.write("}\n")


class SearchRemoveLabelJob(Job):
    """
    Remove some labels from the search output, e.g. "<blank>".
    """

    __sis_hash_exclude__ = {"output_gzip": False}

    def __init__(self, search_py_output: tk.Path, *, remove_label: Union[str, Set[str]], output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param remove_label: label(s) to remove from the output, e.g. "<blank>"
        :param output_gzip: gzip the output
        """
        self.search_py_output = ConvertPyLiteralToNDJSONJob(in_file=search_py_output).out
        if isinstance(remove_label, str):
            remove_label = {remove_label}
        assert isinstance(remove_label, set)
        self.remove_label = remove_label
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "wt") as out:
            gen: Iterator[Tuple[str, List[Tuple[float, str]]  | str]] = iter_ndjson(self.search_py_output)
            out.write("{\n")
            while True:
                pair = safe_next(gen)
                if pair is None:
                    break  # all exhausted
                seq_tag, entry = pair
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
        tokens = [t for t in tokens if t not in self.remove_label]
        return " ".join(tokens)


class SearchCollapseRepeatedLabelsJob(Job):
    """
    Collapse all repeated (white-space delimited) labels.
    """

    def __init__(self, search_py_output: tk.Path, *, output_gzip: bool = False):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param output_gzip: gzip the output
        """
        self.search_py_output = ConvertPyLiteralToNDJSONJob(in_file=search_py_output).out
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        assert not os.path.exists(self.out_search_results.get_path())
        with util.uopen(self.out_search_results, "wt") as out:
            gen: Iterator[Tuple[str, List[Tuple[float, str]]  | str]] = iter_ndjson(self.search_py_output)
            out.write("{\n")
            while True:
                pair = safe_next(gen)
                if pair is None:
                    break  # all exhausted
                seq_tag, entry = pair
                if isinstance(entry, list):
                    # n-best list as [(score, text), ...]
                    out.write("%r: [\n" % (seq_tag,))
                    for score, text in entry:
                        out.write("(%f, %r),\n" % (score, self._filter(text)))
                    out.write("],\n")
                else:
                    raise NotImplementedError
            out.write("}\n")

    def _filter(self, txt: str) -> str:
        tokens = txt.split(" ")
        tokens = [t1 for (t1, t2) in zip(tokens, [None] + tokens) if t1 != t2]
        return " ".join(tokens)



