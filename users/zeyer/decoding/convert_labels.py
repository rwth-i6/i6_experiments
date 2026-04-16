"""
Convert labels
"""

from __future__ import annotations

import sys
from typing import Optional, Any, Dict, Callable
from sisyphus import Job, Task, tk
from i6_core import util


def spm_merge_v3(text: str) -> str:
    """
    Merge SPM text.
    E.g. "▁This ▁is ▁a ▁test" -> "This is a test"
    """
    return text.replace(" ", "").replace("▁", " ").strip()


def spm_merge_and_lower_case_v3(text: str) -> str:
    """
    Merge SPM text and lower case.
    E.g. "▁This ▁is ▁a ▁test" -> "this is a test"
    """
    return text.replace(" ", "").replace("▁", " ").strip().lower()


class SearchOutputConvertLabelsJob(Job):
    """
    Use function to convert the search output labels.
    White-space delimited text is returned, using the target vocab.
    The input can be in any format - ``source_text_post_process`` should properly convert it if needed.
    """

    def __init__(
        self,
        search_py_output: tk.Path,
        *,
        source_text_post_process: Optional[Callable] = None,
        target_vocab: Dict[str, Any],
        returnn_root: Optional[tk.Path] = None,
        output_gzip: bool = True,
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param source_text_post_process: E.g. apply SPM/BPE merging if the input text is like that.
        :param target_vocab: the target vocab (e.g. LM labels)
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.source_text_post_process = source_text_post_process
        self.target_vocab = target_vocab
        self.returnn_root = returnn_root
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        returnn_root = util.get_returnn_root(self.returnn_root)
        sys.path.insert(0, returnn_root.get_path())

        from returnn.datasets.util.vocabulary import Vocabulary

        target_vocab = Vocabulary.create_vocab(**self.target_vocab)

        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]
                out.write(f"{seq_tag!r}: [\n")
                for score, text in entry:
                    assert isinstance(text, str)
                    if self.source_text_post_process is not None:
                        text = self.source_text_post_process(text)
                    target_vocab_labels_list = target_vocab.get_seq(text)
                    text = " ".join(target_vocab.id_to_label(idx) for idx in target_vocab_labels_list)
                    out.write(f"({score}, {text!r}),\n")
                out.write("],\n")
            out.write("}\n")


class CombineScoresAndSeparateSearchOutputJob(Job):
    """
    Combine scores and separate search output.
    """

    __sis_version__ = 2

    def __init__(
        self,
        *,
        search_py_output: tk.Path,
        scores_py_output: tk.Path,
        output_gzip: bool = True,
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best).
            We take the hypotheses from there, but ignore the scores, and instead take the scores from the scores file.
        :param scores_py_output: a scores file from RETURNN in python format (single or n-best)
            We take the scores from there, but ignore the hypotheses,
            and instead take the hypotheses from the search output file.
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.scores_py_output = scores_py_output
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        d_search = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        d_scores = eval(util.uopen(self.scores_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d_search, dict)  # seq_tag -> bpe string
        assert isinstance(d_scores, dict)  # seq_tag -> score

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d_search.items():
                assert isinstance(entry, list)
                scores_entry = d_scores.get(seq_tag)
                assert isinstance(scores_entry, list), (
                    f"Expected list of scores for seq_tag {seq_tag!r}, but got {type(scores_entry)}"
                )
                assert len(scores_entry) == len(entry), (
                    f"Expected same number of scores and hypotheses for seq_tag {seq_tag!r},"
                    f" but got {len(scores_entry)} scores and {len(entry)} hypotheses"
                )
                # n-best list as [(score, text), ...]
                out.write(f"{seq_tag!r}: [\n")
                for i, (_, text) in enumerate(entry):
                    assert isinstance(text, str)
                    score, _ = scores_entry[i]
                    assert isinstance(score, float)
                    out.write(f"({score}, {text!r}),\n")
                out.write("],\n")
            out.write("}\n")
