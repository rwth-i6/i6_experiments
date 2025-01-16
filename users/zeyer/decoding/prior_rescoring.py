"""
Use a prior to rescore some recog output.
"""

from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass

from sisyphus import Job, Task, tk
from i6_core import util
from i6_experiments.users.zeyer.datasets.score_results import RecogOutput

from .rescoring import combine_scores


@dataclass
class Prior:
    """Prior"""

    file: tk.Path  # will be read via numpy.loadtxt
    type: str  # "log_prob" or "prob"
    vocab: tk.Path  # line-based, potentially gzipped


def prior_score(res: RecogOutput, *, prior: Prior) -> RecogOutput:
    """
    Use prior to score some recog output.

    :param res: previous recog output, some hyps to rescore. the score in those hyps is ignored
    :param prior:
    :param prior_scale: scale for the prior. the negative of this will be used for the weight
    :param orig_scale: scale for the original score
    """
    return RecogOutput(
        output=SearchPriorRescoreJob(
            res.output, prior=prior.file, prior_type=prior.type, vocab=prior.vocab
        ).out_search_results
    )


def prior_rescore(res: RecogOutput, *, prior: Prior, prior_scale: float, orig_scale: float = 1.0) -> RecogOutput:
    """
    Use prior to rescore some recog output.

    :param res: previous recog output, some hyps to rescore. the score in those hyps is ignored
    :param prior:
    :param prior_scale: scale for the prior. the negative of this will be used for the weight
    :param orig_scale: scale for the original score
    """
    scores = [(orig_scale, res), (-prior_scale, prior_score(res, prior=prior))]
    return combine_scores(scores)


class SearchPriorRescoreJob(Job):
    """
    Use prior to rescore some recog output.
    """

    __sis_version__ = 2

    def __init__(
        self, search_py_output: tk.Path, *, prior: tk.Path, prior_type: str, vocab: tk.Path, output_gzip: bool = True
    ):
        """
        :param search_py_output: a search output file from RETURNN in python format (single or n-best)
        :param prior:
        :param prior_type: "log_prob" or "prob"
        :param vocab: line-based, potentially gzipped
        :param output_gzip: gzip the output
        """
        self.search_py_output = search_py_output
        self.prior = prior
        assert prior_type in ["log_prob", "prob"], f"invalid prior_type {prior_type!r}"
        self.prior_type = prior_type
        self.vocab = vocab
        self.out_search_results = self.output_path("search_results.py" + (".gz" if output_gzip else ""))

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import numpy as np

        d = eval(util.uopen(self.search_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")})
        assert isinstance(d, dict)  # seq_tag -> bpe string

        vocab: List[str] = util.uopen(self.vocab, "rt").read().splitlines()
        vocab_to_idx: Dict[str, int] = {word: i for (i, word) in enumerate(vocab)}

        prior = np.loadtxt(self.prior.get_path())
        assert prior.shape == (len(vocab),), f"prior shape {prior.shape} vs vocab size {len(vocab)}"
        # The `type` is about what is stored in the file.
        # We always want it in log prob here, so we potentially need to convert it.
        if self.prior_type == "log_prob":
            pass  # already log prob
        elif self.prior_type == "prob":
            prior = np.log(prior)
        else:
            raise ValueError(f"invalid static_prior type {self.prior_type!r}")

        with util.uopen(self.out_search_results, "wt") as out:
            out.write("{\n")
            for seq_tag, entry in d.items():
                assert isinstance(entry, list)
                # n-best list as [(score, text), ...]. we ignore the input score.
                out.write(f"{seq_tag!r}: [\n")
                for _, text in entry:
                    assert isinstance(text, str)
                    scores = []
                    for label in text.split():
                        if label not in vocab_to_idx:
                            raise ValueError(f"unknown label {label!r} in seq_tag {seq_tag!r}, seq {text!r}")
                        scores.append(prior[vocab_to_idx[label]])
                    score = float(np.sum(scores))
                    out.write(f"({score}, {text!r}),\n")
                out.write("],\n")
            out.write("}\n")
