"""
Calculates perplexities
"""

from typing import Optional, Any, Callable, Sequence, Dict, List, Tuple
from sisyphus import Job, Path, Task
from sisyphus.job_path import Variable

import i6_core.util as util

from i6_experiments.users.zeyer.datasets.task import RecogOutput
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint

from .lm_rescoring import lm_score, ngram_score
from .concat_hyps import ExtendSingleRefToHypsJob


def get_lm_perplexity(
    ref: RecogOutput,
    *,
    lm: ModelWithCheckpoint,
    vocab: Path,
    vocab_opts_file: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    label_post_process_funcs: Optional[Sequence[Callable]] = None,
) -> Variable:
    """
    Calculates perplexity of the ref with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    :param label_post_process_funcs:
    :return: perplexity
    """
    scored = lm_score_single(ref, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=rescore_rqmt)
    if label_post_process_funcs is not None:
        for f in label_post_process_funcs:
            scored = f(scored)
    return CalcPerplexityFromScoresJob(scored.output).out_perplexity


def get_ngram_perplexity(
    ref: RecogOutput,
    *,
    lm: Path,
    vocab: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    label_post_process_funcs: Optional[Sequence[Callable]] = None,
) -> Variable:
    """
    Calculates perplexity of the ref with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param rescore_rqmt:
    :param label_post_process_funcs:
    :return: perplexity
    """
    scored = ngram_score_single(ref, lm=lm, vocab=vocab, rescore_rqmt=rescore_rqmt)
    if label_post_process_funcs is not None:
        for f in label_post_process_funcs:
            scored = f(scored)
    return CalcPerplexityFromScoresJob(scored.output).out_perplexity


def lm_score_single(
    ref: RecogOutput,
    *,
    lm: ModelWithCheckpoint,
    vocab: Path,
    vocab_opts_file: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param vocab_opts_file: for LM labels. contains info about EOS, BOS, etc
    :param rescore_rqmt:
    """
    ref_as_hyps = RecogOutput(ExtendSingleRefToHypsJob(ref.output).out_hyps)
    return lm_score(ref_as_hyps, lm=lm, vocab=vocab, vocab_opts_file=vocab_opts_file, rescore_rqmt=rescore_rqmt)


def ngram_score_single(
    ref: RecogOutput,
    *,
    lm: Path,
    vocab: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param vocab: labels (line-based, maybe gzipped)
    :param rescore_rqmt:
    """
    ref_as_hyps = RecogOutput(ExtendSingleRefToHypsJob(ref.output).out_hyps)
    return ngram_score(ref_as_hyps, lm=lm, vocab=vocab, rescore_rqmt=rescore_rqmt)


class CalcPerplexityFromScoresJob(Job):
    """
    Calculates perplexities from scores file
    """

    def __init__(self, scores_py_output: Path):
        """
        :param scores_py_output: Path to scores.py output. list of (score, hyp) tuples per line,
            but only take the first hyp per seq.
        """
        self.scores_py_output = scores_py_output

        self.out_total_score = self.output_var("total_score")
        self.num_labels = self.output_var("num_labels")
        self.out_perplexity = self.output_var("perplexity")

    def tasks(self):
        """task"""
        yield Task("run", mini_task=True)

    def run(self):
        import numpy as np

        data: Dict[str, List[Tuple[float, str]]] = eval(
            util.uopen(self.scores_py_output, "rt").read(), {"nan": float("nan"), "inf": float("inf")}
        )

        scores = []
        num_labels = 0
        for _, scores_hyps in data.items():
            assert isinstance(scores_hyps, list) and len(scores_hyps) > 0
            score, text = scores_hyps[0]
            assert isinstance(score, float) and isinstance(text, str)
            scores.append(score)
            num_labels += len(text.split()) + 1  # +1 for EOS

        total_score = np.array(scores).sum()
        perplexity = np.exp(-total_score / num_labels)

        self.out_total_score.set(float(total_score))
        self.num_labels.set(num_labels)
        self.out_perplexity.set(float(perplexity))
