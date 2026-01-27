"""
Calculates perplexities
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Callable, Sequence, Literal, Dict, List, Tuple
from sisyphus import Job, Path, Task
from sisyphus.job_path import Variable

import i6_core.util as util

from i6_experiments.users.zeyer.datasets.task import RecogOutput, Task as DatasetsTask
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.datasets.utils.vocab import ExtractVocabLabelsJob, ExtractVocabSpecialLabelsJob

from .lm_rescoring import lm_score, ngram_score_v2
from .concat_hyps import ExtendSingleRefToHypsJob

if TYPE_CHECKING:
    from returnn_common.datasets_old_2022_10.interface import DatasetConfig


def get_lm_perplexities_for_task_evals(
    task: DatasetsTask,
    *,
    lm: ModelWithCheckpoint,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    label_level: Literal["task", "word"],
):
    """
    Returns a function that can be used in task evals to compute LM perplexity.
    """
    vocab_file = ExtractVocabLabelsJob(_get_vocab_opts_from_task(task)).out_vocab
    vocab_opts_file = ExtractVocabSpecialLabelsJob(_get_vocab_opts_from_task(task)).out_vocab_special_labels_dict

    refs = get_refs_from_task_eval_datasets(
        task, post_proc_funcs=task.recog_post_proc_funcs if label_level == "word" else ()
    )
    perplexities = {
        name: get_lm_perplexity(
            ref,
            lm=lm,
            vocab=vocab_file,
            vocab_opts_file=vocab_opts_file,
            rescore_rqmt=rescore_rqmt,
            label_post_process_funcs=task.recog_post_proc_funcs if label_level == "task" else (),
        )
        for name, ref in refs.items()
    }
    return perplexities


def _get_vocab_opts_from_task(task: DatasetsTask) -> Dict[str, Any]:
    dataset = task.dev_dataset
    extern_data_dict = dataset.get_extern_data()
    target_dict = extern_data_dict[dataset.get_default_target()]
    return target_dict["vocab"]


def get_lm_perplexity(
    ref: RecogOutput,
    *,
    lm: ModelWithCheckpoint,
    vocab: Path,
    vocab_opts_file: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    label_post_process_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
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
    for f in label_post_process_funcs:
        scored = f(scored)
    return CalcPerplexityFromScoresJob(scored.output).out_perplexity


def get_ngram_perplexities_for_task_evals(task: DatasetsTask, *, lm: Path, label_level: Literal["task", "word"]):
    """
    Returns a function that can be used in task evals to compute ngram perplexity.
    """
    refs = get_refs_from_task_eval_datasets(
        task, post_proc_funcs=task.recog_post_proc_funcs if label_level == "word" else ()
    )
    perplexities = {
        name: get_ngram_perplexity(
            ref,
            lm=lm,
            label_post_process_funcs=task.recog_post_proc_funcs if label_level == "task" else (),
        )
        for name, ref in refs.items()
    }
    return perplexities


def get_ngram_perplexity(
    ref: RecogOutput,
    *,
    lm: Path,
    rescore_rqmt: Optional[Dict[str, Any]] = None,
    label_post_process_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = (),
) -> Variable:
    """
    Calculates perplexity of the ref with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param rescore_rqmt:
    :param label_post_process_funcs:
    :return: perplexity
    """
    scored = ngram_score_single(ref, lm=lm, rescore_rqmt=rescore_rqmt)
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
    rescore_rqmt: Optional[Dict[str, Any]] = None,
) -> RecogOutput:
    """
    Scores the hyps with the LM.

    :param ref: The format is: {"<seq_tag>": "<text>", ...}.
    :param lm: language model
    :param rescore_rqmt:
    """
    ref_as_hyps = RecogOutput(ExtendSingleRefToHypsJob(ref.output).out_hyps)
    return ngram_score_v2(ref_as_hyps, lm=lm, rescore_rqmt=rescore_rqmt)


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


def get_refs_from_task_eval_datasets(
    task: DatasetsTask, *, post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = ()
) -> Dict[str, RecogOutput]:
    return {name: get_refs_from_dataset(ds, post_proc_funcs=post_proc_funcs) for name, ds in task.eval_datasets.items()}


def get_refs_from_dataset(
    dataset: DatasetConfig, *, post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = ()
) -> RecogOutput:
    from i6_experiments.users.zeyer.datasets.utils.serialize import ReturnnDatasetToTextDictJob

    ref = RecogOutput(
        output=ReturnnDatasetToTextDictJob(
            returnn_dataset=dataset.get_main_dataset(), data_key=dataset.get_default_target()
        ).out_txt
    )
    for f in post_proc_funcs:
        ref = f(ref)
    return ref
