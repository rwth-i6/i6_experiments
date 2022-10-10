"""
Task definition
"""

from __future__ import annotations
from typing import Dict
from i6_experiments.users.zeyer.datasets.base import Task, DatasetConfig, ScoreResultCollection, MeasureType, \
    ScoreResult, RecogOutput


def get_librispeech_task() -> Task:
    """
    Librispeech
    """
    # TODO ...


def get_nltk_timit_task() -> Task:
    """
    NLTK TIMIT (small subset of TIMIT, but freely available via NLTK)
    """
    from i6_experiments.users.zeyer.datasets.nltk_timit import NltkTimit

    return Task(
        train_dataset=NltkTimit(),
        train_epoch_split=1,
        dev_dataset=NltkTimit(main_key="dev"),
        eval_datasets={"dev": NltkTimit(main_key="dev")},

        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="dev",

        score_recog_output_func=_dummy_score_recog_output_func,  # TODO
        collect_score_results_func=_dummy_collect_score_results_func,  # TODO
    )


def _dummy_score_recog_output_func(dataset: DatasetConfig, recog: RecogOutput) -> ScoreResult:
    return ScoreResult(
        dataset_name=dataset.get_main_name(),
        main_measure_value=recog.output,
        report=recog.output,
    )


def _dummy_collect_score_results_func(results: Dict[str, ScoreResult]) -> ScoreResultCollection:
    from i6_experiments.users.zeyer.utils import GroupJob
    group = GroupJob(inputs=results)
    return ScoreResultCollection(main_measure_value=group.output, output=group.output)
