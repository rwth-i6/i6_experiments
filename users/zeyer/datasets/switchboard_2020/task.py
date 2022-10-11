"""
Defines the task
"""

from __future__ import annotations
from typing import Dict
from i6_core.returnn.search import SearchBPEtoWordsJob
from ..task import Task, MeasureType, RecogOutput, ScoreResult, ScoreResultCollection


def get_switchboard_task_bpe1k() -> Task:
    """
    Switchboard
    """
    from . import bpe1k, bpe1k_with_unk, SwitchboardExternSprint
    from .score import score

    vocab = bpe1k
    train_epoch_split = 6
    train_dataset = SwitchboardExternSprint(vocab=vocab, train_epoch_split=train_epoch_split)
    dev_dataset = SwitchboardExternSprint(vocab=bpe1k_with_unk, main_key="dev")
    eval_datasets = {
        "hub5e_00": dev_dataset,
        "hub5e_01": SwitchboardExternSprint(vocab=bpe1k_with_unk, main_key="hub5e_01"),
        "rt03s": SwitchboardExternSprint(vocab=bpe1k_with_unk, main_key="rt03s"),
    }

    return Task(
        name="swb_bpe1k",
        train_dataset=train_dataset,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,

        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="hub5e_01",

        score_recog_output_func=score,
        collect_score_results_func=_dummy_collect_score_results_func,  # TODO
        recog_post_proc_funcs=[_bpe_to_words],
    )


def _bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    words = SearchBPEtoWordsJob(bpe.output).out_word_search_results
    return RecogOutput(output=words)


def _dummy_collect_score_results_func(results: Dict[str, ScoreResult]) -> ScoreResultCollection:
    from i6_experiments.users.zeyer.utils import GroupJob
    group = GroupJob(inputs=results)
    return ScoreResultCollection(main_measure_value=group.output, output=group.output)
