"""
Defines the task
"""

from __future__ import annotations
from typing import Dict
from i6_core.returnn.search import SearchBPEtoWordsJob
from ..task import Task
from ..score_results import RecogOutput, ScoreResult, ScoreResultCollection, MeasureType


def get_switchboard_task_bpe1k(*, bpe_sample: float = 0.) -> Task:
    """
    Switchboard
    """
    from . import bpe1k, bpe1k_with_unk, SwitchboardExternSprintOld
    from .score import score

    vocab = bpe1k
    if bpe_sample:
        assert not vocab.other_opts  # not expected here
        vocab = vocab.copy(other_opts={"class": "SamplingBytePairEncoding", "breadth_prob": bpe_sample})
    train_epoch_split = 6
    train_dataset = SwitchboardExternSprintOld(vocab=vocab, train_epoch_split=train_epoch_split)
    dev_dataset = SwitchboardExternSprintOld(vocab=bpe1k_with_unk, main_key="dev")
    eval_datasets = {
        "hub5e_00": dev_dataset,
        "hub5e_01": SwitchboardExternSprintOld(vocab=bpe1k_with_unk, main_key="hub5e_01"),
        "rt03s": SwitchboardExternSprintOld(vocab=bpe1k_with_unk, main_key="rt03s"),
    }

    return Task(
        name="swb_bpe1k",
        train_dataset=train_dataset,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,

        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="hub5e_00",

        score_recog_output_func=score,
        recog_post_proc_funcs=[_bpe_to_words],
    )


def _bpe_to_words(bpe: RecogOutput) -> RecogOutput:
    """BPE to words"""
    words = SearchBPEtoWordsJob(bpe.output).out_word_search_results
    return RecogOutput(output=words)


def _compare_old_new():
    # Now that we have the new datasets interface, check that we get the same hash.
    from sisyphus.hash import short_hash
    task = get_switchboard_task_bpe1k()
    # dataset in training
    config = dict(
        default_input=task.train_dataset.get_default_input(),
        target=task.train_dataset.get_default_target(),
        train=task.train_dataset.get_train_dataset(),
        eval_datasets=task.train_dataset.get_eval_datasets(),
    )
    print("train dataset hash:", short_hash(config))
    # datasets in recognition
    for name, dataset in task.eval_datasets.items():
        # dataset
        config = dict(
            default_input=dataset.get_default_input(),
            target=dataset.get_default_target(),
            dev=dataset.get_main_dataset(),
        )
        print(f"eval {name} dataset hash:", short_hash(config))
    # train dataset hash: jqGQaAMPnEm6
    # eval hub5e_00 dataset hash: RzkAWyRoWjqa
    # eval hub5e_01 dataset hash: Y76oLHItr2Pt
    # eval rt03s dataset hash: qgqYRU0DGiw1

    # TODO now use SwitchboardExternSprint instead.
    # TODO how to handle extern_data?
    # TODO where/how to keep default target, default input?


py = _compare_old_new
