"""
Task definition
"""

from i6_experiments.users.zeyer.datasets.base import Task, ScoreResultCollection, MeasureType


def get_switchboard_task() -> Task:
    """
    Switchboard
    """
    from i6_experiments.users.zeyer.datasets.switchboard_2020 import bpe1k, SwitchboardExternSprint
    from i6_experiments.users.zeyer.datasets.switchboard_2020.score import score

    vocab = bpe1k
    train_epoch_split = 6
    train_dataset = SwitchboardExternSprint(vocab=vocab, train_epoch_split=train_epoch_split)
    dev_dataset = SwitchboardExternSprint(vocab=vocab, main_key="dev")
    eval_datasets = {
        "hub5e_00": dev_dataset,
        "hub5e_01": SwitchboardExternSprint(vocab=vocab, main_key="hub5e_01"),
        "rt03s": SwitchboardExternSprint(vocab=vocab, main_key="rt03s"),
    }

    return Task(
        train_dataset=train_dataset,
        train_epoch_split=train_epoch_split,
        dev_dataset=dev_dataset,
        eval_datasets=eval_datasets,

        main_measure_type=MeasureType(short_name="WER%"),
        main_measure_name="hub5e_01",

        score_recog_output_func=score,
        collect_score_results_func=None,  # TODO
    )


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

        score_recog_output_func=None,  # TODO
        collect_score_results_func=None,  # TODO
    )
