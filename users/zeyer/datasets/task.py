"""
Dataset / task interface
"""

from __future__ import annotations

import types
from typing import Dict, Callable, Sequence
import dataclasses

from returnn_common.datasets_old_2022_10.interface import DatasetConfig
from .score_results import RecogOutput, ScoreResult, ScoreResultCollection, MeasureType


@dataclasses.dataclass
class Task:
    """
    Covers the training dataset and dev/eval etc. for recognition, including how to score it.
    This goes beyond :class:`DatasetConfig`, or rather covers multiple :class:`DatasetConfig`.

    It should be possible to replace Librispeech by Switchboard. Maybe even translation tasks later.

    Note that the dataset would also already include things like feature extraction details, output labels (BPE etc).
    """

    name: str  # to differentiate between different tasks. might be used for the output dir name

    # for training
    train_dataset: DatasetConfig  # also includes cross-validation dataset for learning rate scheduling etc
    train_epoch_split: int

    # for recognition
    dev_dataset: DatasetConfig  # used to select best epoch, maybe tune LM scale or so.
    eval_datasets: Dict[str, DatasetConfig]

    main_measure_type: MeasureType  # e.g. WER%
    main_measure_name: str  # e.g. dataset name but arbitrary, just to describe the main measure value

    # E.g. for WER, CER, BLEU, ... Usually uses sclite for ASR.
    # Currently, many score funcs in use only use ``get_main_name()`` of the dataset,
    # and not really the dataset itself.
    # E.g. see librispeech _score_recog_out_v2.
    score_recog_output_func: Callable[[DatasetConfig, RecogOutput], ScoreResult]

    # e.g. for bpe_to_words or so. This is here because it depends on the type of vocab.
    recog_post_proc_funcs: Sequence[Callable[[RecogOutput], RecogOutput]] = ()

    def default_collect_score_results(self, score_results: Dict[str, ScoreResult]) -> ScoreResultCollection:
        """using main_measure_name as the main key in score_results"""
        from .score_results import join_score_results

        return join_score_results(score_results, main_measure_key=self.main_measure_name)

    collect_score_results_func: Callable[[Dict[str, ScoreResult]], ScoreResultCollection] = None

    def __post_init__(self):
        if self.collect_score_results_func is None:
            self.collect_score_results_func = self.default_collect_score_results
        elif isinstance(self.collect_score_results_func, types.MethodType) and isinstance(
            self.collect_score_results_func.__self__, Task
        ):
            # When we do dataclasses.replace to copy the task,
            # we would have the bound method bound to the wrong Task instance.
            # This is very likely unintended, so we rebind it to self.
            self.collect_score_results_func = types.MethodType(self.collect_score_results_func.__func__, self)
