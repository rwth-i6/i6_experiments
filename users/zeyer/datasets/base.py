"""
Dataset / task interface
"""

from __future__ import annotations
from typing import Dict, Callable
import dataclasses
from sisyphus import tk
from returnn_common.datasets.interface import DatasetConfig


@dataclasses.dataclass
class Task:
    """
    Covers the training dataset and dev/eval etc. for recognition, including how to score it.
    This goes beyond :class:`DatasetConfig`, or rather covers multiple :class:`DatasetConfig`.

    It should be possible to replace Librispeech by Switchboard. Maybe even translation tasks later.

    Note that the dataset would also already include things like feature extraction details, output labels (BPE etc).
    """

    # for training
    train_dataset: DatasetConfig  # also includes cross-validation dataset for learning rate scheduling etc
    train_epoch_split: int

    # for recognition
    dev_dataset: DatasetConfig  # used to select best epoch, maybe tune LM scale or so.
    eval_datasets: Dict[str, DatasetConfig]

    main_measure_type: MeasureType  # e.g. WER%
    main_measure_name: str  # e.g. dataset name but arbitrary, just to describe the main measure value

    score_recog_output_func: Callable[[DatasetConfig, RecogOutput], ScoreResult]
    collect_score_results_func: Callable[[Dict[str, ScoreResult]], ScoreResultCollection]


@dataclasses.dataclass
class RecogOutput:
    """
    Corresponds to the target values of datasets defined by :class:`Task`
    """
    output: tk.Path


@dataclasses.dataclass
class ScoreResult:
    """
    Corresponds to one dataset. E.g. via sclite, or sth else...
    """
    dataset_name: str
    main_measure_value: tk.Path
    report: tk.Path


@dataclasses.dataclass
class ScoreResultCollection:
    """
    Intended to cover all relevant results over all eval datasets.
    """
    main_measure_value: tk.Path  # e.g. the final best WER% on test-other
    output: tk.Path  # JSON dict with all score outputs


@dataclasses.dataclass(frozen=True)
class MeasureType:
    """measure type, e.g. WER%"""
    short_name: str  # e.g. "WER%"
    lower_is_better: bool = True
