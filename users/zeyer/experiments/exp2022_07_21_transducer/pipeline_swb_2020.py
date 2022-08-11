"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer
"""

import dataclasses
from typing import Any, Optional

from i6_experiments.users.zeyer.datasets.switchboard_2020 import get_switchboard_task
from returnn_common.datasets.interface import Task, DatasetConfig, VocabConfig


# This an alignment for one specific dataset.
# TODO Type unclear... this is a dataset as well?
Alignment = Any

AlignmentCollection = Any


Model = Any


@dataclasses.dataclass(frozen=True)
class Setup:
    task: Task

    alignment: Optional[Alignment] = None
    model: Optional[Model] = None


def from_scratch_training(setup: Setup) -> Setup:
    pass


def get_alignments(setup: Setup) -> Setup:
    pass


def train_extended(setup: Setup) -> Setup:
    pass


def run():
    task = get_switchboard_task()
    setup0 = Setup(task=task)
    setup1 = from_scratch_training(setup0)
    setup2 = get_alignments(setup1)
    setup3 = train_extended(setup2)
    setup4 = train_extended(setup3)
