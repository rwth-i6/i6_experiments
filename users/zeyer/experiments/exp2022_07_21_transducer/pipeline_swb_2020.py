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
class State:
    task: Task

    alignment: Optional[Alignment] = None
    model: Optional[Model] = None


def from_scratch_training(state: State) -> State:
    pass


def get_alignments(state: State) -> State:
    pass


def train_extended(state: State) -> State:
    pass


def run():
    task = get_switchboard_task()
    step0 = State(task=task)
    step1 = from_scratch_training(step0)
    step2 = get_alignments(step1)
    step3 = train_extended(step2)
    step4 = train_extended(step3)
