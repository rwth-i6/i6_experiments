"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer
"""

import dataclasses
from typing import Any, Optional
from sisyphus import tk
from .task import Task, ScoreResultCollection, get_switchboard_task
from .model import Model, get_model_definition_from_module
from .train import train
from .recog import recog

# This an alignment for one specific dataset.
# TODO Type unclear... this is a dataset as well?
Alignment = Any

AlignmentCollection = Any


@dataclasses.dataclass(frozen=True)
class State:
    task: Task
    model: Model
    alignment: Optional[AlignmentCollection] = None


def from_scratch_training(task: Task) -> State:
    from .configs import from_scratch
    model_def = get_model_definition_from_module(from_scratch)
    train()


def get_alignments(state: State) -> State:
    pass


def train_extended(state: State) -> State:
    pass


def run():
    """run"""
    task = get_switchboard_task()

    step1 = from_scratch_training(task)
    step2 = get_alignments(step1)
    step3 = train_extended(step2)
    step4 = train_extended(step3)

    tk.register_output('step1', recog(task, step1.model).main_measure_value)
    tk.register_output('step3', recog(task, step3.model).main_measure_value)
    tk.register_output('step4', recog(task, step4.model).main_measure_value)
