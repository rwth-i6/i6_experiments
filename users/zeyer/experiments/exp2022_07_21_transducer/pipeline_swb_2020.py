"""
Replicating the pipeline of my 2020 transducer work:
https://github.com/rwth-i6/returnn-experiments/tree/master/2020-rnn-transducer

Note: this file is loaded in different contexts:

- as a sisyphus config file. In this case, the function `py` is called.
- via the generated RETURNN configs. In this case, all the Sisyphus stuff is ignored,
  and only selected functions will run.

"""

import dataclasses
from typing import Any, Optional
from sisyphus import tk
from .task import Task, get_switchboard_task
from .model import ModelWithCheckpoint, AlignmentCollection
from .train import train
from .recog import recog
from returnn_common import nn
from returnn_common.nn.encoder.blstm_cnn_specaug import BlstmCnnSpecAugEncoder


def sis_config_main():
    """sis config function"""
    task = get_switchboard_task()

    step1 = sis_from_scratch_training(task)
    step2 = sis_get_alignments(step1)
    step3 = sis_train_extended(step2)  # use step1 model params; different to the paper
    step4 = sis_train_extended(step3)

    tk.register_output('step1', recog(task, step1.model).main_measure_value)
    tk.register_output('step3', recog(task, step3.model).main_measure_value)
    tk.register_output('step4', recog(task, step4.model).main_measure_value)


py = sis_config_main  # `py` is the default sis config function name


@dataclasses.dataclass(frozen=True)
class State:
    """current state of the pipeline"""
    task: Task  # including dataset etc
    model: ModelWithCheckpoint
    alignment: Optional[AlignmentCollection] = None


class Model(nn.Module):
    """Model definition"""

    def __init__(self):
        super(Model, self).__init__()
        self.encoder = BlstmCnnSpecAugEncoder(num_layers=6)
        # TODO pretrain...
        # TODO decoder...


def from_scratch_training(*,
                          data: nn.Data, data_spatial_dim: nn.Dim,
                          targets: nn.Data, targets_spatial_dim: nn.Dim
                          ) -> Model:
    """Function is run within RETURNN."""
    model = Model()
    # TODO...
    # TODO feed through model, define full sum loss, mark_as_loss
    # TODO pretrain epoch dependent...
    return model


def sis_from_scratch_training(task: Task) -> State:
    model = train(task=task, model=from_scratch_training)
    return State(task=task, model=model)


def sis_get_alignments(state: State) -> State:
    pass


def sis_train_extended(state: State) -> State:
    pass

