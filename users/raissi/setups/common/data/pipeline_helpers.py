__all__ = [
    "Experiment"
    "PriorType",
    "SingleSoftmaxType",
    "TrainingCriterion",


]


from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, TypedDict

import i6_core.mm as mm
import i6_core.returnn as returnn


from i6_experiments.users.raissi.setups.common.decoder.config import (
    PriorInfo
)


class TrainingCriterion(Enum):
    """The training criterion."""

    VITERBI = "viterbi"
    FULLSUM = "fullsum"
    sMBR = "smbr"

    def __str__(self):
        return self.value


class SingleSoftmaxType(Enum):
    """The type of single softmax for joint FH."""
    TRAIN = "train"
    PRIOR = "prior"
    DECODE = "decode"

    def __str__(self):
        return self.value



class Experiment(TypedDict):
    """
    The class is used in the config files as a single experiment
    """
    name: str
    priors: Optional[PriorInfo]
    prior_job: Optional[returnn.ReturnnRasrComputePriorJobV2]
    returnn_config: Optional[returnn.ReturnnConfig]
    train_job: Optional[returnn.ReturnnRasrTrainingJob]
    align_job: Optional[mm.AlignmentJob]


class InputKey(Enum):
    """This is the dictionary key for factored hybrid system inputs."""

    BASE= "standard-system-input"
    HDF = "hdf-input"


class PriorType(Enum):
    """The type of single softmax for joint FH."""
    TRANSCRIPT = auto()
    AVERAGE = auto()
    ONTHEFLY = auto()

    def __str__(self):
        return self.value

