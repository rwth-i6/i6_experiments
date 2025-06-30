import torch
from torch import nn
from typing import Tuple

from .train_handler import TrainStepMode, LossEntry, List, Dict
from ..streamable_module import StreamableModule
from ..common import Mode


# TODO
class CTCTrainStepMode(TrainStepMode):
    pass