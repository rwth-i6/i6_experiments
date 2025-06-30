import torch
from enum import Enum
from dataclasses import dataclass
from typing import Tuple, List, Dict

from ..streamable_module import StreamableModule
from ..common import Mode

class TrainMode(Enum):
    OFFLINE = 0
    STREAMING = 1
    UNIFIED = 2
    SWITCHING = 3

@dataclass
class LossEntry:
    name: str
    loss: torch.Tensor
    scale: float


class TrainStepMode:
    """
    Abstract class for a training step in either Mode.OFFLINE or Mode.STREAMING for StreamableModule's
    E.g. CTCTrainStepMode, RNNTTrainStepMode, MRNNTTrainStepMode, ...
    """
    def __init__(self):
        pass

    # NOTE: should return LossEntry
    def step(self, model: StreamableModule, data: dict, mode: Mode, scale: float) -> Tuple[Dict, int]:
        raise NotImplementedError


class TrainingStrategy:
    """
    Abstract class for training strategies involving streaming and offline forward passes.
    """
    def __init__(self, model: StreamableModule, train_step_mode: TrainStepMode) -> None:
        self.train_step_mode = train_step_mode
        self.model = model

    # NOTE: should return List[LossEntry] containing attrs (name, loss, inv_norm_factor, scale)
    def step(self, data: dict):
        raise NotImplementedError
    
class TrainOffline(TrainingStrategy):
    def __init__(self, model: StreamableModule, train_step_mode: TrainStepMode) -> None:
        super().__init__(model, train_step_mode)

    def step(self, data: dict):
        return self.train_step_mode.step(self.model, data, Mode.OFFLINE, scale=1)

class TrainStreaming(TrainingStrategy):
    def __init__(self, model: StreamableModule, train_step_mode: TrainStepMode) -> None:
        super().__init__(model, train_step_mode)

    def step(self, data: dict):
        return self.train_step_mode.step(self.model, data, Mode.STREAMING, scale=1)

class TrainUnified(TrainingStrategy):
    def __init__(self, model: StreamableModule, train_step_mode: TrainStepMode, streaming_scale: float) -> None:
        super().__init__(model, train_step_mode)
        self.streaming_scale = streaming_scale

    def step(self, data: dict):
        str_loss, num_phon = self.train_step_mode.step(self.model, data, Mode.STREAMING, scale=self.streaming_scale)
        off_loss, _ = self.train_step_mode.step(self.model, data, Mode.OFFLINE, scale=1-self.streaming_scale)
        return {**str_loss, **off_loss}, num_phon

class TrainSwitching(TrainingStrategy):
    def __init__(self, model: StreamableModule, train_step_mode: TrainStepMode, run_ctx) -> None:
        super().__init__(model, train_step_mode)
        self.run_ctx = run_ctx

    def step(self, data: dict):
        mode = Mode.STREAMING if self.run_ctx.global_step % 2 == 0 else Mode.OFFLINE
        return self.train_step_mode.step(self.model, data, mode, scale=1)
