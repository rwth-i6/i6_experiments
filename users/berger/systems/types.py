from typing import Literal, TypeVar, Union

from i6_core import returnn

EpochType = Union[int, Literal["best"]]
CheckpointType = Union[returnn.Checkpoint, returnn.PtCheckpoint]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")
