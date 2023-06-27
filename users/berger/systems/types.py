from typing import Literal, Type, TypeVar, Union
from i6_core import recognition, returnn


ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]
CheckpointType = Union[returnn.Checkpoint, returnn.PtCheckpoint]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")
