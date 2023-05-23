from typing import Literal, Type, TypeVar, Union
from i6_core import recognition


ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")
