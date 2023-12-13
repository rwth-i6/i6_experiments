from typing import Literal, Type, TypeVar, Union
from i6_core import recognition, returnn
from i6_experiments.users.berger.recipe.converse.scoring import MeetEvalJob


ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob], Type[MeetEvalJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]
TrialType = Union[int, Literal["best"]]
CheckpointType = Union[returnn.Checkpoint, returnn.PtCheckpoint]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")
