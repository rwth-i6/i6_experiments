from typing import Literal, Protocol, TypeVar, Union, Type
from i6_core import returnn
from sisyphus import tk

from i6_core import returnn

class ScoreJob(Protocol):
    def __init__(self, ref: tk.Path, hyp: tk.Path, *args, **kwargs) -> None: ...


ScoreJobType = Type[ScoreJob]

# ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob], Type[MeetEvalJob]]
# ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]
CheckpointType = Union[returnn.Checkpoint, returnn.PtCheckpoint]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")
