import copy
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Generic, Literal, Optional, Type, TypeVar, Union
from i6_core import rasr, recognition
from i6_experiments.common.setups.rasr.util.rasr import RasrDataInput

from sisyphus import tk

ScoreJobType = Union[Type[recognition.ScliteJob], Type[recognition.Hub5ScoreJob]]
ScoreJob = Union[recognition.ScliteJob, recognition.Hub5ScoreJob]
EpochType = Union[int, Literal["best"]]

ConfigType = TypeVar("ConfigType")
TrainJobType = TypeVar("TrainJobType")


@dataclass
class NamedConfig(Generic[ConfigType]):
    name: str
    config: ConfigType


@dataclass
class NamedTrainJob(Generic[TrainJobType]):
    name: str
    job: TrainJobType


@dataclass
class ScorerInfo:
    ref_file: Optional[tk.Path] = None
    job_type: ScoreJobType = recognition.ScliteJob
    score_kwargs: Dict = field(default_factory=dict)

    def get_score_job(self, ctm: tk.Path) -> ScoreJob:
        assert self.ref_file is not None
        return self.job_type(hyp=ctm, ref=self.ref_file, **self.score_kwargs)


@dataclass
class CorpusInfo:
    data: RasrDataInput
    crp: rasr.CommonRasrParameters
    scorer: Optional[ScorerInfo] = None


@dataclass
class NamedCorpusInfo:
    name: str
    corpus_info: CorpusInfo


@dataclass
class ReturnnConfigs(Generic[ConfigType]):
    train_config: ConfigType
    prior_config: ConfigType = None  # type: ignore
    align_config: ConfigType = None  # type: ignore
    recog_configs: Dict[str, ConfigType] = None  # type: ignore

    def __post_init__(self):
        if self.prior_config is None:
            self.prior_config = copy.deepcopy(self.train_config)
        if self.recog_configs is None:
            self.recog_configs = {"recog": copy.deepcopy(self.prior_config)}
        if self.align_config is None:
            self.align_config = copy.deepcopy(next(iter(self.recog_configs.values())))


class SummaryKey(Enum):
    TRAIN_NAME = "Train name"
    RECOG_NAME = "Recog name"
    CORPUS = "Corpus"
    EPOCH = "Epoch"
    TRIAL = "Trial"
    PRON = "Pron"
    PRIOR = "Prior"
    LM = "Lm"
    WER = "WER"
    SUB = "Sub"
    DEL = "Del"
    INS = "Ins"
    ERR = "#Err"
