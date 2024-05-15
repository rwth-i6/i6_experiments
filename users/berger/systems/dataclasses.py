import copy
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Generic, Optional, Union
from i6_core import rasr, recognition, returnn
from i6_experiments.users.berger.helpers import RasrDataInput
from i6_experiments.users.berger.helpers.hdf import build_hdf_from_alignment
from . import types

from sisyphus import tk


@dataclass
class DualSpeakerConfig(Generic[types.ConfigType]):
    config_0: types.ConfigType
    config_1: Optional[types.ConfigType] = None

    def get_config_for_speaker(self, speaker_idx: int) -> types.ConfigType:
        assert speaker_idx in {0, 1}
        if speaker_idx == 0:
            return self.config_0
        if self.config_1 is not None:
            return self.config_1
        else:
            return self.config_0


DualSpeakerReturnnConfig = DualSpeakerConfig[returnn.ReturnnConfig]


@dataclass
class AlignmentData:
    alignment_cache_bundle: tk.Path
    allophone_file: tk.Path
    state_tying_file: tk.Path
    silence_phone: str = "<blank>"

    def get_hdf(self, returnn_python_exe: tk.Path, returnn_root: tk.Path) -> tk.Path:
        return build_hdf_from_alignment(
            alignment_cache=self.alignment_cache_bundle,
            allophone_file=self.allophone_file,
            state_tying_file=self.state_tying_file,
            silence_phone=self.silence_phone,
            returnn_python_exe=returnn_python_exe,
            returnn_root=returnn_root,
        )


@dataclass
class NamedConfig(Generic[types.ConfigType]):
    name: str
    config: types.ConfigType


@dataclass
class NamedTrainJob(Generic[types.TrainJobType]):
    name: str
    job: types.TrainJobType


@dataclass
class ScorerInfo:
    ref_file: Optional[tk.Path] = None
    job_type: types.ScoreJobType = recognition.ScliteJob
    score_kwargs: Dict = field(default_factory=dict)

    def get_score_job(self, ctm: tk.Path) -> types.ScoreJob:
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


class ConfigVariant(Enum):
    TRAIN = auto()
    PRIOR = auto()
    ALIGN = auto()
    RECOG = auto()


class FeatureType(Enum):
    SAMPLES = auto()
    GAMMATONE_8K = auto()
    GAMMATONE_CACHED_8K = auto()
    GAMMATONE_16K = auto()
    GAMMATONE_CACHED_16K = auto()
    CONCAT_SEC_GAMMATONE_16K = auto()
    CONCAT_MIX_GAMMATONE_16K = auto()
    CONCAT_SEC_MIX_GAMMATONE_16K = auto()
    LOGMEL_16K = auto()


@dataclass
class EncDecConfig(Generic[types.ConfigType]):
    encoder_config: types.ConfigType
    decoder_config: types.ConfigType


@dataclass
class ReturnnConfigs(Generic[types.ConfigType]):
    train_config: types.ConfigType
    prior_config: types.ConfigType = None  # type: ignore
    align_config: Union[types.ConfigType, EncDecConfig[types.ConfigType]] = None  # type: ignore
    recog_configs: Dict[str, Union[types.ConfigType, EncDecConfig[types.ConfigType]]] = None  # type: ignore

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
    RTF = "RTF"
    WER = "WER"
    SUB = "Sub"
    DEL = "Del"
    INS = "Ins"
    ERR = "#Err"
