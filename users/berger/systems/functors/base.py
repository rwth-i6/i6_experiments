from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List

from .. import types


class AbstractTrainFunctor(ABC, Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(
        self, train_config: types.NamedConfig[types.ConfigType], **kwargs
    ) -> types.TrainJobType:
        ...


class AbstractRecognitionFunctor(ABC, Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(
        self,
        train_job: types.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        recog_config: types.NamedConfig[types.ConfigType],
        recog_corpus: types.NamedCorpusInfo,
        **kwargs,
    ) -> List[Dict]:
        ...


class AbstractAlignmentFunctor(ABC, Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(
        self,
        train_job: types.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        align_config: types.ConfigType,
        align_corpus: types.NamedCorpusInfo,
        **kwargs,
    ) -> None:
        ...


@dataclass
class Functors(Generic[types.TrainJobType, types.ConfigType]):
    train: AbstractTrainFunctor[types.TrainJobType, types.ConfigType]
    recognize: AbstractRecognitionFunctor[types.TrainJobType, types.ConfigType]
    align: AbstractAlignmentFunctor[types.TrainJobType, types.ConfigType]
