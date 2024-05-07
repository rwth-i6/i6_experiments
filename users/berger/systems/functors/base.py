from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Union

from i6_experiments.users.berger.systems.dataclasses import AlignmentData

from .. import dataclasses, types


class TrainFunctor(Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(self, train_config: dataclasses.NamedConfig[types.ConfigType], **kwargs) -> types.TrainJobType:
        pass


class RecognitionFunctor(Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        recog_config: dataclasses.NamedConfig[Union[types.ConfigType, dataclasses.EncDecConfig[types.ConfigType]]],
        recog_corpus: dataclasses.NamedCorpusInfo,
        **kwargs,
    ) -> List[Dict]:
        pass


class AlignmentFunctor(Generic[types.TrainJobType, types.ConfigType]):
    @abstractmethod
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        align_config: Union[types.ConfigType, dataclasses.EncDecConfig[types.ConfigType]],
        align_corpus: dataclasses.NamedCorpusInfo,
        **kwargs,
    ) -> AlignmentData:
        pass


@dataclass
class Functors(Generic[types.TrainJobType, types.ConfigType]):
    train: TrainFunctor[types.TrainJobType, types.ConfigType]
    recognize: RecognitionFunctor[types.TrainJobType, types.ConfigType]
    align: AlignmentFunctor[types.TrainJobType, types.ConfigType]
