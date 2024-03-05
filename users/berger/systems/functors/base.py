from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Tuple, Union

from .. import dataclasses, types
from sisyphus import tk


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
        recog_config: dataclasses.NamedConfig[types.ConfigType],
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
        align_config: types.ConfigType,
        align_corpus: dataclasses.NamedCorpusInfo,
        **kwargs,
    ) -> Union[Dict[Tuple[float, types.EpochType], tk.Path], tk.Path]:
        pass


@dataclass
class Functors(Generic[types.TrainJobType, types.ConfigType]):
    train: TrainFunctor[types.TrainJobType, types.ConfigType]
    recognize: RecognitionFunctor[types.TrainJobType, types.ConfigType]
    align: AlignmentFunctor[types.TrainJobType, types.ConfigType]
