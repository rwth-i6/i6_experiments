from dataclasses import dataclass
from typing import Dict, Generic, List

from .. import dataclasses
from .. import types


class TrainFunctor(Generic[types.TrainJobType, types.ConfigType]):
    def __call__(
        self, train_config: dataclasses.NamedConfig[types.ConfigType], **kwargs
    ) -> types.TrainJobType:
        raise NotImplementedError


class RecognitionFunctor(Generic[types.TrainJobType, types.ConfigType]):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        recog_config: dataclasses.NamedConfig[types.ConfigType],
        recog_corpus: dataclasses.NamedCorpusInfo,
        **kwargs,
    ) -> List[Dict]:
        raise NotImplementedError


class AlignmentFunctor(Generic[types.TrainJobType, types.ConfigType]):
    def __call__(
        self,
        train_job: dataclasses.NamedTrainJob[types.TrainJobType],
        prior_config: types.ConfigType,
        align_config: types.ConfigType,
        align_corpus: dataclasses.NamedCorpusInfo,
        **kwargs,
    ) -> None:
        raise NotImplementedError


@dataclass
class Functors(Generic[types.TrainJobType, types.ConfigType]):
    train: TrainFunctor[types.TrainJobType, types.ConfigType]
    recognize: RecognitionFunctor[types.TrainJobType, types.ConfigType]
    align: AlignmentFunctor[types.TrainJobType, types.ConfigType]
