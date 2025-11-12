from dataclasses import dataclass
from typing import Dict

from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    cv: Dataset
    devtrain: Dataset
    datastreams: Dict[str, Datastream]
