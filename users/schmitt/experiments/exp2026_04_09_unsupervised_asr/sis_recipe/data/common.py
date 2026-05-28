from typing import Dict
from dataclasses import dataclass

from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    eval_datasets: Dict[str, Dataset]
    datastreams: Dict[str, Datastream]
