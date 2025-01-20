from dataclasses import dataclass

from ....data.base import LmDataConfig
from ...common.learning_rates import ConstDecayLRConfig


@dataclass
class TrainRoutineConfig:
    train_data_config: LmDataConfig
    cv_data_config: LmDataConfig
    batch_size: int
    num_epochs: int
    lr_config: ConstDecayLRConfig
    gradient_clip: float
