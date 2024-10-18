from dataclasses import dataclass, auto
import numpy as np

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo

from i6_models.assemblies.conformer import (
    ConformerEncoderV1Config,
)
from i6_models.config import ModelConfiguration


@dataclass
class MultiTaskConfig:
    scale: int
    combination_strategy: [np.sum, np.average]


@dataclass
class SpecaugmentByLengthConfigV1(ModelConfiguration):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


@dataclass
class ConformerFactoredHybridConfig(ModelConfiguration):
    specaugment_cfg: SpecaugmentConfigV1
    conformer_cfg: ConformerEncoderV1Config
    label_info: LabelInfo
