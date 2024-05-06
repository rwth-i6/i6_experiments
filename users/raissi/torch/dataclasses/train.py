from dataclasses import dataclass, auto

from i6_experiments.users.raissi.setups.common.data.factored_label import LabelInfo

@dataclass
class ConformerHybridConfig(ModelConfiguration):
    specaugment_cfg: specaugment.SpecaugmentConfigV1
    conformer_cfg: ConformerEncoderV1Config
    label_info: LabelInfo




@dataclass
class SpecaugmentByLengthConfigV1(ModelConfiguration):
    time_min_num_masks: int
    time_max_mask_per_n_frames: int
    time_mask_max_size: int
    freq_min_num_masks: int
    freq_max_num_masks: int
    freq_mask_max_size: int


