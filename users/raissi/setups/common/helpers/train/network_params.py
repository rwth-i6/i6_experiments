from sisyphus import tk

from dataclasses import dataclass, asdict
from typing import Optional

import i6_experiments.users.raissi.setups.common.helpers.train as train_helpers
from i6_experiments.users.raissi.setups.common.encoder.conformer.best_setup import Size


@dataclass
class SpecAugmentParams:
    min_reps_time: int = 0
    min_reps_feature: int = 0
    max_reps_time: int = 20
    max_len_time: int = 20
    max_reps_feature: int = 1
    max_len_feature: int = 15


@dataclass
class GeneralNetworkParams:
    l2: float
    encoder_out_len: [Size, int] = 512
    chunking: Optional[str] = None
    label_smoothing: Optional[float] = 0.0
    use_multi_task: Optional[bool] = True
    add_mlps: Optional[bool] = True
    specaug_args: Optional[dict] = None
    frame_rate_reduction_ratio_factor: Optional[int] = 1

    def __post_init__(self):
        if self.frame_rate_reduction_ratio_factor > 1:
            self.chunking = train_helpers.chunking_with_nfactor(self.chunking, self.frame_rate_reduction_ratio_factor)


# SpecAug params
default_sa_args = SpecAugmentParams()


# no chunking for full-sum
default_blstm_fullsum = GeneralNetworkParams(l2=1e-3, use_multi_task=False, add_mlps=False)
default_conformer_viterbi = GeneralNetworkParams(chunking="400:200", l2=1e-6, specaug_args=asdict(default_sa_args))
