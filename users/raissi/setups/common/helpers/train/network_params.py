from sisyphus import tk

from dataclasses import dataclass, asdict
from typing import Optional, List

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
    auxilary_loss_layers: Optional[List] = [6],
    specaug_args: Optional[dict] = None
    frame_rate_reduction_ratio_factor: Optional[int] = 1

    def __post_init__(self):
        if self.frame_rate_reduction_ratio_factor > 1 and self.chunking is not None:
            self.chunking = train_helpers.chunking_with_nfactor(self.chunking, self.frame_rate_reduction_ratio_factor)


# SpecAug params
default_sa_args = SpecAugmentParams()

################
# BLSTM
################
# no chunking for full-sum
default_blstm_fullsum = GeneralNetworkParams(l2=1e-4, use_multi_task=False, add_mlps=False)

################
# Conformer
################
default_conformer_viterbi = GeneralNetworkParams(chunking="400:200", l2=1e-6, specaug_args=asdict(default_sa_args))

frameshift40_conformer_viterbi = GeneralNetworkParams(
    l2=1e-6, chunking="400:200", specaug_args=asdict(default_sa_args), frame_rate_reduction_ratio_factor=4
)

frameshift40_conformer_fullsum_mix = GeneralNetworkParams(
    l2=5e-6, specaug_args=asdict(default_sa_args), frame_rate_reduction_ratio_factor=4
)

frameshift40_conformer_fullsum = GeneralNetworkParams(
    l2=1e-6, specaug_args=asdict(default_sa_args), frame_rate_reduction_ratio_factor=4
)

#Comaprison with transducer
frameshift40_conformer_viterbi_zhou = GeneralNetworkParams(
    l2=5e-6, chunking="256:128", frame_rate_reduction_ratio_factor=4, auxilary_loss_layers=[6, 12]
)
frameshift40_conformer_viterbi_mix_48 = GeneralNetworkParams(
    l2=5e-6, chunking="256:128", frame_rate_reduction_ratio_factor=4, auxilary_loss_layers=[4,8]
)

frameshift40_conformer_viterbi_mix_369 = GeneralNetworkParams(
    l2=5e-6, chunking="256:128", frame_rate_reduction_ratio_factor=4, auxilary_loss_layers=[3,6,9]
)

frameshift40_conformer_viterbi_mix_base = GeneralNetworkParams(
    l2=5e-6, chunking="256:128", frame_rate_reduction_ratio_factor=4, auxilary_loss_layers=[6]
)

