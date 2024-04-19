__all__ = [
    "TEDSearchParameters",
]

import dataclasses
from dataclasses import dataclass
import typing

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.util.tdp import (
    TDP,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    PhoneticContext,
)

from i6_experiments.users.raissi.setups.common.decoder.config import (
    SearchParameters,
    PriorInfo,
)


@dataclass(eq=True, frozen=True)
class TEDSearchParameters(SearchParameters):
    @classmethod
    def default_monophone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "TEDSearchParameters":
        return cls(
            beam=22,
            beam_limit=500_000,
            lm_scale=1.0 if frame_rate > 1 else 4.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(0.2),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_diphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "TEDSearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=1.8 if frame_rate > 1 else 6.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(center=0.2, left=0.1),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_triphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "TEDSearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=11.0,
            tdp_scale=0.6,
            prior_info=priors.with_scale(center=0.2, left=0.1, right=0.1),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_joint_diphone(cls, *, priors: PriorInfo, frame_rate: int = 1) -> "TEDSearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=1.8 if frame_rate > 1 else 6.0,
            tdp_scale=0.1 if frame_rate > 1 else 0.4,
            prior_info=priors.with_scale(diphone=0.4),
            tdp_speech=(8.0, 0.0, "infinity", 0.0) if frame_rate > 1 else (3.0, 0.0, "infinity", 0.0),
            tdp_silence=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            tdp_non_word=(11.0, 0.0, "infinity", 20.0) if frame_rate > 1 else (0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )
