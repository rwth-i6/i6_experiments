__all__ = [
    "SearchParameters",
]

import dataclasses
from dataclasses import dataclass
import typing

from sisyphus import tk

from i6_experiments.users.raissi.setups.common.util.tdp import (
    TDP,
    Float,
)

from i6_experiments.users.raissi.setups.common.data.factored_label import (
    PhoneticContext,
)

from i6_experiments.users.raissi.setups.common.decoder.config import SearchParameters


@dataclass(eq=True, frozen=True)
class LBSSearchParameters(SearchParameters):
    @classmethod
    def default_monophone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=22,
            beam_limit=500_000,
            lm_scale=4.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            prior_info=priors.with_scale(0.2),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_diphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=9.0,
            tdp_scale=0.4,
            pron_scale=2.0,
            prior_info=priors.with_scale(center=0.2, left=0.1),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )

    @classmethod
    def default_triphone(cls, *, priors: PriorInfo) -> "SearchParameters":
        return cls(
            beam=20,
            beam_limit=500_000,
            lm_scale=11.0,
            tdp_scale=0.6,
            pron_scale=2.0,
            prior_info=priors.with_scale(center=0.2, left=0.1, right=0.1),
            tdp_speech=(3.0, 0.0, "infinity", 0.0),
            tdp_silence=(0.0, 3.0, "infinity", 20.0),
            tdp_nonword=(0.0, 3.0, "infinity", 20.0),
            non_word_phonemes="[UNKNOWN]",
            we_pruning=0.5,
            we_pruning_limit=10000,
        )
