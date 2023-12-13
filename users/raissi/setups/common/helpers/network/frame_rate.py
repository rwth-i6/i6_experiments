from dataclasses import dataclass
from typing import List, Optional, Union

from i6_experiments.users.raissi.setups.common.helpers.train.returnn_time_tag import (
    get_default_time_tag_str,
    get_default_time_tag_prolog,
)


@dataclass(frozen=True, eq=True)
class FrameRateReductionRatioinfo:
    factor: Union[int, List[int]]
    time_tag_name: str
    single_state_alignment: bool

    @classmethod
    def default(cls) -> "FrameRateReductionRatioinfo":
        return FrameRateReductionRatioinfo(
            factor=1, time_tag_name=get_default_time_tag_str(), single_state_alignment=False
        )

    @classmethod
    def get_time_tag_prolog_for_returnn_config(cls):
        return get_default_time_tag_prolog()
