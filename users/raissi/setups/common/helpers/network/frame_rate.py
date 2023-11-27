from dataclasses import dataclass
from typing import List, Union

from i6_experiments.users.raissi.setups.common.helpers.train.returnn_time_tag import get_default_time_tag_str

@dataclass(frozen=True, eq=True)
class FrameRateReductionRatioinfo:
    factor: Union[int, List[int]]
    time_tag_name: str

    @classmethod
    def default(cls) -> "FrameRateReductionRatioinfo":
        return FrameRateReductionRatioinfo(
            factor=1,
            time_tag_name=get_default_time_tag_str()
        )