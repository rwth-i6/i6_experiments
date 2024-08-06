from dataclasses import dataclass
from enum import Enum


class FeatureType(Enum):
    samples    = "samples"
    gammatones = "gt"
    filterbanks = "fb"

    def get(self):
        return self.value


@dataclass(eq=True, frozen=True)
class FeatureInfo:
    feature_type: FeatureType
    sampling_rate: int #16000 or 8000
    is_cached: bool

    @classmethod
    def default(cls) -> "FeatureInfo":
        return FeatureInfo(
            feature_type = FeatureType.gammatones,
            sampling_rate = 16000,
            is_cached = True,
        )