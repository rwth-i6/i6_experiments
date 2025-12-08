from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class FeatureExtractionConfig:
    """
    Feature extraction configuration base dataclass.

    Can contain default values.
    """
    feature_extraction_config: Optional[Dict[str, Any]]
    sampling_rate: int
    n_mels: int
    num_enc_layers: int


"""
Specific configurations set below.
"""


def feature_extraction_baseline() -> FeatureExtractionConfig:
    return FeatureExtractionConfig(
        feature_extraction_config={
            "class": "LogMelFeatureExtractionV1",
            "win_size": 0.025,
            "hop_size": 0.01,
            "f_min": 60,
            "f_max": 7600,
            "min_amp": 1e-10,
            "num_filters": 80,
            "center": False,
        },
        sampling_rate=16_000,
        n_mels=80,
        num_enc_layers=12,
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
