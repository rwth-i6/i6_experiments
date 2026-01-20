import dataclasses
from dataclasses import dataclass

from .decoder_config import DecoderConfig, decoder_base, decoder_small
from .feature_extraction_config import FeatureExtractionConfig, feature_extraction_baseline
from ..protocols.has_name_protocol import HasNameProtocol


@dataclass(frozen=True)
class NetworkConfig(HasNameProtocol):
    """
    Network configuration base dataclass.

    Can contain default values.
    """

    feature_extraction: FeatureExtractionConfig  # Needed for model init...
    decoder: DecoderConfig

    network_file_name: str
    network_class_name: str
    training_step_file_name: str

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass

    @property
    def name(self) -> str:
        return f"{self.decoder.name}"


"""
param groups
"""

_MODEL_V2_KWARGS = dict(
    network_file_name="lm_conformer_qwen_v2",
    network_class_name="SllmV2Lm",

    training_step_file_name="train_step",
)

"""
Specific configurations set below.
"""


def network_base() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        decoder=decoder_base(),
        **_MODEL_V2_KWARGS
    )


def network_small() -> NetworkConfig:
    return dataclasses.replace(network_base(), decoder=decoder_small())

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
