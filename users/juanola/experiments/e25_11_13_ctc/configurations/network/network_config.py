from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.network.adapter_config import (
    AdapterConfig,
    linear_adapter_with_downsampling,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.network.decoder_config import (
    DecoderConfig,
    decoder_baseline,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.network.encoder_config import (
    EncoderConfig,
    encoder_baseline,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.network.feature_extraction_config import (
    FeatureExtractionConfig,
    feature_extraction_baseline,
)
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.protocols.has_name_protocol import (
    HasNameProtocol,
)


@dataclass(frozen=True)
class NetworkConfig(HasNameProtocol):
    """
    Network configuration base dataclass.

    Can contain default values.
    """

    feature_extraction: FeatureExtractionConfig
    encoder: EncoderConfig

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
        return f"{self.encoder.name}"


"""
param groups
"""

_MODEL_V2_KWARGS = dict(
    network_file_name="conformer_qwen_v2",
    network_class_name="SllmV2",
    training_step_file_name="train_step",
)

"""
Specific configurations set below.
"""


def network_baseline() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        **_MODEL_V2_KWARGS
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
