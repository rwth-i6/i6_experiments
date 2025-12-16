from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.decoder_config import (
    DecoderConfig,
    decoder_baseline,
    decoder_dropout,
    decoder_dropout_tuned, small_decoder,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.encoder_config import (
    EncoderConfig,
    encoder_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.feature_extraction_config import (
    FeatureExtractionConfig,
    feature_extraction_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.protocols.has_name_protocol import HasNameProtocol


@dataclass(frozen=True)
class NetworkConfig(HasNameProtocol):
    """
    Network configuration base dataclass.

    Can contain default values.
    """

    feature_extraction: FeatureExtractionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig

    @property
    def name(self) -> str:
        return f"{self.encoder.name}-{self.decoder.name}"


"""
Specific configurations set below.
"""


def network_baseline() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        decoder=decoder_baseline(),
    )


def network_SLLM_dropout() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        decoder=decoder_dropout(),
    )


def network_SLLM_tuned_dropout() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        decoder=decoder_dropout_tuned(),
    )

def network_SLLM_small_decoder() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        decoder=small_decoder(),
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
