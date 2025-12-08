from dataclasses import dataclass

from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.decoder_config import DecoderConfig, \
    decoder_baseline
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.encoder_config import EncoderConfig, \
    encoder_baseline
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.feature_extraction_config import \
    FeatureExtractionConfig, feature_extraction_baseline


@dataclass(frozen=True)
class NetworkConfig:
    """
    Network configuration base dataclass.

    Can contain default values.
    """
    feature_extraction: FeatureExtractionConfig
    encoder: EncoderConfig
    decoder: DecoderConfig


"""
Specific configurations set below.
"""


def network_v1() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        decoder=decoder_baseline(),
    )


def network_v2() -> NetworkConfig:
    raise NotImplementedError

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)