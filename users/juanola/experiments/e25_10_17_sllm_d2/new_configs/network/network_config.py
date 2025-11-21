from dataclasses import dataclass

from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.decoder_config import DecoderConfig
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.encoder_config import EncoderConfig


@dataclass
class NetworkConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """
    # TODO: this only for example
    feature_extraction_config: FeatureExtractionConfig = None
    encoder_config: EncoderConfig = None
    decoder_config: DecoderConfig = None


"""
Specific configurations set below.
"""


def get_dataset_config_v1() -> NetworkConfig:
    return NetworkConfig()
