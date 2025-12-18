import warnings
from dataclasses import dataclass, replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.adapter_config import (
    AdapterConfig,
    linear_adapter_with_downsampling,
    linear_adapter,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.decoder_config import (
    DecoderConfig,
    decoder_baseline,
    decoder_dropout,
    decoder_dropout_tuned,
    small_decoder, decoder_dropout_tuned_v2,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.encoder_config import (
    EncoderConfig,
    encoder_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.feature_extraction_config import (
    FeatureExtractionConfig,
    feature_extraction_baseline,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.protocols.has_name_protocol import (
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
    adapter: AdapterConfig
    decoder: DecoderConfig

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass

    @property
    def name(self) -> str:
        return f"{self.encoder.name}-{self.adapter.name}-{self.decoder.name}"


"""
Specific configurations set below.
"""


def network_baseline() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=decoder_baseline(),
    )


def network_SLLM_dropout() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=decoder_dropout(),
    )


def network_SLLM_tuned_dropout() -> NetworkConfig:
    warnings.warn(
        "[BUG] Doesn't use DROPOUT DROPOUT + intermediate_size too large",
        DeprecationWarning,
        stacklevel=2,
    )
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=decoder_dropout_tuned(),
    )

def network_SLLM_tuned_dropout_v2() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=decoder_dropout_tuned_v2(),
    )


def network_SLLM_small_decoder() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=small_decoder(),
    )


def network_linear_adapter() -> NetworkConfig:
    return replace(network_SLLM_small_decoder(), adapter=linear_adapter())


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
