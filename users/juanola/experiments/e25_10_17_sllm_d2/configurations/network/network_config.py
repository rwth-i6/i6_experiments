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
    small_decoder_td,
    decoder_dropout_tuned_v2,
    decoder_v2_tuned,
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
        return f"{self.encoder.name}-{self.adapter.name}-{self.decoder.name}"


"""
param groups
"""

_MODEL_V1_KWARGS = dict(
    network_file_name="conformer_qwen_v1",
    network_class_name="Model",
    training_step_file_name="train_step",
)

_MODEL_V2_KWARGS = dict(
    network_file_name="conformer_qwen_v2",
    network_class_name="SllmV2",
    training_step_file_name="train_step_v2",
)

"""
Specific configurations set below.
"""


def network_baseline() -> NetworkConfig:
    return NetworkConfig(
        feature_extraction=feature_extraction_baseline(),
        encoder=encoder_baseline(),
        adapter=linear_adapter_with_downsampling(),
        decoder=decoder_baseline(),
        **_MODEL_V1_KWARGS,
    )


def network_baseline_v2() -> NetworkConfig:
    return replace(network_baseline(), **_MODEL_V2_KWARGS)


def network_SLLM_dropout() -> NetworkConfig:
    return replace(network_baseline(), decoder=decoder_dropout())


def network_SLLM_tuned() -> NetworkConfig:
    return replace(network_baseline(), decoder=decoder_v2_tuned())


def network_SLLM_tuned_dropout() -> NetworkConfig:
    warnings.warn(
        "[BUG] Doesn't use DROPOUT DROPOUT + intermediate_size too large",
        DeprecationWarning,
        stacklevel=2,
    )
    return replace(network_baseline(), decoder=decoder_dropout_tuned())


def network_SLLM_tuned_dropout_v2() -> NetworkConfig:
    return replace(network_baseline(), decoder=decoder_dropout_tuned_v2())


def network_baseline_v2_td() -> NetworkConfig:
    return replace(network_baseline_v2(), decoder=decoder_dropout_tuned_v2())


def network_baseline_v2_td_linear() -> NetworkConfig:
    return replace(network_baseline_v2(), adapter=linear_adapter(), decoder=decoder_dropout_tuned_v2())


"""
small decoders
"""


def network_SLLM_small_decoder_td() -> NetworkConfig:
    return replace(network_baseline(), decoder=small_decoder_td())


def network_small_linear_adapter() -> NetworkConfig:
    return replace(network_SLLM_small_decoder_td(), adapter=linear_adapter())


def network_linear_adapter() -> NetworkConfig:
    return replace(network_SLLM_tuned_dropout_v2(), adapter=linear_adapter())


def network_baseline_v2_td_linear_small() -> NetworkConfig:
    return replace(network_baseline_v2(), adapter=linear_adapter(), decoder=small_decoder_td())


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
