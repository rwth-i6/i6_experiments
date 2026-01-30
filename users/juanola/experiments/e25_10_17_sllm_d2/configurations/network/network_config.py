import warnings
from dataclasses import dataclass, replace
from typing import Optional

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

    # frozen layers
    freeze_encoder_ranges: Optional[list[tuple[int, int]]] = None  # All inclusive
    freeze_decoder_ranges: Optional[list[tuple[int, int]]] = None  # All inclusive

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        for ranges in [self.freeze_encoder_ranges, self.freeze_decoder_ranges]:
            if ranges is not None:
                for t in ranges:
                    assert t is not None, "Ranges must be defined."

                assert ranges[0][0] >= 1, f"Ranges must start at least at epoch 1."
                prev = -1
                for (a,b) in ranges:
                    assert a <= b, f"Encoder freeze range ({a}, {b}) where {a} > {b}"
                    assert prev < a, f"Range must be ordered and non overlapping."
                    prev = b

    def frozen_encoder_from_the_start(self):
        if self.freeze_encoder_ranges is None:
            return False
        else:
            return self.freeze_encoder_ranges[0][0] == 1

    def frozen_decoder_from_the_start(self):
        if self.freeze_decoder_ranges is None:
            return False
        else:
            return self.freeze_decoder_ranges[0][0] == 1

    def get_frozen_encoder_epochs(self) -> list[int]:
        """
        Returns a sorted list of all epochs where the encoder should be frozen.
        """
        if self.freeze_encoder_ranges is None:
            return []

        frozen_epochs = set()
        for start, end in self.freeze_encoder_ranges:
            frozen_epochs.update(range(start, end + 1))  # +1 because range is exclusive at end

        return sorted(frozen_epochs)

    def get_frozen_decoder_epochs(self) -> list[int]:
        """
        Returns a sorted list of all epochs where the decoder should be frozen.
        """
        if self.freeze_decoder_ranges is None:
            return []

        frozen_epochs = set()
        for start, end in self.freeze_decoder_ranges:
            frozen_epochs.update(range(start, end + 1))  # +1 because range is exclusive at end

        return sorted(frozen_epochs)

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

def network_baseline_v2_td_small() -> NetworkConfig:
    return replace(network_baseline_v2(), decoder=small_decoder_td())


"""
frozen
"""

def network_with_frozen_layers(network: NetworkConfig, encoder_epochs: Optional[int] = None, decoder_epochs: Optional[int] = None) -> NetworkConfig:
    freeze_encoder_ranges = [(1, encoder_epochs)] if encoder_epochs is not None else None
    freeze_decoder_ranges = [(1, decoder_epochs)] if decoder_epochs is not None else None
    return replace(
        network,
        freeze_encoder_ranges=freeze_encoder_ranges,
        freeze_decoder_ranges=freeze_decoder_ranges,
    )


def network_baseline_v2_td_frozen_n_first_epochs(
        encoder_epochs: Optional[int] = None, decoder_epochs: Optional[int] = None
) -> NetworkConfig:
    freeze_encoder_ranges = [(1, encoder_epochs)] if encoder_epochs is not None else None
    freeze_decoder_ranges = [(1, decoder_epochs)] if decoder_epochs is not None else None
    return replace(
        network_baseline_v2_td(),
        freeze_encoder_ranges=freeze_encoder_ranges,
        freeze_decoder_ranges=freeze_decoder_ranges,
    )




# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
