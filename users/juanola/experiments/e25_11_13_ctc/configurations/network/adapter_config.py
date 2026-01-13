from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.protocols.has_name_protocol import \
    HasNameProtocol


@dataclass(frozen=True)
class AdapterConfig(HasNameProtocol):
    """
    Encoder configuration base dataclass.

    Can contain default values.
    """
    adapter_class_path: str

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass

    # add parameters if needed

    @property
    def name(self) -> str:
        return self._get_model_name()

    def _get_model_name(self) -> str:
        return self.adapter_class_path.split(".")[-1]


"""
Specific configurations set below.
"""


def linear_adapter_with_downsampling() -> AdapterConfig:
    return AdapterConfig(
        adapter_class_path="i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.networks.adapters.linear_adapter_with_concat_downsampling.LinearAdapterWithConcatDownsampling",
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
