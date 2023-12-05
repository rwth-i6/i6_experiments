"""
Trying to make the aligner more AppTek-Like

Extended weight init code
"""

from dataclasses import dataclass


from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.config import ModuleFactoryV1, ModelConfiguration


@dataclass
class TwoLayer1DFrontendConfig(ModelConfiguration):
    """
    Attributes:
        in_features: number of input features to module
        conv1_channels: number of channels for first conv layer
        conv2_channels: number of channels for second conv layer
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv1_kernel_size: int
    conv1_stride: int
    conv2_kernel_size: int
    conv2_stride: int
    dropout: float

    def check_valid(self):
        pass

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return TwoLayer1DFrontendConfig(**d)


@dataclass
class ConformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV1Config


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return SpecaugConfig(**d)


@dataclass
class ModelConfig:
    frontend_config: TwoLayer1DFrontendConfig
    specaug_config: SpecaugConfig
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["frontend_config"] = TwoLayer1DFrontendConfig.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        return ModelConfig(**d)
