from __future__ import annotations

__all__ = [
    "ConformerRelPosBlockV1Config",
    "ConformerRelPosEncoderV1Config",
    "ConformerRelPosBlockV1",
    "ConformerRelPosEncoderV1",
]

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Tuple, List, Optional

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV2Config,
    ConformerConvolutionV3,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.model.mhsa_rel_pos_v2 import ConformerMHSARelPosV2, ConformerMHSARelPosV2Config
from i6_models.assemblies.conformer import ConformerEncoderV2


@dataclass
class ConformerRelPosBlockV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV2
        mhsa_cfg: Configuration for ConformerMHSARelPosV2
        conv_cfg: Configuration for ConformerConvolutionV2
        modules: List of modules to use for ConformerRelPosBlockV1,
            "ff" for feed forward module, "mhsa" for multi-head self attention module, "conv" for conv module
        scales: List of scales to apply to the module outputs before the residual connection
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV2Config
    mhsa_cfg: ConformerMHSARelPosV2Config
    conv_cfg: ConformerConvolutionV2Config
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])

    def __post__init__(self):
        super().__post_init__()
        assert len(self.modules) == len(self.scales), "modules and scales must have same length"
        for module_name in self.modules:
            assert module_name in ["ff", "mhsa", "conv"], "module not supported"


class ConformerRelPosBlockV2(nn.Module):
    """
    Conformer block module, modifications compared to ConformerBlockV1:
    - uses ConfomerMHSARelPosV1 as MHSA module
    - enable constructing the block with self-defined module_list as ConformerBlockV2
    """

    def __init__(self, cfg: ConformerRelPosBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()

        modules = []
        for module_name in cfg.modules:
            if module_name == "ff":
                modules.append(ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg))
            elif module_name == "mhsa":
                modules.append(ConformerMHSARelPosV2(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionV3(model_cfg=cfg.conv_cfg))
            else:
                raise NotImplementedError

        self.module_list = nn.ModuleList(modules)
        self.scales = cfg.scales
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor, if_dropout=True) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        for scale, module in zip(self.scales, self.module_list):
            if isinstance(module, ConformerMHSARelPosV2):
                x = scale * module(x, sequence_mask, if_dropout)
            elif isinstance(module, ConformerConvolutionV3):
                x = scale * module(x, sequence_mask) + x
            else:
                x = scale * module(x) + x
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerRelPosEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerRelPosBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerRelPosBlockV1Config


class ConformerRelPosEncoderV2(nn.Module):
    """
    Modifications compared to ConformerEncoderV2:
    - supports Shaw's relative positional encoding using learnable position embeddings
      and Transformer-XL style relative PE using fixed sinusoidal or learnable position embeddings
    """

    def __init__(self, cfg: ConformerRelPosEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerRelPosBlockV2(cfg.block_cfg) for _ in range(cfg.num_layers)])
    def forward(
        self, data_tensor: torch.Tensor, /, sequence_mask: torch.Tensor, return_layers: Optional[List[int]] = None,
            if_dropout: bool = True
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :param return_layers: list of layer indices specifying which layers to return, starting from 0
        :return: (outputs, out_seq_mask)
            where outputs is a list of torch.Tensor of shape [B, T, F']
            for each of the layers in return_layers,
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """

        if return_layers is None:
            return_layers = [len(self.module_list) - 1]

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        outputs = []
        assert max(return_layers) < len(self.module_list) and min(return_layers) >= 0, (
            f"invalid layer index, should be between 0 and {len(self.module_list) - 1}"
        )

        for i in range(max(return_layers) + 1):
            x = self.module_list[i](x, sequence_mask, if_dropout)  # [B, T, F']
            if i in return_layers:
                outputs.append(x)

        return outputs, sequence_mask

