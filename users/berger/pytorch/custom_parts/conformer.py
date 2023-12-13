from __future__ import annotations
from typing import Tuple

__all__ = [
    "ConformerBlockConvFirstV1Config",
    "ConformerEncoderConvFirstV1Config",
    "ConformerBlockConvFirstV1",
    "ConformerEncoderConvFirstV1",
]

import torch
from torch import nn
from dataclasses import dataclass

from i6_models.config import ModelConfiguration, ModuleFactoryV1
import i6_experiments.users.berger.pytorch.custom_parts as custom_parts
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerConvolutionV1Config,
    ConformerPositionwiseFeedForwardV1,
    ConformerPositionwiseFeedForwardV1Config,
)


@dataclass
class ConformerBlockConvFirstV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for Conformer
        conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    rel_pos_enc_cfg: custom_parts.SelfAttRelPosEncodingV1Config
    ff_cfg: ConformerPositionwiseFeedForwardV1Config
    mhsa_cfg: custom_parts.ConformerMHSARelposV1Config
    conv_cfg: ConformerConvolutionV1Config


class ConformerBlockConvFirstV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockConvFirstV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.rel_pos_enc = custom_parts.SelfAttRelPosEncodingV1(cfg=cfg.rel_pos_enc_cfg)
        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = custom_parts.ConformerMHSARelposV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor):
        """
        :param tensor: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = tensor  #  [B, T, F]
        x = 0.5 * x + self.ff_1(x)  #  [B, T, F]
        x = x + self.conv(x)  # [B, T, F]
        x = x + self.mhsa(x, self.rel_pos_enc(x), sequence_mask)  #  [B, T, F]
        x = 0.5 * x + self.ff_2(x)  #  [B, T, F]
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x


@dataclass
class ConformerEncoderConvFirstV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockConvFirstV1Config


class ConformerEncoderConvFirstV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderConvFirstV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList(
            [ConformerBlockConvFirstV1(cfg.block_cfg) for _ in range(cfg.num_layers)]
        )

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: torch.Tensor of shape [B, T, F']

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, mask)  # [B, T, F']

        return x, mask
