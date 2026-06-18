"""
Shared 12x512 relative-positional Conformer encoder, used identically by the BEST-RQ SSL
model and the CTC finetuning model so the pretrained encoder weights transfer 1:1.

Built on the i6_models ``ConformerRelPosEncoderV1`` (reviewed as clean & trustworthy): VGG 4x
frontend + macaron rel-pos blocks, True=valid sequence mask handled end-to-end through the
frontend. ``return_layers`` lets us pull intermediate layers if ever needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV2Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV2Config
from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerRelPosBlockV1Config,
    ConformerRelPosEncoderV1Config,
    ConformerRelPosEncoderV1,
)


@dataclass
class VGGFrontendConfig:
    """Plain (py3.9-safe) VGG 4-layer frontend config.

    Stores the activation as a string so it round-trips through dict (de)serialization; ``build()``
    produces the real i6_models ``VGG4LayerActFrontendV1Config`` with the nn.Module activation.

    We deliberately do NOT subclass ``VGG4LayerActFrontendV1Config`` (the posterior_hmm trick uses
    ``@dataclass(kw_only=True)``, which needs Python >= 3.10 -- but the Sisyphus manager that builds
    the graph runs Python 3.9, so this must import cleanly there).
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Tuple[int, int]
    pool1_kernel_size: Tuple[int, int]
    pool1_stride: Tuple[int, int]
    pool2_kernel_size: Tuple[int, int]
    pool2_stride: Tuple[int, int]
    out_features: int
    activation_str: str = "ReLU"
    conv_padding: Optional[Tuple[int, int]] = None
    pool1_padding: Optional[Tuple[int, int]] = None
    pool2_padding: Optional[Tuple[int, int]] = None

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def build(self) -> VGG4LayerActFrontendV1Config:
        if self.activation_str == "ReLU":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unsupported activation {self.activation_str!r}")

        def _t(x):
            return tuple(x) if x is not None else None

        return VGG4LayerActFrontendV1Config(
            in_features=self.in_features,
            conv1_channels=self.conv1_channels,
            conv2_channels=self.conv2_channels,
            conv3_channels=self.conv3_channels,
            conv4_channels=self.conv4_channels,
            conv_kernel_size=_t(self.conv_kernel_size),
            conv_padding=_t(self.conv_padding),
            pool1_kernel_size=_t(self.pool1_kernel_size),
            pool1_stride=_t(self.pool1_stride),
            pool1_padding=_t(self.pool1_padding),
            pool2_kernel_size=_t(self.pool2_kernel_size),
            pool2_stride=_t(self.pool2_stride),
            pool2_padding=_t(self.pool2_padding),
            activation=activation,
            out_features=self.out_features,
        )


@dataclass
class ConformerPosEmbConfig(ModelConfiguration):
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float


@dataclass
class ConformerEncoderConfig:
    frontend_config: VGGFrontendConfig
    pos_emb_config: ConformerPosEmbConfig
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    mhsa_with_bias: bool
    conv_kernel_size: int
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    module_list: List[str]
    module_scales: List[float]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["frontend_config"] = VGGFrontendConfig.from_dict(d["frontend_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        return ConformerEncoderConfig(**d)


def build_conformer_encoder(cfg: ConformerEncoderConfig) -> ConformerRelPosEncoderV1:
    conformer_config = ConformerRelPosEncoderV1Config(
        num_layers=cfg.num_layers,
        frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=cfg.frontend_config.build()),
        block_cfg=ConformerRelPosBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV2Config(
                input_dim=cfg.conformer_size,
                hidden_dim=cfg.ff_dim,
                dropout=cfg.ff_dropout,
                activation=nn.functional.silu,
                dropout_broadcast_axes=cfg.dropout_broadcast_axes,
            ),
            mhsa_cfg=ConformerMHSARelPosV1Config(
                input_dim=cfg.conformer_size,
                num_att_heads=cfg.num_heads,
                att_weights_dropout=cfg.att_weights_dropout,
                with_bias=cfg.mhsa_with_bias,
                dropout=cfg.mhsa_dropout,
                dropout_broadcast_axes=cfg.dropout_broadcast_axes,
                learnable_pos_emb=cfg.pos_emb_config.learnable_pos_emb,
                rel_pos_clip=cfg.pos_emb_config.rel_pos_clip,
                with_linear_pos=cfg.pos_emb_config.with_linear_pos,
                with_pos_bias=cfg.pos_emb_config.with_pos_bias,
                separate_pos_emb_per_head=cfg.pos_emb_config.separate_pos_emb_per_head,
                pos_emb_dropout=cfg.pos_emb_config.pos_emb_dropout,
            ),
            conv_cfg=ConformerConvolutionV2Config(
                channels=cfg.conformer_size,
                kernel_size=cfg.conv_kernel_size,
                dropout=cfg.conv_dropout,
                activation=nn.functional.silu,
                norm=LayerNormNC(cfg.conformer_size),
                dropout_broadcast_axes=cfg.dropout_broadcast_axes,
            ),
            modules=cfg.module_list,
            scales=cfg.module_scales,
        ),
    )
    return ConformerRelPosEncoderV1(cfg=conformer_config)


def default_pos_emb_config() -> ConformerPosEmbConfig:
    """Learnable Shaw-style relative positional encoding (repo-proven pure-Shaw recipe)."""
    return ConformerPosEmbConfig(
        learnable_pos_emb=True,
        rel_pos_clip=16,  # required when learnable; table size 2*16+1
        with_linear_pos=False,
        with_pos_bias=False,
        separate_pos_emb_per_head=False,
        pos_emb_dropout=0.0,
    )


def default_encoder_config(
    *,
    conformer_size: int = 512,
    num_layers: int = 12,
    num_heads: int = 8,
    ff_dim: int = 2048,
    dropout: float = 0.1,
) -> ConformerEncoderConfig:
    """The settled 12x512 conformer encoder (canonical i6 LibriSpeech CTC baseline + learnable Shaw
    rel-pos). Shared verbatim by the BEST-RQ SSL model and the CTC model so weights transfer 1:1.

    ``dropout`` sets all four conformer dropouts uniformly (att/conv/ff/mhsa). Default 0.1 keeps the
    SSL-pretrain encoder config byte-identical (hash-stable); CTC runs raise it per regime (LS100
    overfit fix: scratch 0.2, finetune 0.15, LS960 stays 0.1)."""
    frontend = VGGFrontendConfig(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(2, 1),
        pool1_stride=(2, 1),
        pool1_padding=None,
        pool2_kernel_size=(2, 1),
        pool2_stride=(2, 1),
        pool2_padding=None,
        out_features=conformer_size,
        activation_str="ReLU",
    )
    return ConformerEncoderConfig(
        frontend_config=frontend,
        pos_emb_config=default_pos_emb_config(),
        conformer_size=conformer_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ff_dim=ff_dim,
        att_weights_dropout=dropout,
        conv_dropout=dropout,
        ff_dropout=dropout,
        mhsa_dropout=dropout,
        mhsa_with_bias=True,
        conv_kernel_size=31,
        dropout_broadcast_axes=None,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
    )


def sequence_mask(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """Boolean mask [B, T], True = valid (in-sequence) position. (i6 convention.)"""
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    return r[None, :] < seq_len[:, None]
