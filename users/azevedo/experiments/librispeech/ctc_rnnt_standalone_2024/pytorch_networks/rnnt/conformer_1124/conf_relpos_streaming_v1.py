from __future__ import annotations

__all__ = [
    "ConformerRelPosBlockV1Config",
    "ConformerRelPosEncoderV1Config",
    "ConformerRelPosBlockV1",
    "ConformerRelPosEncoderV1",
    "ConformerRelPosBlockV1COV1Config",
]

import math
import torch
from torch import nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional

from ..conformer_0924.conf_lah_carryover_v2 import (
    ConformerBlockV3COV1,
    ConformerEncoderCOV2
)

from i6_models.util import compat
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import (
    ConformerConvolutionV2,
    ConformerConvolutionV2Config,
    ConformerMHSARelPosV1,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from i6_models.assemblies.conformer.conformer_rel_pos_v1 import (
    ConformerRelPosBlockV1Config,
    ConformerRelPosEncoderV1Config,

    ConformerRelPosBlockV1,
    ConformerRelPosEncoderV1,
)


class ConformerMHSARelPosV2(ConformerMHSARelPosV1):
    def __init__(self, cfg: ConformerMHSARelPosV1Config):
        super().__init__(cfg)

    def forward(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside
        :param attn_mask: bool mask of shape (T, T)
        """
        output_tensor = self.layernorm(input_tensor)  # [B, T, F]

        time_dim_size = output_tensor.shape[1]
        batch_dim_size = output_tensor.shape[0]

        # attention mask
        # T: query seq. length, T' key/value seg length; T = T' if same input tensor

        inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T']
        inv_sequence_mask = inv_sequence_mask.unsqueeze(1)  # [B, 1, T']
        if attn_mask is not None:
            inv_attn_mask = compat.logical_not(attn_mask)
            inv_attn_mask = inv_attn_mask.unsqueeze(0)  # [1, T', T']

        total_mask = inv_sequence_mask
        if attn_mask is not None:
            total_mask = total_mask + inv_attn_mask
        total_mask = total_mask.unsqueeze(1)  # [B, 1, T', T']

        # query, key and value sequences
        query_seq, key_seq, value_seq = self.qkv_proj(output_tensor).chunk(3, dim=-1)  # [B, T, #heads * F']
        q = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']
        k = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']

        if self.learnable_pos_emb:
            pos_seq_q = torch.arange(time_dim_size, device=input_tensor.device)
            pos_seq_k = torch.arange(time_dim_size, device=input_tensor.device)

            distance_mat = pos_seq_k[None, :] - pos_seq_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.rel_pos_clip, self.rel_pos_clip)

            final_mat = distance_mat_clipped + self.rel_pos_clip

            rel_pos_embeddings = self.rel_pos_embeddings[final_mat]  # [T, T', pos_emb_dim]
        else:
            rel_pos_embeddings = self._sinusoidal_pe(
                torch.arange(time_dim_size - 1, -time_dim_size, -1, device=input_tensor.device, dtype=torch.float32),
                self.pos_emb_dim,
            ).view(
                1, 2 * time_dim_size - 1, self.pos_emb_dim
            )  # [1, T+T'-1, pos_emb_dim]

        # dropout relative positional embeddings
        rel_pos_embeddings = self.pos_emb_dropout(
            rel_pos_embeddings
        )  # [T, T', pos_emb_dim] or [1, T+T'-1, pos_emb_dim]
        rel_pos_embeddings = rel_pos_embeddings.unsqueeze(2)  # [T, T', 1, pos_emb_dim] or [1, T+T'-1, 1, pos_emb_dim]

        # linear transformation or identity
        rel_pos_embeddings = self.linear_pos(rel_pos_embeddings)  # [T, T', 1, F'|F] or [1, T+T'-1, 1, F'|F]

        if self.separate_pos_emb_per_head:
            rel_pos_embeddings = rel_pos_embeddings.squeeze(2).reshape(
                *rel_pos_embeddings.shape[:2], -1, self.embed_dim_per_head
            )  # [T, T', #heads, F'] or [1, T+T'-1, #heads, F']

        q_with_bias_u = q + self.pos_bias_u if self.with_pos_bias else q  # [B, T, #heads, F']
        q_with_bias_v = q + self.pos_bias_v if self.with_pos_bias else q

        # attention matrix a and c
        attn_ac = torch.einsum("bihf, bjhf -> bhij", q_with_bias_u, k)  # [B, #heads, T, T']

        # attention matrix b and d
        attn_bd = torch.einsum(
            "bihf, ijhf -> bhij", q_with_bias_v, rel_pos_embeddings
        )  # [B, #heads, T, T'] or [B, #heads, T, T+T'+1]

        if not self.learnable_pos_emb:
            attn_bd = self._rel_shift_bhij(attn_bd, k_len=time_dim_size)  # [B, #heads, T, T']

        attn = attn_ac + attn_bd  # [B, #heads, T, T']
        attn_scaled = attn * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # NOTE: mask applied with masked_fill instead of addition for stable grads for zero-rows
        attn_scaled = attn_scaled.masked_fill(total_mask, float("-inf"))
        # softmax and dropout
        attn_output_weights = self.att_weights_dropout(F.softmax(attn_scaled, dim=-1))  # [B, #heads, T, T']

        attn_output_weights = attn_output_weights.masked_fill(total_mask, 0.0)

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']
        attn_output = torch.einsum("bhij, bjhf -> bihf", attn_output_weights, v).reshape(
            batch_dim_size, -1, self.embed_dim
        )

        output_tensor = self.out_proj(attn_output)

        output_tensor = self.dropout(output_tensor)

        return output_tensor  # [B,T,F]
    
    def infer(
        self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:
        # x: [t, F]

        attn_mask = torch.ones(x.size(0), x.size(0), device=x.device, dtype=torch.bool)
        y = self.forward(
            input_tensor=x.unsqueeze(0), sequence_mask=seq_mask.unsqueeze(0), attn_mask=attn_mask)
        y = y[0, -ext_chunk_sz:]

        return y + x[-ext_chunk_sz:]


@dataclass
class ConformerRelPosBlockV1COV1Config:
    """
        Attributes:
            ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
            mhsa_cfg: Configuration for ConformerMHSAV1
            conv_cfg: Configuration for ConformerConvolutionV1
        """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardV2Config
    mhsa_cfg: ConformerMHSARelPosV1Config
    conv_cfg: ConformerConvolutionV2Config


class ConformerRelPosBlockV1COV1(ConformerBlockV3COV1):
    """
    Conformer block module, modifications compared to ConformerBlockV1:
    - uses ConfomerMHSARelPosV1 as MHSA module
    - enable constructing the block with self-defined module_list as ConformerBlockV2
    """

    def __init__(self, cfg: ConformerRelPosBlockV1COV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__(cfg)
        self.ff1 = ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSARelPosV2(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV2(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)


@dataclass
class ConformerRelPosEncoderV1COV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerRelPosBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerRelPosBlockV1COV1Config


class ConformerRelPosEncoderV1COV1(ConformerEncoderCOV2):
    """
    Modifications compared to ConformerEncoderV2:
    - supports Shaw's relative positional encoding using learnable position embeddings
      and Transformer-XL style relative PE using fixed sinusoidal or learnable position embeddings
    """

    def __init__(self, cfg: ConformerRelPosEncoderV1COV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__(cfg)

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerRelPosBlockV1COV1(cfg.block_cfg) for _ in range(cfg.num_layers)])
