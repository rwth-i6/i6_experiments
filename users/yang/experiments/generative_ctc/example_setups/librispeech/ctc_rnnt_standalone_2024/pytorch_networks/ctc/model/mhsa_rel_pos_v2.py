from __future__ import annotations

__all__ = ["ConformerMHSARelPosV1", "ConformerMHSARelPosV1Config"]

from dataclasses import dataclass
import math
from typing import Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F

from i6_models.config import ModelConfiguration
from i6_models.util import compat
from i6_models.parts.dropout import BroadcastDropout


@dataclass
class ConformerMHSARelPosV2Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        with_bias: whether to add bias to qkv and output linear projections
        att_weights_dropout: attention weights dropout
        learnable_pos_emb: whether to use learnable relative positional embeddings instead of fixed sinusoidal ones
        rel_pos_clip: maximal relative postion for embedding
        with_linear_pos: whether to linearly transform the positional embeddings
        separate_pos_emb_per_head: whether to create head-dependent positional embeddings
        with_pos_bias: whether to add additional position bias terms to the attention scores
        pos_emb_dropout: dropout for the positional embeddings
        dropout: multi-headed self attention output dropout
        dropout_broadcast_axes: string of axes to which dropout is broadcast, e.g. "T" for broadcasting to the time axis
                                setting to None to disable broadcasting
    """

    input_dim: int
    num_att_heads: int
    with_bias: bool
    att_weights_dropout: float
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float
    dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"
        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"

from i6_models.parts.conformer.mhsa_rel_pos import ConformerMHSARelPosV1Config
class ConformerMHSARelPosV2(nn.Module):
    """
    allow optional mhsa dropout (or maybe also other type of dropout?) to make the model prior more consistent in traning and decoding
    Conformer multi-headed self-attention module supporting
        - self-attention with relative positional encoding proposed by Shaw et al. (cf. https://arxiv.org/abs/1803.02155)
            * learnable_pos_emb = True
            * with_pos_bias = False
            * with_linear_pos = False
            * separate_pos_emb_per_head = False (RETURNN default)
            * with_bias = False (RETURNN default)
        - and self-attention with Transformer-XL style relative PE by Dai et al.
            (cf. https://arxiv.org/abs/1901.02860, https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py,
                 https://github.com/espnet/espnet/blob/master/espnet2/asr_transducer/encoder/modules/attention.py#L9)
            * learnable_pos_emb = False
            * with_pos_bias = True
            * with_linear_pos = False (paper implementation) / with_linear_pos = True (ESPnet default)
            * separate_pos_emb_per_head = False (paper implementation) / separate_pos_emb_per_head = True (ESPnet default)
            * with_bias = False (paper implementation) / with_bias = True (ESPnet default)
    """

    def __init__(self, cfg: ConformerMHSARelPosV1Config):
        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.learnable_pos_emb = cfg.learnable_pos_emb
        self.rel_pos_clip = cfg.rel_pos_clip
        self.separate_pos_emb_per_head = cfg.separate_pos_emb_per_head
        self.with_pos_bias = cfg.with_pos_bias
        self.pos_emb_dropout = nn.Dropout(cfg.pos_emb_dropout)

        assert not self.learnable_pos_emb or self.rel_pos_clip

        self.att_weights_dropout = nn.Dropout(cfg.att_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=cfg.with_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.with_bias)

        self.register_parameter("rel_pos_embeddings", None)
        self.register_parameter("pos_bias_u", None)
        self.register_parameter("pos_bias_v", None)

        self.pos_emb_dim = (
            self.embed_dim if cfg.with_linear_pos or cfg.separate_pos_emb_per_head else self.embed_dim_per_head
        )
        if self.learnable_pos_emb:
            self.rel_pos_embeddings = nn.parameter.Parameter(torch.empty(self.rel_pos_clip * 2 + 1, self.pos_emb_dim))
        if cfg.with_linear_pos:
            self.linear_pos = nn.Linear(
                self.pos_emb_dim,
                self.embed_dim if cfg.separate_pos_emb_per_head else self.embed_dim_per_head,
                bias=False,
            )
        else:
            self.linear_pos = nn.Identity()

        if self.with_pos_bias:
            self.pos_bias_u = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))
            self.pos_bias_v = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))

        self.dropout = BroadcastDropout(cfg.dropout, dropout_broadcast_axes=cfg.dropout_broadcast_axes)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.learnable_pos_emb:
            nn.init.xavier_normal_(self.rel_pos_embeddings)
        if self.with_pos_bias:
            # init taken from espnet default
            nn.init.xavier_uniform_(self.pos_bias_u)
            nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, if_dropout: bool) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside
        """
        output_tensor = self.layernorm(input_tensor)  # [B, T, F]

        time_dim_size = output_tensor.shape[1]
        batch_dim_size = output_tensor.shape[0]

        # attention mask
        # T: query seq. length, T' key/value seg length; T = T' if same input tensor
        inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T']
        mask = (
            torch.zeros_like(inv_sequence_mask, dtype=input_tensor.dtype)
            .masked_fill(inv_sequence_mask, float("-inf"))
            .reshape(batch_dim_size, 1, 1, time_dim_size)
        )  # [B, 1, 1, T']

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
            ).view(1, 2 * time_dim_size - 1, self.pos_emb_dim)  # [1, T+T'-1, pos_emb_dim]

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

        attn = attn_ac + attn_bd + mask  # [B, #heads, T, T']
        attn_scaled = attn * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # softmax and dropout
        attn_output_weights = self.att_weights_dropout(F.softmax(attn_scaled, dim=-1))  # [B, #heads, T, T']

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']
        attn_output = torch.einsum("bhij, bjhf -> bihf", attn_output_weights, v).reshape(
            batch_dim_size, -1, self.embed_dim
        )

        output_tensor = self.out_proj(attn_output)
        if if_dropout:
            output_tensor = self.dropout(output_tensor)

        return output_tensor  # [B,T,F]

    @staticmethod
    def _rel_shift_bhij(x, k_len=None):
        """
        :param x: input tensor of shape (B, H, T, L) to apply left shift
        :k_len: length of the key squence
        """
        x_shape = x.shape

        x = torch.nn.functional.pad(x, (1, 0))  # [B, H, T, L+1]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2])  # [B, H, L+1, T]
        x = x[:, :, 1:]  # [B, H, L, T]
        x = x.reshape(x_shape)  # [B, H, T, L]]

        return x[:, :, :, :k_len] if k_len else x  # [B, H, T, T']

    @staticmethod
    def _sinusoidal_pe(pos_seq: torch.Tensor, embed_dim: int):
        """
        :param pos_seq: 1-D position sequence for which to compute embeddings
        :param embed_dim: embedding dimension
        """
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0, device=pos_seq.device) / embed_dim))

        sinusoid_input = torch.outer(pos_seq, inv_freq)

        pos_emb = torch.zeros(pos_seq.shape[0], embed_dim, device=pos_seq.device)

        pos_emb[:, 0::2] = sinusoid_input.sin()
        pos_emb[:, 1::2] = sinusoid_input.cos()

        return pos_emb
