from __future__ import annotations
from typing import Optional

__all__ = ["ConformerMHSARelposV1", "ConformerMHSARelposV1Config"]
from dataclasses import dataclass
import torch
import numpy as np

from i6_models.config import ModelConfiguration
from i6_models.util import compat


@dataclass
class ConformerMHSARelposV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dim and total dimension for query/key and value projections, should be divisible by `num_att_heads`
        num_att_heads: number of attention heads
        att_weights_dropout: attention weights dropout
        dropout: multi-headed self attention output dropout
    """

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


class ConformerMHSARelposV1(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: ConformerMHSARelposV1Config):
        super().__init__()

        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.dim_per_head = cfg.input_dim // cfg.num_att_heads
        self.num_att_heads = cfg.num_att_heads
        self.linear_q = torch.nn.Linear(cfg.input_dim, cfg.input_dim)
        self.linear_k = torch.nn.Linear(cfg.input_dim, cfg.input_dim)
        self.linear_v = torch.nn.Linear(cfg.input_dim, cfg.input_dim)
        self.linear_out = torch.nn.Linear(cfg.input_dim, cfg.input_dim)
        self.att_dropout = cfg.att_weights_dropout
        self.dropout = cfg.dropout

    def forward(
        self, input_tensor: torch.Tensor, rel_pos_enc: torch.Tensor, sequence_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param rel_pos_enc: Relative positional encoding tensor of shape (T, T, D).
        :param sequence_mask: Optional bool mask of shape (B, T), True signals within sequence, False outside.
        """

        x = self.layernorm(input_tensor)

        n_batch = x.size(0)
        n_time = x.size(1)

        q = self.linear_q(x).reshape(n_batch, -1, self.num_att_heads, self.dim_per_head)  # [B, T, H, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = self.linear_k(x).reshape(n_batch, -1, self.num_att_heads, self.dim_per_head)  # [B, T, H, D]
        k = k.transpose(1, 2)  # [B, H, T, D]
        v = self.linear_v(x).reshape(n_batch, -1, self.num_att_heads, self.dim_per_head)  # [B, T, H, D]
        v = v.transpose(1, 2)  # [B, H, T, D]

        qk = torch.matmul(q, k.transpose(-2, -1))  # [B, H, T, T]

        q_reshape = q.permute(2, 0, 1, 3).reshape(
            (n_time, n_batch * self.num_att_heads, self.dim_per_head)
        )  # [T, B*H, D]
        qp_reshape = torch.matmul(q_reshape, rel_pos_enc.transpose(-2, -1))  # [T, B*H, T]
        qp = qp_reshape.permute(1, 0, 2).reshape((n_batch, self.num_att_heads, n_time, n_time))  # [B, H, T, T]

        scores = (qk + qp) / np.sqrt(self.dim_per_head)  # [B, H, T, T]

        if sequence_mask is not None:
            inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T]
            inv_sequence_mask = inv_sequence_mask.reshape(n_batch, 1, 1, -1)  # [B, 1, 1, T]
            min_value = torch.finfo(scores.dtype).min  # essentially negative infinity

            scores = scores.masked_fill(inv_sequence_mask, min_value)  # [B, H, T, T]
            attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]
            attn = attn.masked_fill(inv_sequence_mask, 0.0)  # [B, H, T, T]
        else:
            attn = torch.softmax(scores, dim=-1)  # [B, H, T, T]

        attn = torch.nn.functional.dropout(attn, p=self.att_dropout, training=self.training)  # [B, H, T, T]

        x = torch.matmul(attn, v)  # [B, H, T, D]
        x = x.transpose(1, 2)  # [B, T, H, D]
        x = x.reshape(n_batch, -1, self.num_att_heads * self.dim_per_head)  # [B, T, H*D]

        x = self.linear_out(x)  # [B, T, H*D]
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)  # [B, T, H*D]

        return x
