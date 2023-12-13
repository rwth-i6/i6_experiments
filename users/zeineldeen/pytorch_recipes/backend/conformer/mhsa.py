import torch
from torch import nn
import math
from i6_models.config import ModelConfiguration

from typing import Tuple, Any


class MultiHeadSelfAttentionConfig(ModelConfiguration):
    att_heads: int
    """number of attention heads"""

    att_dim: int
    """attention dimension"""

    att_weights_dropout: float
    """dropout applied on attention weights"""


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Attention layer.
    Adapted from: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py
    """

    def __init__(self, cfg):
        super().__init__()

        att_heads = cfg.att_heads
        att_dim = cfg.att_dim
        assert att_heads % att_dim == 0

        self.d_k = att_dim // att_heads
        self.att_heads = att_heads
        self.linear_q = nn.Linear(att_dim, att_dim)
        self.linear_k = nn.Linear(att_dim, att_dim)
        self.linear_v = nn.Linear(att_dim, att_dim)
        self.linear_out = nn.Linear(att_dim, att_dim)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)

    def forward_qkv(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        :param query: query tensor of shape [B,T_q,F]
        :param key: key tensor of shape [B,T_k,F]
        :param value: value tensor of shape [B,T_k,F]
        :return:
            transformed query tensor of shape [B,H,T_q,d_k]
            transformed key and value tensors of shape [B,H,T_k,d_k]
        """

        batch_size = query.size(0)

        # proj then reshape to [B,H,T,d_k] to split and apply att for each head
        q = self.linear_q(query).view(batch_size, -1, self.att_heads, self.d_k).transpose(1, 2)
        k = self.linear_k(key).view(batch_size, -1, self.att_heads, self.d_k).transpose(1, 2)
        v = self.linear_v(value).view(batch_size, -1, self.att_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def forward_attention(self, value: torch.Tensor, scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute attention context vector.
        :param value: transformed value tensor of shape [B,H,T_k,d_k]
        :param scores: attention score tensor of shape [B,H,T_q,T_k]
        :param mask: mask tensor of shape [B,1,T_k]
        :return: attention context vector tensor of shape [B,T_q,d_k]
        """

        batch_size = value.size(0)
        if mask:
            mask = mask.unsqueeze(1).eq(0)  # [B,1,1,T_k]
            min_value = torch.finfo(scores.dtype).min  # -inf
            scores = scores.masked_fill(mask, min_value)
            att_weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # [B,H,T_q,T_k]
        else:
            att_weights = torch.softmax(scores, dim=-1)  # [B,H,T_q,T_k]
        att_weights_drop = self.dropout(att_weights)  # [B,H,T_q,T_k]
        out = torch.matmul(att_weights_drop, value)  # [B,H,T_q,d_k]
        out = out.reshape(
            out.transpose(1, 2).contiguous().view(batch_size, -1, self.att_heads * self.d_k)
        )  # concat heads [B,T_q,H*d_k]
        return self.linear_out(out)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        :param query:
        :param key:
        :param value:
        :param mask:
        :return:
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B,H,T_q,T_k]
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadSelfAttentionConfig(MultiHeadSelfAttentionConfig):
    zero_triu: bool
    """if True, the upper triangular part of the attention matrix is zeroed out"""


class RelPositionMultiHeadedAttention(MultiHeadSelfAttention):
    """
    Multi-Head Attention layer with relative position encoding.
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860

    Adapted from: https://github.com/espnet/espnet/blob/master/espnet/nets/pytorch_backend/transformer/attention.py#L209
    """

    def __init__(self, cfg):
        super().__init__(cfg)

        self.zero_triu = cfg.zero_triu

        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(cfg.att_dim, cfg.att_dim, bias=False)

        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute relative positional encoding.
        :param x: input tensor of shape [B,H,T_q,2 * T_k - 1]
        """

        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]  # only keep the positions from 0 to time2

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.

        :param query: query tensor of shape [B,T_q,F]
        :param key: key tensor of shape [B,T_k,F]
        :param value: value tensor of shape [B,T_k,F]
        :param pos_emb: positional embedding tensor of shape [B,2*T_k-1,d_k]
        :param mask: mask tensor of shape [B,1,T_k] or [B,T_q,T_k]
        :return: output tensor of shape [B,T_q,d_k]
        """

        q, k, v = self.forward_qkv(query, key, value)  # [B,H,T_q,d_k]
        q = q.transpose(1, 2)  # [B,T_q,H,d_k]

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # [B,H,2*T_q-1,d_k]

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)  # [B,H,T_q,d_k]
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)  # [B,H,T_q,d_k]

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))  # [B,H,T_q,T_k]

        # compute matrix b and matrix d
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))  # [B,H,T_q,2 * T_q - 1]
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)  # [B,H,T_q,T_k]

        return self.forward_attention(v, scores, mask)
