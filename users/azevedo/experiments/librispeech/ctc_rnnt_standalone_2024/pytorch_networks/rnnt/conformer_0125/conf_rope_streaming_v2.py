import torch
import numpy as np
from typing import List, Optional, Tuple
import math

from i6_models.util import compat

from i6_models.parts.dropout import BroadcastDropout
from ..conformer_2.mhav2 import _scaled_dot_product_attention

from ..auxil.functional import rotary_pos_encoding, create_chunk_mask, add_lookahead_v2, mask_tensor


def init_sinusoidal_embeds(out, num_embeddings: int, embedding_dim: int, device=None) -> torch.Tensor:
    # TODO: test
    assert embedding_dim % 2 == 0, "Even embedding dimension required."
    sinusoids = np.pow(
        10e4, - 2/embedding_dim * (np.arange(embedding_dim // 2) - 1), device=device
    )[np.newaxis, :]  # [1, d//2]

    positions = np.arange(num_embeddings, device=device)[:, np.newaxis]  # [T, 1]

    pos_sinusoidals = positions @ sinusoids  # [T, d//2]

    cos_weights = torch.from_numpy(np.cos(pos_sinusoidals))
    sin_weights = torch.from_numpy(np.sin(pos_sinusoidals))

    torch.stack((sin_weights, cos_weights), dim=0)

    out.requires_grad = False
    out[:, :embedding_dim//2] = sin_weights
    out[:, embedding_dim//2:] = cos_weights
    out.detach_()

    return out


def apply_rotary_embeddings(x, weights):
    """
    see https://github.com/JunnYu/RoFormer_pytorch/blob/roformer_v2/src/roformer/modeling_roformer.py

    x: [B, T, d]
    weights: [2, T, d//2]
    """
    # TODO: check dim of weights
    sin, cos = weights  # [T, d//2]
    x, y = x[:, :, ::2], x[:, :, 1::2]  # [B, T, d//2]
    return torch.stack([x * cos - y * sin, y * cos + x * sin], dim=-1).flatten(-2, -1)


class SinusoidalEmbeddings(torch.nn.Embedding):
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            padding_idx: Optional[int] = None,
            device=None,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, padding_idx, device)

        self.weight = init_sinusoidal_embeds(
            self.weight, self.num_embeddings, self.embedding_dim, device=self.device
        )

    @torch.no_grad()
    def forward(self, seq_len: int):
        positions = torch.arange(
            seq_len,
            dtype=torch.long,
            device=self.weight.device
        )
        return super().forward(positions)


class RoformerMHSAV1(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.att_weights_dropout = cfg.att_weights_dropout

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.qkv_proj = torch.nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=cfg.with_bias)
        self.out_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.with_bias)

        self.pos_embed = SinusoidalEmbeddings(
            num_embeddings=cfg.num_embeddings,
            embedding_dim=self.embed_dim,
            padding_idx=0,
        )

        self.dropout = BroadcastDropout(cfg.dropout, dropout_broadcast_axes=cfg.dropout_broadcast_axes)

    # TODO: test
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
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']

        sinusoidal_weights = self.pos_embed(time_dim_size)

        q_rope = apply_rotary_embeddings(q, sinusoidal_weights)
        k_rope = apply_rotary_embeddings(k, sinusoidal_weights)
        v_rope = apply_rotary_embeddings(v, sinusoidal_weights)

        attn_output = _scaled_dot_product_attention(
            query=q_rope,
            key=k_rope,
            value=v_rope,
            attn_mask=total_mask,
            dropout_p=self.att_weights_dropout
        )

        output_tensor = self.out_proj(attn_output)

        output_tensor = self.dropout(output_tensor)

        return output_tensor  # [B,T,F]





