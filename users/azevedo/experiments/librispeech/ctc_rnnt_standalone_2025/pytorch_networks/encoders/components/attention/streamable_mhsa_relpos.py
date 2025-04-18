import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
import math

from i6_models.util import compat
from i6_models.parts.conformer import ConformerMHSARelPosV1Config
from i6_models.parts.dropout import BroadcastDropout

from ..streamable_layernorm import StreamableLayerNormV1
from ....streamable_module import StreamableModule



# TODO: config (put dual_mode in config)

class StreamableConformerMHSARelPosV2(StreamableModule):
    def __init__(self, cfg: ConformerMHSARelPosV1Config, dual_mode: bool):
        super().__init__()

        self.layernorm = StreamableLayerNormV1(cfg.input_dim, dual_mode=dual_mode)

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

    def forward_offline(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, E)
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

        return output_tensor + input_tensor  # [B,T,F]

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

    def forward_streaming(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param input_tensor: [B, N, C, E]
        :param sequence_mask: [B, N, C]
        :param attn_mask: [N*C, N*C]

        :return: [B, N, C, E]
        """
        assert input_tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = input_tensor.shape

        # [B, N, C, F] -> [B, N*C, F]
        x = input_tensor.flatten(1, 2)
        sequence_mask = sequence_mask.flatten(1, 2)

        out = self.forward_offline(x, sequence_mask, attn_mask)

        out = out.view(bsz, num_chunks, chunk_sz, input_tensor.size(-1))

        return out

    def infer(
            self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:

        # x.shape: [t, F]
        attn_mask = torch.ones(x.size(0), x.size(0), device=x.device, dtype=torch.bool)
        y = self.forward_offline(
            input_tensor=x.unsqueeze(0), sequence_mask=seq_mask.unsqueeze(0), attn_mask=attn_mask)
        
        return y[0, -ext_chunk_sz:]  # [C+R, F]