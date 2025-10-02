"""
adds streamable mhsa for streamable ctc models without QAT
"""

import copy

import torch
from torch import nn
from torch.nn import init
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional, Union, List
from .baseline_no_qat_v1_cfg import MultiheadAttentionNoQuantV1Config
import math
from returnn.torch.context import get_run_ctx
from torch.ao.quantization.utils import check_min_max_valid
from torch.nn.quantized._reference.modules import Linear

from ...streamable_module import StreamableModule
from i6_models.util import compat



###############################################################################################################
# NOTE: now streamable
class MultiheadAttentionNoQuantStreamable(StreamableModule):
    def __init__(self, cfg: MultiheadAttentionNoQuantV4Config):
        super().__init__()
        self.cfg = cfg
        self.num_att_heads = cfg.num_att_heads
        self.input_dim = cfg.input_dim
        self.dim_heads = self.input_dim // self.num_att_heads

        self.in_proj = nn.Linear(self.input_dim, self.input_dim, bias=True)
        self.out_proj = nn.Linear(self.input_dim, self.input_dim, bias=True)

        self.norm = math.sqrt(self.dim_heads)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)

    def forward_offline(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
            sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ):
        """

        :param query: [B, T, D]
        :param key: [B, T, D]
        :param value: [B, T, D]
        :param sequence_mask: [B, T]
        :param attn_mask: [T, T]
        :return: [B, T, D], [B, num_att_heads, T, T]

        B: batch size
        T: time dimension (number of frames)
        D: feature dimension
        """

        batch_dim = query.shape[0]

        sequence_mask = sequence_mask.unsqueeze(1)  # [B, 1, T]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)  # [1, T, T]

        total_mask = sequence_mask
        if attn_mask is not None:
            total_mask = total_mask + attn_mask
        total_mask = total_mask.unsqueeze(1)  # [B, 1, T, T]

        # query = self.W_q(query)
        # key = self.W_k(key)
        # value = self.W_v(value)
        assert query is value is key, "currently only this case is implemented"

        x = self.in_proj(query)
        hidden_dim = query.size(-1)
        query, key, value = x.unflatten(-1, (3, hidden_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

        query = query.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        key = key.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        value = value.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']

        query = torch.transpose(query, 1, 2)  # [B, D//H, T, D']
        key = torch.transpose(key, 1, 2)  # [B, D//H, T, D']
        value = torch.transpose(value, 1, 2)  # [B, D//H, T, D']

        key = torch.transpose(key, -2, -1)  # [B, D//H, D', T]

        dot = torch.matmul(query, key)  # [B, D//H, T, T]
        dot = dot / self.norm

        # total_mask = total_mask.view(batch_dim, 1, 1, total_mask.size(1))
        dot = dot.masked_fill(total_mask, float("-inf"))
        alpha = self.softmax(dot)
        alpha = alpha.masked_fill(total_mask, 0.0)  # double masking due to nan vals from combination of seq_mask and attn_mask
        # alpha = self.dropout(alpha)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, D//H, T, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.out_proj(att_out)

        return att_out, alpha

    def forward_streaming(
            self, query: torch.Tensor, key: torch.Tensor, value:torch.Tensor,
            sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Calls self.forward_offline with provided attn_mask after combining chunks and returns
        the result again in a chunked format.
        Currently only supports MHSA, i.e. expects query = key = value.

        :param query: [B, N, C, D]
        :param key: [B, N, C, D]
        :param value: [B, N, C, D]
        :param sequence_mask: [B, N, C]
        :param attn_mask: expecting a causal mask that prevents chunks from attending to future chunks with shape [N*C, N*C]
        :return: [B, N, C, D], [B, num_att_heads, T, T]

        B: batch size
        N: number of chunks,
        C: chunk size,
        D: feature dimension
        """
        assert query.dim() == 4, ""
        assert torch.all((query == key) & (key == value)), "MHA currently only supports MHSA during streaming"

        bsz, num_chunks, chunk_sz, _ = query.shape

        # combine chunks [B, N, C, D] -> [B, N*C, D]
        x = query.flatten(1, 2)
        sequence_mask = sequence_mask.flatten(1, 2)

        att_out, alpha = self.forward_offline(query=x, key=x, value=x, sequence_mask=sequence_mask, attn_mask=attn_mask)

        # chunk again after attention
        att_out = att_out.view(bsz, num_chunks, chunk_sz, query.size(-1))

        return att_out, alpha

###############################################################################################################
