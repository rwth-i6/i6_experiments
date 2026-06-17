"""
code based on code from Jaehyeon Kim; taken and modified from: https://github.com/jaywalnut310/glow-tts
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from typing import Optional

from i6_models.config import ModelConfiguration

def convert_pad_shape(pad_shape):
  l = pad_shape[::-1]
  pad_shape = [item for sublist in l for item in sublist]
  return pad_shape


@dataclass
class GlowTTSMultiHeadAttentionV1Config(ModelConfiguration):
    # like default MHSA
    input_dim: int
    num_att_heads: int
    dropout: float
    att_weights_dropout: float

    # additional args
    window_size: Optional[int] = None
    heads_share: bool = True
    block_length: Optional[int] = None
    proximal_bias: bool = False
    proximal_init: bool = False


class GlowTTSMultiHeadAttentionV1(nn.Module):
    """

    """
    def __init__(self, cfg: GlowTTSMultiHeadAttentionV1Config, features_last=True):
        """

        :param cfg:
        :param features_last: consider inputs and ouputs as [B, T, F]
        """
        super().__init__()
        assert cfg.input_dim % cfg.num_att_heads == 0

        self.cfg = cfg
        self.features_last = features_last
        self.attn = None

        self.k_channels = cfg.input_dim // cfg.num_att_heads
        self.conv_q = nn.Conv1d(cfg.input_dim, cfg.input_dim, 1)
        self.conv_k = nn.Conv1d(cfg.input_dim, cfg.input_dim, 1)
        self.conv_v = nn.Conv1d(cfg.input_dim, cfg.input_dim, 1)
        if cfg.window_size is not None:
            n_heads_rel = 1 if cfg.heads_share else cfg.num_att_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, cfg.window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, cfg.window_size * 2 + 1, self.k_channels) * rel_stddev)
        self.conv_o = nn.Conv1d(cfg.input_dim, cfg.input_dim, 1)
        self.drop = nn.Dropout(cfg.dropout)
        self.att_weights_drop = nn.Dropout(cfg.att_weights_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if cfg.proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None, causal=False):
        """

        :param x: [B, T, F] if features_last else [B, F, T]
        :param c: [B, T, F] if features_last else [B, F, T]
        :param attn_mask: [B, T] if features_last else [B, 1, T]
        :param causal: multiply attention mask with causal mask
        :return:
        """
        if self.features_last:
            q = self.conv_q(x.transpose(1, 2))
            k = self.conv_k(c.transpose(1, 2))
            v = self.conv_v(c.transpose(1, 2))

            attn_mask = attn_mask.unsqueeze(1)
        else:
            q = self.conv_q(x)
            k = self.conv_k(c)
            v = self.conv_v(c)

        attn_mask_full = attn_mask.unsqueeze(2) * attn_mask.unsqueeze(-1)

        if causal:
            causal_mask = torch.ones((1, 1, attn_mask_full.size(2), attn_mask_full.size(3))).tril(0)
            attn_mask_full = attn_mask_full * causal_mask

        x, self.attn = self.attention(q, k, v, mask=attn_mask_full)

        x = self.conv_o(x)
        return x.transpose(1, 2) if self.features_last else x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.cfg.num_att_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.cfg.num_att_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.cfg.num_att_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.k_channels)
        if self.cfg.window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        if self.cfg.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.cfg.block_length is not None:
                block_mask = torch.ones_like(scores).triu(-self.cfg.block_length).tril(self.cfg.block_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        p_attn = nn.functional.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        p_attn = self.att_weights_drop(p_attn)
        output = torch.matmul(p_attn, value)
        if self.cfg.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        output = self.drop(output)
        return output, p_attn

    def _matmul_with_relative_values(self, x, y):
        """
        x: [b, h, l, m]
        y: [h or 1, m, d]
        ret: [b, h, l, d]
        """
        ret = torch.matmul(x, y.unsqueeze(0))
        return ret

    def _matmul_with_relative_keys(self, x, y):
        """
        x: [b, h, l, d]
        y: [h or 1, m, d]
        ret: [b, h, l, m]
        """
        ret = torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))
        return ret

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.cfg.window_size + 1
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.cfg.window_size + 1), 0)
        slice_start_position = max((self.cfg.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = nn.functional.pad(
                relative_embeddings,
                convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:, slice_start_position:slice_end_position]
        return used_relative_embeddings

    def _relative_position_to_absolute_position(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()
        # Concat columns of pad to shift from relative to absolute indexing.
        x = nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))

        # Concat extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))

        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    def _absolute_position_to_relative_position(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = nn.functional.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = nn.functional.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    def _attention_bias_proximal(self, length):
        """Bias for self-attention to encourage attention to close positions.
        Args:
          length: an integer scalar.
        Returns:
          a Tensor with shape [1, 1, length, length]
        """
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)