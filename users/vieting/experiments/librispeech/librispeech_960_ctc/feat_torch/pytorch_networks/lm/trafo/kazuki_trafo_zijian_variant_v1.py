import math
import torch
from torch import nn

from i6_models.util import compat

from .kazuki_trafo_zijian_variant_v1_cfg import (
    TransformerMHSAConfig,
    TransformerLinearConfig,
    TransformerBlockConfig,
    TransformerLMConfig,
)

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T] as boolean
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask



class TransformerMHSA(nn.Module):
    def __init__(self, cfg: TransformerMHSAConfig, **kwargs):
        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)
        self.batch_first = cfg.batch_first
        self.mhsa = nn.MultiheadAttention(
            cfg.input_dim, cfg.num_heads, dropout=cfg.dropout, batch_first=self.batch_first
        )

    def forward(self, input, key_padding_mask, attn_mask):
        """
        input: shape (T,B,D) or (B,T,D)
        key_padding_mask: shape (B, T)
        attn_mask: shape (T,T') where T' is the length of key, T is the length of query, in this case they are the same
        """

        inv_key_padding_mask = compat.logical_not(key_padding_mask)
        inv_attn_mask = compat.logical_not(attn_mask)
        if not self.batch_first:
            x = input.transpose(0,1) # (T,B,D) -> (B,T,D)
        else:
            x = input
        x = self.layernorm(x) # always (B,T,D)
        if not self.batch_first:
            x = x.transpose(0,1) # (B,T,D) -> (T,B,D)
        output, _ = self.mhsa(x, x, x, key_padding_mask=inv_key_padding_mask, need_weights=False, attn_mask=inv_attn_mask)

        return output




class TransformerLinear(nn.Module):
    def __init__(self, cfg: TransformerLinearConfig):
        super().__init__()
        self.layernorm = nn.LayerNorm(cfg.input_dim)
        self.ff1 = nn.Linear(cfg.input_dim, cfg.ff_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.ff2 = nn.Linear(cfg.ff_dim, cfg.output_dim)
        self.batch_first = cfg.batch_first

    def forward(self, input):
        if not self.batch_first:
            x = input.transpose(0, 1)  # (B,T,D)
        else:
            x = input
        x = self.layernorm(x)
        if not self.batch_first:
            x = x.transpose(0,1)
        x = self.ff1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        x = self.ff2(x)
        output = self.dropout(x)

        return output


class TransformerBlock(nn.Module):
    def __init__(self, cfg:TransformerBlockConfig):
        super().__init__()
        self.linear_block = TransformerLinear(cfg.linear_config)
        self.mhsa_block = TransformerMHSA(cfg.mhsa_config)

    def forward(self, input, key_padding_mask, attn_mask):
        x = self.mhsa_block(input, key_padding_mask, attn_mask)
        x_res = input + x
        x = self.linear_block(x_res)
        output = x_res + x

        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor, batch_first: bool=False) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        if not batch_first:
            x = x + self.pe[:x.size(0)]
        else:
            x = x + self.pe.transpose(0,1)[0, :x.size(1)]
        return self.dropout(x)


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = TransformerLMConfig.from_dict(model_config_dict)
        self.embed = nn.Embedding(self.cfg.vocab_dim, self.cfg.embed_dim)
        self.positional_encoding = PositionalEncoding(self.cfg.embed_dim, dropout=self.cfg.dropout)
        self.input_linear = nn.Linear(self.cfg.embed_dim, self.cfg.hidden_dim, bias=False)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(self.cfg.block_config) for _ in range(self.cfg.num_layers)])
        self.output_layernorm = nn.LayerNorm(self.cfg.hidden_dim)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.output_linear = nn.Linear(self.cfg.hidden_dim, self.cfg.vocab_dim)
        self.causal_mask = torch.tril(torch.ones(2000,2000)).to(torch.bool)
        self.batch_first = self.cfg.batch_first
        self._param_init()

    def forward(self, input, seq_mask):
        # input shape (B,T,D)
        batch, max_seq_length = input.size()
        if max_seq_length > self.causal_mask.shape[0]:
            self.causal_mask = torch.tril(torch.ones(max_seq_length, max_seq_length)).to(torch.bool)
        causal_mask = self.causal_mask[:max_seq_length,:max_seq_length].to(input.device)
        if not self.batch_first:
            x = torch.transpose(input, 0,1)
        else:
            x = input
        x = self.embed(x)
        x = self.positional_encoding(x, self.batch_first)
        x = self.input_linear(x)
        for block in self.transformer_blocks:
            x = block(x, seq_mask, causal_mask)
        x = self.output_layernorm(x)
        x = self.dropout(x)
        output_logit = self.output_linear(x)
        if not self.batch_first:
            output_logit = torch.transpose(output_logit, 0, 1) # the output shape is always (B,T,D)

        return output_logit


    def _param_init(self):
        """
        initialization used in Kazuki's setup
        """
        for m in self.modules():
            for name, param in m.named_parameters():
                if "bias" or "layernorm" in name:
                    continue
                else:
                    nn.init.kaiming_uniform_(param,mode='fan_in', nonlinearity='linear') # consistent with kazuki's init


def train_step(*, model: Model, data, run_ctx, **kwargs):
    labels = data["data"]
    labels_len = data["data:size1"]
    delayed_labels = data["delayed"]

    seq_mask = mask_tensor(labels, labels_len)
    lm_logits = model(delayed_labels, seq_mask)  # (B, S, F)

    ce_loss = torch.nn.functional.cross_entropy(lm_logits.transpose(1, 2), labels.long(), reduction='none')
    ce_loss = (ce_loss * seq_mask).sum()
    total_length = torch.sum(labels_len)

    run_ctx.mark_as_loss(name="ce", loss=ce_loss, inv_norm_factor=total_length)