import math
import torch
from torch import nn
from dataclasses import dataclass


from i6_models.config import ModelConfiguration
from i6_models.util import compat
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import Import
# by default now bias in self-attn layer, same as kazuki's trafo

@dataclass
class TransformerLinearConfig(ModelConfiguration):
    input_dim: int
    ff_dim: int
    output_dim: int
    dropout: float=0.0
    batch_first: bool=False
@dataclass
class TransformerMHSAConfig(ModelConfiguration):
    input_dim: int
    num_heads: int
    dropout: float=0.0
    batch_first: bool=False
    bias: bool=False
@dataclass
class TransformerBlockConfig(ModelConfiguration):
    linear_config: TransformerLinearConfig
    mhsa_config: TransformerMHSAConfig


@dataclass
class TransformerLMConfig(ModelConfiguration):
    embed_dim: int
    hid_dim: int
    vocab_dim: int
    num_layers: int
    block_config: TransformerBlockConfig
    batch_first: bool=False
    dropout: float=0.0


class TransformerMHSA(nn.Module):
    def __init__(self, cfg: TransformerMHSAConfig, **kwargs):
        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)
        self.batch_first = cfg.batch_first
        self.mhsa = nn.MultiheadAttention(
            cfg.input_dim, cfg.num_heads, dropout=cfg.dropout, bias=cfg.bias, batch_first=self.batch_first
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
            x = x + self.pe.transpose(0,1)[0, :x.size(0)]
        return self.dropout(x)


class TransformerLM(nn.Module):
    def __init__(self, step: int, cfg: TransformerLMConfig, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_dim, cfg.embed_dim)
        self.positional_encoding = PositionalEncoding(cfg.embed_dim, dropout=cfg.dropout)
        self.input_linear = nn.Linear(cfg.embed_dim, cfg.hid_dim, bias=False)
        self.transformer_blocks = nn.ModuleList([TransformerBlock(cfg.block_config) for _ in range(cfg.num_layers)])
        self.output_layernorm = nn.LayerNorm(cfg.hid_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.output_linear = nn.Linear(cfg.hid_dim, cfg.vocab_dim)
        self.causal_mask = torch.tril(torch.ones(2000,2000)).to(torch.bool)
        self.batch_first = cfg.batch_first
        self._param_init()

    def forward(self, input, seq_mask):
        # input shape (B,T,D)
        batch, max_seq_length = input.size()
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
                    # variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=1.0)



def get_train_serializer(
    model_config: TransformerLMConfig,
    train_step_package: str
) -> Collection:
    # pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{TransformerLM.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{train_step_package}.train_step"),
        ],
    )
