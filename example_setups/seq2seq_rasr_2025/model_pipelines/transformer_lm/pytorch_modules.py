import math
from dataclasses import dataclass
from typing import Union

import torch
from i6_models.config import ModelConfiguration
from i6_models.util import compat
from sisyphus import tk


@dataclass
class TransformerLinearConfig(ModelConfiguration):
    input_dim: int
    ff_dim: int
    output_dim: int
    dropout: float


@dataclass
class TransformerMHSAConfig(ModelConfiguration):
    input_dim: int
    num_heads: int
    dropout: float


@dataclass
class PositionalEncodingConfig(ModelConfiguration):
    embed_dim: int
    max_len: int
    dropout: float


@dataclass
class TransformerBlockConfig(ModelConfiguration):
    linear_config: TransformerLinearConfig
    mhsa_config: TransformerMHSAConfig


@dataclass
class TransformerLmConfig(ModelConfiguration):
    embed_dim: int
    hid_dim: int
    vocab_dim: Union[int, tk.Variable]
    num_layers: int
    pos_enc_config: PositionalEncodingConfig
    block_config: TransformerBlockConfig
    dropout: float


class TransformerMHSA(torch.nn.Module):
    def __init__(self, cfg: TransformerMHSAConfig, **_):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = torch.nn.MultiheadAttention(
            cfg.input_dim, cfg.num_heads, dropout=cfg.dropout, bias=False, batch_first=False
        )
        self.num_heads = cfg.num_heads
        self.dropout = cfg.dropout

    def torch_forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, D]
            key_padding_mask: Shape [B, N]
            attn_mask: Shape [N, N]

        Returns:
            Tensor of shape [N, B, D]
        """
        inv_key_padding_mask = compat.logical_not(key_padding_mask)  # [B, N]
        inv_attn_mask = compat.logical_not(attn_mask)  # [N, N]
        x = input.transpose(0, 1)  # [B, N, D]
        x = self.layernorm.forward(x)  # [B, N, D]
        x = x.transpose(0, 1)  # [N, B, D]
        output, _ = self.mhsa.forward(
            x, x, x, key_padding_mask=inv_key_padding_mask, need_weights=False, attn_mask=inv_attn_mask
        )

        return output

    def onnx_forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, D]
            key_padding_mask: Shape [B, N]
            attn_mask: Shape [N, N]

        Returns:
            Tensor of shape [N, B, D]
        """
        x = input.transpose(0, 1)  # [B, N, D]
        x = self.layernorm(x)  # [B, N, D]

        B, N, D = x.shape

        head_dim = D // self.num_heads
        BH = B * self.num_heads

        qkv = torch.nn.functional.linear(x, self.mhsa.in_proj_weight, bias=None)  # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, D]

        shape_qkv = (BH, N, head_dim)
        q = torch.reshape(q, shape_qkv)  # [B*H, N, head_dim]
        k = torch.reshape(k, shape_qkv)  # [B*H, N, head_dim]
        v = torch.reshape(v, shape_qkv)  # [B*H, N, head_dim]

        scale = torch.rsqrt(torch.tensor(head_dim, dtype=torch.float32))
        q = q * scale  # [B*H, N, head_dim]
        scores = torch.bmm(q, k.transpose(1, 2))  # [B*H, N, N]

        pad = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, N]
        scores = scores.reshape(B, self.num_heads, N, N)  # [B, H, N, N]
        scores = scores.masked_fill(compat.logical_not(pad), float("-inf"))  # [B, H, N, N]

        scores = scores.masked_fill(
            compat.logical_not(attn_mask).unsqueeze(0).unsqueeze(0), float("-inf")
        )  # [B, H, N, N]

        attn = torch.nn.functional.softmax(scores, dim=-1)  # [B, H, N, N]
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)  # [B, H, N, N]

        attn = attn.reshape(B * self.num_heads, N, N)  # [B*H, N, N]
        out = torch.bmm(attn, v)  # [B*H, N, head_dim]

        out = out.reshape(B, self.num_heads, N, head_dim)  # [B, H, N, head_dim]
        out = out.permute(2, 0, 1, 3)  # [N, B, H, head_dim]
        out = out.reshape(N, B, D)  # [B, N, D]

        return torch.nn.functional.linear(out, self.mhsa.out_proj.weight, bias=None)  # [N, B, D]

    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, D]
            key_padding_mask: Shape [B, N]
            attn_mask: Shape [N, N]

        Returns:
            Tensor of shape [N, B, D]
        """
        if torch.onnx.is_in_onnx_export():
            return self.onnx_forward(input, key_padding_mask, attn_mask)
        else:
            return self.torch_forward(input, key_padding_mask, attn_mask)


class TransformerLinear(torch.nn.Module):
    def __init__(self, cfg: TransformerLinearConfig):
        super().__init__()
        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.ff1 = torch.nn.Linear(cfg.input_dim, cfg.ff_dim)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.ff2 = torch.nn.Linear(cfg.ff_dim, cfg.output_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, D]

        Returns:
            Tensor of shape [N, B, D]
        """
        x = input.transpose(0, 1)  # [B, N, D]
        x = self.layernorm.forward(x)  # [B, N, D]
        x = x.transpose(0, 1)  # [N, B, D]
        x = self.ff1.forward(x)  # [N, B, F]
        x = torch.nn.functional.relu(x)  # [N, B, F]
        x = self.dropout.forward(x)  # [N, B, F]
        x = self.ff2.forward(x)  # [N, B, D]
        return self.dropout.forward(x)  # [N, B, D]


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: TransformerBlockConfig):
        super().__init__()
        self.linear_block = TransformerLinear(cfg.linear_config)
        self.mhsa_block = TransformerMHSA(cfg.mhsa_config)

    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, D]
            key_padding_mask: Shape [B, N]
            attn_mask: Shape [N, N]

        Returns:
            Tensor of shape [N, B, D]
        """
        mhsa_out = self.mhsa_block.forward(input, key_padding_mask, attn_mask)  # [N, B, D]
        mhsa_out_res = input + mhsa_out  # [N, B, D]
        linear_out = self.linear_block.forward(mhsa_out_res)  # [N, B, D]
        return mhsa_out_res + linear_out  # [N, B, D]


class PositionalEncoding(torch.nn.Module):
    def __init__(self, cfg: PositionalEncodingConfig):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=cfg.dropout)

        position = torch.arange(cfg.max_len).unsqueeze(1)
        d_model_2 = cfg.embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_model_2) * (-math.log(10000.0) / (d_model_2 - 1)))
        pe = torch.zeros(cfg.max_len, 1, cfg.embed_dim)
        pe[:, 0, :d_model_2] = torch.sin(position * div_term)
        pe[:, 0, d_model_2:] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Shape [N, B, E]

        Returns:
            Tensor of shape [N, B, E]
        """
        input = input + self.pe[: input.size(0)]  # [N, B, E]
        return self.dropout.forward(input)  # [N, B, E]


class TransformerLm(torch.nn.Module):
    def __init__(self, step: int, epoch: int, cfg: TransformerLmConfig, **_):
        super().__init__()
        assert isinstance(cfg.vocab_dim, int)

        self.embed = torch.nn.Embedding(cfg.vocab_dim, cfg.embed_dim)
        self.positional_encoding = PositionalEncoding(cfg.pos_enc_config)
        self.input_linear = torch.nn.Linear(cfg.embed_dim, cfg.hid_dim, bias=False)
        self.transformer_blocks = torch.nn.ModuleList(
            [TransformerBlock(cfg.block_config) for _ in range(cfg.num_layers)]
        )
        self.output_layernorm = torch.nn.LayerNorm(cfg.hid_dim)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.output_linear = torch.nn.Linear(cfg.hid_dim, cfg.vocab_dim)
        self.causal_mask = torch.tril(torch.ones(2000, 2000)).to(torch.bool)

        self._param_init()

    def forward(self, input: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Shape [B, N, F]
            seq_mask: Shape [B, N]

        Returns:
            Tensor of shape [B, N, V]
        """
        max_seq_length = input.size(1)
        causal_mask = self.causal_mask[:max_seq_length, :max_seq_length].to(input.device)
        x = input.transpose(0, 1)  # [N, B, F]
        x = self.embed.forward(x)  # [N, B, E]
        x = self.positional_encoding.forward(x)  # [N, B, E]
        x = self.input_linear.forward(x)  # [N, B, D]
        for block in self.transformer_blocks:
            x = block.forward(x, seq_mask, causal_mask)  # [N, B, D]
        x = self.output_layernorm.forward(x)  # [N, B, D]
        x = self.dropout.forward(x)  # [N, B, D]
        output_logit = self.output_linear.forward(x)  # [N, B, V]
        output_logit = output_logit.transpose(0, 1)  # [B, N, V]

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
                    torch.nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="linear")
