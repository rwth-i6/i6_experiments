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
        self.input_dim = cfg.input_dim

    def torch_forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        inv_key_padding_mask = compat.logical_not(key_padding_mask)
        inv_attn_mask = compat.logical_not(attn_mask)
        x = input.transpose(0, 1)  # [B, N, D]
        x = self.layernorm(x)
        x = x.transpose(0, 1)  # [N, B, D]
        output, _ = self.mhsa(
            x, x, x, key_padding_mask=inv_key_padding_mask, need_weights=False, attn_mask=inv_attn_mask
        )
        return output

    def onnx_forward(
        self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor
    ) -> torch.Tensor:
        x = input.transpose(0, 1)  # [B, N, D]
        x = self.layernorm(x)  # [B, N, D]

        B, N, D = x.shape
        head_dim = D // self.num_heads

        qkv = torch.nn.functional.linear(x, self.mhsa.in_proj_weight, bias=None)  # [B, N, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, Hd]
        k = k.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        scale = torch.rsqrt(torch.tensor(head_dim, dtype=torch.float32, device=x.device))
        q = q * scale

        scores = torch.matmul(q, k.transpose(-2, -1))  # [B, H, N, N]

        # key_padding_mask: [B, N], True=valid
        key_valid = key_padding_mask.unsqueeze(1).unsqueeze(1)  # [B,1,1,N]
        scores = scores.masked_fill(~key_valid, float("-inf"))

        # attn_mask: [N, N], True=allowed
        attn_valid = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        scores = scores.masked_fill(~attn_valid, float("-inf"))

        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [B, H, N, Hd]
        out = out.permute(2, 0, 1, 3).reshape(N, B, D)  # [N, B, D]

        return torch.nn.functional.linear(out, self.mhsa.out_proj.weight, bias=None)

    def forward(self, input: torch.Tensor, key_padding_mask: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        if torch.onnx.is_in_onnx_export():
            return self.onnx_forward(input, key_padding_mask, attn_mask)
        return self.torch_forward(input, key_padding_mask, attn_mask)

    def forward_with_kv_cache(
        self,
        input: torch.Tensor,
        labels_lens: torch.Tensor,
        state_lengths: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ONNX decoding forward with explicit KV cache.

        Args:
            input: [T_suf, B, D]
            labels_lens: [B]             valid suffix lengths
            state_lengths: [B]           valid prefix lengths
            k_cache: [B, T_pref, D]      right-aligned prefix K cache
            v_cache: [B, T_pref, D]      right-aligned prefix V cache

        Returns:
            output: [T_suf, B, D]
            k_new: [B, T_suf, D]         suffix K only
            v_new: [B, T_suf, D]         suffix V only
        """
        x = input.transpose(0, 1)  # [B, T_suf, D]
        x = self.layernorm(x)  # [B, T_suf, D]

        B, T_suf, D = x.shape
        T_pref = k_cache.shape[1]
        H = self.num_heads
        Hd = D // H

        qkv = torch.nn.functional.linear(x, self.mhsa.in_proj_weight, bias=None)  # [B, T_suf, 3D]
        q_new, k_new, v_new = qkv.chunk(3, dim=-1)  # each [B, T_suf, D]

        # Full K/V for attention = cached prefix + new suffix
        k_full = torch.cat([k_cache, k_new], dim=1)  # [B, T_pref + T_suf, D]
        v_full = torch.cat([v_cache, v_new], dim=1)

        # Reshape for multi-head attention
        q = q_new.reshape(B, T_suf, H, Hd).permute(0, 2, 1, 3)  # [B,H,T_suf,Hd]
        k = k_full.reshape(B, T_pref + T_suf, H, Hd).permute(0, 2, 1, 3)  # [B,H,T_tot,Hd]
        v = v_full.reshape(B, T_pref + T_suf, H, Hd).permute(0, 2, 1, 3)

        scale = torch.rsqrt(torch.tensor(Hd, dtype=torch.float32, device=x.device))
        q = q * scale

        scores = torch.matmul(q, k.transpose(-2, -1))  # [B,H,T_suf,T_tot]

        # Prefix validity mask for right-aligned prefix cache
        # valid prefix positions are the LAST state_lengths[b] positions in T_pref
        pref_pos = torch.arange(T_pref, device=x.device, dtype=state_lengths.dtype).unsqueeze(0)  # [1,T_pref]
        pref_valid = pref_pos >= (T_pref - state_lengths).unsqueeze(1)  # [B,T_pref]

        # Suffix validity mask
        suf_pos = torch.arange(T_suf, device=x.device, dtype=labels_lens.dtype).unsqueeze(0)  # [1,T_suf]
        suf_valid = suf_pos < labels_lens.unsqueeze(1)  # [B,T_suf]

        key_valid = torch.cat([pref_valid, suf_valid], dim=1)  # [B,T_tot]
        scores = scores.masked_fill(~key_valid.unsqueeze(1).unsqueeze(1), float("-inf"))

        # Causal mask for suffix queries over [prefix | suffix]
        # - all prefix positions are visible
        # - in suffix, query t can see suffix positions <= t
        pref_visible = torch.ones((T_suf, T_pref), dtype=torch.bool, device=x.device)
        suf_causal = torch.tril(torch.ones((T_suf, T_suf), dtype=torch.bool, device=x.device))
        attn_valid = torch.cat([pref_visible, suf_causal], dim=1)  # [T_suf, T_tot]

        scores = scores.masked_fill(~attn_valid.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = torch.nn.functional.softmax(scores, dim=-1)
        attn = torch.nn.functional.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)  # [B,H,T_suf,Hd]
        out = out.permute(2, 0, 1, 3).reshape(T_suf, B, D)  # [T_suf,B,D]
        out = torch.nn.functional.linear(out, self.mhsa.out_proj.weight, bias=None)

        return out, k_new, v_new


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
        mhsa_out = self.mhsa_block(input, key_padding_mask, attn_mask)
        mhsa_out_res = input + mhsa_out
        linear_out = self.linear_block(mhsa_out_res)
        return mhsa_out_res + linear_out

    def forward_with_kv_cache(
        self,
        input: torch.Tensor,
        labels_lens: torch.Tensor,
        state_lengths: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            input: [T_suf, B, D]
            labels_lens: [B]
            state_lengths: [B]
            k_cache: [B, T_pref, D]
            v_cache: [B, T_pref, D]

        Returns:
            output: [T_suf, B, D]
            k_new: [B, T_suf, D]
            v_new: [B, T_suf, D]
        """
        mhsa_out, k_new, v_new = self.mhsa_block.forward_with_kv_cache(
            input=input,
            labels_lens=labels_lens,
            state_lengths=state_lengths,
            k_cache=k_cache,
            v_cache=v_cache,
        )
        mhsa_out_res = input + mhsa_out
        linear_out = self.linear_block(mhsa_out_res)
        return mhsa_out_res + linear_out, k_new, v_new


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
            input: [N, B, E]
        Returns:
            [N, B, E]
        """
        input = input + self.pe[: input.size(0)]
        return self.dropout(input)

    def forward_with_offsets(self, input: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        ONNX-friendly positional encoding with per-batch start offsets.

        Args:
            input: [T, B, E]
            offsets: [B]  (prefix lengths / absolute start positions)

        Returns:
            [T, B, E]
        """
        T, B, E = input.shape
        device = input.device

        # positions[b, t] = offsets[b] + t
        rel = torch.arange(T, device=device, dtype=offsets.dtype).unsqueeze(0)  # [1, T]
        positions = offsets.unsqueeze(1) + rel  # [B, T]

        pe_table = self.pe[:, 0, :]  # [max_len, E]

        # Gather PE for each batch/time position
        pe_table_b = pe_table.unsqueeze(0).expand(B, -1, -1)  # [B, max_len, E]
        gather_idx = positions.unsqueeze(-1).expand(B, T, E)  # [B, T, E]
        pe_bt = torch.gather(pe_table_b, dim=1, index=gather_idx)  # [B, T, E]

        x = input.transpose(0, 1)  # [B, T, E]
        x = x + pe_bt
        x = self.dropout(x)
        return x.transpose(0, 1)  # [T, B, E]


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

        self.num_layers = cfg.num_layers
        self.hid_dim = cfg.hid_dim

        self._param_init()

    def forward(self, input: torch.Tensor, seq_mask: torch.Tensor) -> torch.Tensor:
        """
        Training/full-sequence forward.

        Args:
            input: [B, N]
            seq_mask: [B, N] True=valid

        Returns:
            [B, N, V]
        """
        max_seq_length = input.size(1)
        causal_mask = self.causal_mask[:max_seq_length, :max_seq_length].to(input.device)
        x = input.transpose(0, 1)  # [N, B]
        x = self.embed(x)  # [N, B, E]
        x = self.positional_encoding(x)  # [N, B, E]
        x = self.input_linear(x)  # [N, B, D]
        for block in self.transformer_blocks:
            x = block(x, seq_mask, causal_mask)
        x = self.output_layernorm(x)
        x = self.dropout(x)
        output_logit = self.output_linear(x)  # [N, B, V]
        return output_logit.transpose(0, 1)  # [B, N, V]

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


class TransformerLmOnnxWrapper(TransformerLm):
    def __init__(self, step: int, epoch: int, cfg: TransformerLmConfig, **_):
        super().__init__(step, epoch, cfg)

    def forward(
        self,
        input: torch.Tensor,
        labels_lens: torch.Tensor,
        state_lengths: torch.Tensor,
        *kv_cache: torch.Tensor,
    ):
        """
        ONNX export / runtime forward with explicit KV cache.

        Args:
            input: [B, T_suf]                suffix tokens
            labels_lens: [B]                valid suffix lengths
            state_lengths: [B]              valid prefix lengths
            kv_cache:
                layer0_k_cache, layer0_v_cache, layer1_k_cache, layer1_v_cache, ...
                each [B, T_pref, D], right-aligned in T_pref

        Returns:
            tuple(
                logits,                     [B, T_suf, V]
                layer0_k_new, layer0_v_new, [B, T_suf, D]
                layer1_k_new, layer1_v_new,
                ...
            )
        """
        assert (
            len(kv_cache) == 2 * self.num_layers
        ), f"Expected {2 * self.num_layers} KV cache tensors, got {len(kv_cache)}"

        x = input.transpose(0, 1)  # [T_suf, B]
        x = self.embed(x)  # [T_suf, B, E]
        x = self.positional_encoding.forward_with_offsets(x, state_lengths)  # [T_suf, B, E]
        x = self.input_linear(x)  # [T_suf, B, D]

        new_states = []

        for layer_idx, block in enumerate(self.transformer_blocks):
            k_cache = kv_cache[2 * layer_idx]  # [B, T_pref, D]
            v_cache = kv_cache[2 * layer_idx + 1]  # [B, T_pref, D]

            x, k_new, v_new = block.forward_with_kv_cache(
                input=x,
                labels_lens=labels_lens,
                state_lengths=state_lengths,
                k_cache=k_cache,
                v_cache=v_cache,
            )
            new_states.extend([k_new, v_new])

        x = self.output_layernorm(x)
        x = self.dropout(x)
        logits = self.output_linear(x).transpose(0, 1)  # [B, T_suf, V]

        scores = -torch.log_softmax(logits, dim=-1)

        return (scores, *new_states)


