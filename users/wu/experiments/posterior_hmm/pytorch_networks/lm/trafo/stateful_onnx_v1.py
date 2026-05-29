"""
Stateful-ONNX wrappers for the kazuki_trafo_zijian_variant_v1 Transformer LM
that match the IO contract expected by RASR's `StatefulOnnxLabelScorer`:

    state-initializer.onnx : ()                          -> caches + pos + last_logits
    state-updater.onnx     : (caches, pos, token)        -> caches' + pos' + last_logits'
    scorer.onnx            : (last_logits)               -> scores

All three modules share weights with the same trained `Model` instance.

State tensor layout (fixed max-length KV cache):
  * CACHE_i  float32 [B, T_max, hidden_dim]   for i in 0..L-1
  * POS      int32   [B]                       (next-write position)
  * LAST_LOGITS float32 [B, vocab_dim]         (-log p(. | history))
"""

from typing import List, Tuple

import torch
from torch import nn

from .kazuki_trafo_zijian_variant_v1 import Model as BaseModel


def _single_step(
    base: BaseModel,
    token: torch.Tensor,
    caches_in: List[torch.Tensor],
    pos: torch.Tensor,
    t_max: int,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Run a single decoding step. Mirrors the cache logic in
    kazuki_trafo_zijian_variant_v1_decoding.py but with a fixed `T_max` cache.

    :param token: int32/int64 tensor of shape [B]
    :param caches_in: list of L cache tensors, each [B, T_max, hidden_dim]
    :param pos: int32 tensor of shape [B]; next-write position
    :returns: (caches_out, neg_log_probs[B, vocab_dim])
    """
    pos_long = pos.long()

    x = base.embed(token.long()).unsqueeze(1)  # [B, 1, embed_dim]
    x = base.positional_encoding(x, batch_first=True, cache_length=pos_long)
    x = base.input_linear(x)  # [B, 1, hidden_dim]

    valid_mask = torch.arange(t_max, device=x.device, dtype=pos_long.dtype)[None, :] <= pos_long[:, None]
    inv_valid_mask = ~valid_mask  # True = position is invalid (ignore)

    caches_out: List[torch.Tensor] = []
    for i, block in enumerate(base.transformer_blocks):
        cache_i = caches_in[i]

        x_normed = block.mhsa_block.layernorm(x)  # [B, 1, hidden_dim]
        index = pos_long[:, None, None].expand(-1, 1, x_normed.shape[-1])
        new_cache = cache_i.scatter(dim=1, index=index, src=x_normed)

        attn_out, _ = block.mhsa_block.mhsa(
            x_normed,
            new_cache,
            new_cache,
            key_padding_mask=inv_valid_mask,
            need_weights=False,
        )
        x_res = x + attn_out
        ffn_out = block.linear_block(x_res)
        x = x_res + ffn_out
        caches_out.append(new_cache)

    x = base.output_layernorm(x)
    logits = base.output_linear(x).squeeze(1)  # [B, vocab_dim]
    neg_log_probs = -torch.log_softmax(logits.float(), dim=-1)
    return caches_out, neg_log_probs


class StateUpdater(nn.Module):
    """
    Single decoding step: extend the KV cache by one new token at index `pos`,
    return updated caches, incremented position and the negative log-probs of
    the next-token distribution conditioned on the new history.
    """

    def __init__(self, base: BaseModel, t_max: int):
        super().__init__()
        self.base = base
        self.t_max = t_max
        self.num_layers = base.cfg.num_layers

    def forward(
        self,
        token: torch.Tensor,
        pos_in: torch.Tensor,
        *caches_in: torch.Tensor,
    ):
        assert len(caches_in) == self.num_layers, (len(caches_in), self.num_layers)
        caches_out, last_logits = _single_step(self.base, token, list(caches_in), pos_in, self.t_max)
        pos_out = (pos_in.long() + 1).to(pos_in.dtype)
        return (*caches_out, pos_out, last_logits)


class StateInitializer(nn.Module):
    """
    Produce the initial state corresponding to "after consuming `<s>`" so the
    first score request returns `-log p(. | <s>)`.

    The state initializer is called by RASR with batch=1 (initial scoring
    contexts are shared across hypotheses), so the output batch dim is
    hardcoded to 1. The single ONNX input is a dummy tensor that RASR feeds
    via the `encoder-states-size` IOSpec slot but whose value is ignored
    inside the graph -- it only exists because torch.onnx.export refuses to
    trace a forward with zero arguments.
    """

    def __init__(self, base: BaseModel, t_max: int, bos_index: int):
        super().__init__()
        self.base = base
        self.t_max = t_max
        self.num_layers = base.cfg.num_layers
        self.hidden_dim = base.cfg.hidden_dim
        self.register_buffer("bos_index", torch.tensor([bos_index], dtype=torch.int64))

    def forward(self, _dummy: torch.Tensor):
        zero_pos = torch.zeros((1,), dtype=torch.int32)
        empty_caches = [
            torch.zeros((1, self.t_max, self.hidden_dim), dtype=torch.float32)
            for _ in range(self.num_layers)
        ]
        caches_out, last_logits = _single_step(self.base, self.bos_index, empty_caches, zero_pos, self.t_max)
        pos_out = torch.ones((1,), dtype=torch.int32)
        return (*caches_out, pos_out, last_logits)


class Scorer(nn.Module):
    """
    Trivial scorer: the (negative log-probability) `last_logits` state IS the
    score vector we want, so the ONNX `scorer` is just an identity readout.
    Kept as a separate ONNX file because RASR's `StatefulOnnxLabelScorer`
    requires the triplet (initializer, updater, scorer).
    """

    def forward(self, last_logits_in: torch.Tensor) -> torch.Tensor:
        return last_logits_in


def state_name_map_initializer(num_layers: int):
    """ONNX custom metadata: output-tensor-name -> state-name."""
    m = {f"cache_{i}_out": f"CACHE_{i}" for i in range(num_layers)}
    m["pos_out"] = "POS"
    m["last_logits_out"] = "LAST_LOGITS"
    return m


def state_name_map_updater(num_layers: int):
    m = {f"cache_{i}_in": f"CACHE_{i}" for i in range(num_layers)}
    m.update({f"cache_{i}_out": f"CACHE_{i}" for i in range(num_layers)})
    m["pos_in"] = "POS"
    m["pos_out"] = "POS"
    m["last_logits_out"] = "LAST_LOGITS"
    return m


def state_name_map_scorer():
    return {"last_logits_in": "LAST_LOGITS"}
