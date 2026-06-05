import math

import torch
from torch import nn
import torch.nn.functional as F

from .decoder_lm_cfg import ModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_size: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_size))
        encoding = torch.zeros(max_len, hidden_size, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        if hidden_size > 1:
            encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, length: int):
        if length > self.encoding.shape[0]:
            raise ValueError(f"Sequence length {length} exceeds max_position_embeddings={self.encoding.shape[0]}")
        return self.encoding[:length]


class Model(nn.Module):
    """
    Decoder-only Transformer language model with tied input/output embeddings.

    The token embedding dim can differ from the Transformer hidden size. Inputs
    are projected to hidden size, sinusoidal positions are added, and the final
    hidden states are projected back to embedding dim before the tied output
    projection.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.hidden_size % self.cfg.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size={self.cfg.hidden_size} must be divisible by "
                f"num_attention_heads={self.cfg.num_attention_heads}"
            )

        self.token_embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embedding_dim,
            padding_idx=self.cfg.pad_token_id,
        )
        self.input_projection = nn.Linear(self.cfg.embedding_dim, self.cfg.hidden_size)
        self.positional_encoding = SinusoidalPositionalEncoding(
            hidden_size=self.cfg.hidden_size,
            max_len=self.cfg.max_position_embeddings,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_size,
            nhead=self.cfg.num_attention_heads,
            dim_feedforward=self.cfg.intermediate_size,
            dropout=self.cfg.dropout,
            activation="gelu",
            layer_norm_eps=self.cfg.layer_norm_eps,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.cfg.num_hidden_layers,
            norm=nn.LayerNorm(self.cfg.hidden_size, eps=self.cfg.layer_norm_eps),
        )
        self.output_projection = nn.Linear(self.cfg.hidden_size, self.cfg.embedding_dim)
        if self.cfg.output_bias:
            self.output_bias = nn.Parameter(torch.zeros(self.cfg.vocab_size))
        else:
            self.register_parameter("output_bias", None)

        self._init_added_parameters()

    def _init_added_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=self.cfg.initializer_range)
        with torch.no_grad():
            self.token_embedding.weight[self.cfg.pad_token_id].zero_()
        nn.init.normal_(self.input_projection.weight, mean=0.0, std=self.cfg.initializer_range)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=self.cfg.initializer_range)
        nn.init.zeros_(self.output_projection.bias)

    @staticmethod
    def _causal_mask(length: int, device):
        return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, time_dim = input_ids.shape
        embeddings = self.token_embedding(input_ids)
        hidden = self.input_projection(embeddings)
        hidden = hidden + self.positional_encoding(time_dim).to(device=hidden.device, dtype=hidden.dtype).unsqueeze(0)
        causal_mask = self._causal_mask(time_dim, device=input_ids.device)
        padding_mask = ~attention_mask.to(dtype=torch.bool)
        hidden = self.transformer(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=padding_mask,
        )
        output_embeddings = F.relu(self.output_projection(hidden))
        logits = torch.matmul(output_embeddings, self.token_embedding.weight.transpose(0, 1))
        if self.output_bias is not None:
            logits = logits + self.output_bias
        return logits


def train_step(*, model: Model, data, run_ctx, **kwargs):
    input_ids = data["data"].to(torch.long)
    seq_len = data["data:size1"].to(torch.long)
    max_time = input_ids.shape[1]

    if max_time <= 1:
        loss = input_ids.new_tensor(0.0, dtype=torch.float32)
        run_ctx.mark_as_loss(name="ce", loss=loss, inv_norm_factor=torch.tensor(1.0, device=input_ids.device))
        return

    lm_input = input_ids[:, :-1]
    lm_target = input_ids[:, 1:]
    lm_len = torch.clamp(seq_len - 1, min=0)
    lm_time = lm_input.shape[1]
    attention_mask = torch.arange(lm_time, device=input_ids.device)[None, :] < lm_len[:, None]
    labels = torch.where(attention_mask, lm_target, torch.full_like(lm_target, -100))

    logits = model(input_ids=lm_input, attention_mask=attention_mask)
    loss = F.cross_entropy(
        logits.reshape(-1, model.cfg.vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    num_tokens = torch.count_nonzero(labels != -100).to(dtype=torch.float32)
    run_ctx.mark_as_loss(
        name="ce",
        loss=loss,
        inv_norm_factor=torch.clamp(num_tokens, min=1.0),
    )
