import torch
from torch import nn
import torch.nn.functional as F

from .ngram_conv_lm_cfg import ModelConfig


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.left_padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]. Left-only padding keeps the convolution causal.
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class Model(nn.Module):
    """
    Small causal convolutional n-gram language model.

    Architecture:
    token embedding -> causal Conv1d -> ReLU -> linear projection -> ReLU -> output projection.
    The output at position t only depends on tokens up to and including position t.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.cfg.vocab_size}")
        if self.cfg.embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be positive, got {self.cfg.embedding_dim}")
        if self.cfg.conv_channels <= 0:
            raise ValueError(f"conv_channels must be positive, got {self.cfg.conv_channels}")
        if self.cfg.projection_dim <= 0:
            raise ValueError(f"projection_dim must be positive, got {self.cfg.projection_dim}")

        self.token_embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embedding_dim,
            padding_idx=self.cfg.pad_token_id,
        )
        self.causal_conv = CausalConv1d(
            in_channels=self.cfg.embedding_dim,
            out_channels=self.cfg.conv_channels,
            kernel_size=self.cfg.conv_kernel_size,
        )
        self.projection = nn.Linear(self.cfg.conv_channels, self.cfg.projection_dim)
        self.output_projection = nn.Linear(self.cfg.projection_dim, self.cfg.vocab_size)
        self.dropout = nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0.0 else nn.Identity()

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.cfg.pad_token_id].zero_()
        nn.init.kaiming_uniform_(self.causal_conv.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.causal_conv.conv.bias)
        nn.init.kaiming_uniform_(self.projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids.to(torch.long))  # [B, T, E]
        hidden = self.causal_conv(embeddings.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        hidden = self.dropout(F.relu(hidden))
        hidden = self.dropout(F.relu(self.projection(hidden)))
        return self.output_projection(hidden)


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
    valid_mask = torch.arange(lm_time, device=input_ids.device)[None, :] < lm_len[:, None]
    labels = torch.where(valid_mask, lm_target, torch.full_like(lm_target, -100))

    logits = model(input_ids=lm_input)
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
