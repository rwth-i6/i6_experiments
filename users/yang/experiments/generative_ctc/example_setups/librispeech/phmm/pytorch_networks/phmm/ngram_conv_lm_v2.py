import torch
from torch import nn
import torch.nn.functional as F

from .ngram_conv_lm_v2_cfg import ModelConfig


class NoPadConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be positive, got {kernel_size}")
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]. No zero-vector padding here; BOS context is added at token level.
        return self.conv(x)


class Model(nn.Module):
    """
    Causal n-gram convolutional LM with token-level BOS padding.

    The model itself performs no convolution padding. The train step prepends
    conv_kernel_size - 1 BOS token ids before the LmDataset-provided sequence,
    so a context-length-3 model starts with [BOS, BOS, BOS] as its first
    prediction context.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.cfg.vocab_size}")
        if not 0 <= self.cfg.bos_token_id < self.cfg.vocab_size:
            raise ValueError(f"bos_token_id={self.cfg.bos_token_id} outside vocab_size={self.cfg.vocab_size}")

        self.token_embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embedding_dim,
            padding_idx=self.cfg.pad_token_id,
        )
        self.context_conv = NoPadConv1d(
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
        nn.init.kaiming_uniform_(self.context_conv.conv.weight, nonlinearity="relu")
        nn.init.zeros_(self.context_conv.conv.bias)
        nn.init.kaiming_uniform_(self.projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.token_embedding(input_ids.to(torch.long))  # [B, T, E]
        hidden = self.context_conv(embeddings.transpose(1, 2)).transpose(1, 2)  # [B, T-K+1, C]
        hidden = self.dropout(F.relu(hidden))
        hidden = self.dropout(F.relu(self.projection(hidden)))
        return self.output_projection(hidden)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    input_ids = data["data"].to(torch.long)
    seq_len = data["data:size1"].to(torch.long)
    batch_size, max_time = input_ids.shape

    if max_time <= 1:
        loss = input_ids.new_tensor(0.0, dtype=torch.float32)
        run_ctx.mark_as_loss(name="ce", loss=loss, inv_norm_factor=torch.tensor(1.0, device=input_ids.device))
        return

    prefix_len = model.cfg.conv_kernel_size - 1
    if prefix_len > 0:
        bos_prefix = input_ids.new_full((batch_size, prefix_len), model.cfg.bos_token_id)
        lm_input = torch.cat([bos_prefix, input_ids], dim=1)
    else:
        lm_input = input_ids

    logits = model(input_ids=lm_input)  # [B, max_time, V]
    logits = logits[:, : max_time - 1]
    lm_target = input_ids[:, 1:]
    lm_len = torch.clamp(seq_len - 1, min=0)
    lm_time = lm_target.shape[1]
    valid_mask = torch.arange(lm_time, device=input_ids.device)[None, :] < lm_len[:, None]
    labels = torch.where(valid_mask, lm_target, torch.full_like(lm_target, -100))

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
