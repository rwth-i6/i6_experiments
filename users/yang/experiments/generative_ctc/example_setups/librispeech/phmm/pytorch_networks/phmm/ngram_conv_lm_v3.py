import torch
from torch import nn
import torch.nn.functional as F

from .ngram_conv_lm_v3_cfg import ModelConfig


class Model(nn.Module):
    """
    Multi-layer causal n-gram convolutional LM with token-level BOS padding.

    The convolution stack itself uses padding=0, as in v2. The train step
    prepends context_length - 1 BOS token ids, where:

        context_length = 1 + sum(dilation * (kernel_size - 1) for kernel_size, dilation in layers)

    Thus a stack with kernels (4, 5) has context_length 8.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.cfg.vocab_size}")
        if not 0 <= self.cfg.bos_token_id < self.cfg.vocab_size:
            raise ValueError(f"bos_token_id={self.cfg.bos_token_id} outside vocab_size={self.cfg.vocab_size}")
        if not self.cfg.conv_kernel_sizes:
            raise ValueError("conv_kernel_sizes must not be empty")
        if self.cfg.num_conv_layers is not None and self.cfg.num_conv_layers != len(self.cfg.conv_kernel_sizes):
            raise ValueError(
                f"num_conv_layers={self.cfg.num_conv_layers} does not match "
                f"len(conv_kernel_sizes)={len(self.cfg.conv_kernel_sizes)}"
            )
        conv_dilations = self.cfg.conv_dilations
        if conv_dilations is None:
            conv_dilations = tuple(1 for _ in self.cfg.conv_kernel_sizes)
        if len(conv_dilations) != len(self.cfg.conv_kernel_sizes):
            raise ValueError(
                f"len(conv_dilations)={len(conv_dilations)} does not match "
                f"len(conv_kernel_sizes)={len(self.cfg.conv_kernel_sizes)}"
            )
        for kernel_size in self.cfg.conv_kernel_sizes:
            if kernel_size <= 0:
                raise ValueError(f"all conv kernel sizes must be positive, got {self.cfg.conv_kernel_sizes}")
        for dilation in conv_dilations:
            if dilation <= 0:
                raise ValueError(f"all conv dilations must be positive, got {conv_dilations}")

        self.conv_dilations = conv_dilations
        self.context_length = 1 + sum(
            dilation * (kernel_size - 1)
            for kernel_size, dilation in zip(self.cfg.conv_kernel_sizes, self.conv_dilations)
        )
        self.token_embedding = nn.Embedding(
            self.cfg.vocab_size,
            self.cfg.embedding_dim,
            padding_idx=self.cfg.pad_token_id,
        )

        conv_layers = []
        in_channels = self.cfg.embedding_dim
        for kernel_size, dilation in zip(self.cfg.conv_kernel_sizes, self.conv_dilations):
            conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=self.cfg.conv_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=0,
                )
            )
            in_channels = self.cfg.conv_channels
        self.conv_layers = nn.ModuleList(conv_layers)
        self.activation = nn.LeakyReLU(negative_slope=self.cfg.leaky_relu_negative_slope)
        self.dropout = nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0.0 else nn.Identity()
        self.projection = nn.Linear(self.cfg.conv_channels, self.cfg.projection_dim)
        self.output_projection = nn.Linear(self.cfg.projection_dim, self.cfg.vocab_size)

        self._init_parameters()

    def _init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.cfg.pad_token_id].zero_()
        for conv in self.conv_layers:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="leaky_relu", a=self.cfg.leaky_relu_negative_slope)
            nn.init.zeros_(conv.bias)
        nn.init.kaiming_uniform_(self.projection.weight, nonlinearity="relu")
        nn.init.zeros_(self.projection.bias)
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.token_embedding(input_ids.to(torch.long)).transpose(1, 2)  # [B, E, T]
        for conv in self.conv_layers:
            hidden = self.dropout(self.activation(conv(hidden)))
        hidden = hidden.transpose(1, 2)
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

    prefix_len = model.context_length - 1
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
