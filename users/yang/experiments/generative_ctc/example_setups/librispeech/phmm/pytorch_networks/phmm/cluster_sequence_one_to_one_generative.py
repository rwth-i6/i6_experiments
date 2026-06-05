import torch
from torch import nn

from .cluster_sequence_one_to_one_generative_cfg import ModelConfig
from ...loss.generative_loss import generative_nce
from ...loss.unsupervised_full_sum_loss import one_to_one_full_sum


class Model(nn.Module):
    """
    Segment-cluster one-to-one generative pHMM model.

    Input is a sequence of cluster ids. The model predicts phoneme log-sigmoid
    scores with two bidirectional Conv1d layers. The LM table keeps silence as a
    normal emitted label for one-to-one full-sum training.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.input_vocab_size <= 0:
            raise ValueError(f"input_vocab_size must be positive, got {self.cfg.input_vocab_size}")
        if self.cfg.label_target_size <= 0:
            raise ValueError(f"label_target_size must be positive, got {self.cfg.label_target_size}")
        if self.cfg.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.cfg.hidden_size}")
        if self.cfg.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {self.cfg.conv_kernel_size}")
        if self.cfg.conv_stride <= 0:
            raise ValueError(f"conv_stride must be positive, got {self.cfg.conv_stride}")
        if self.cfg.conv_dilation <= 0:
            raise ValueError(f"conv_dilation must be positive, got {self.cfg.conv_dilation}")

        self.embedding = nn.Embedding(self.cfg.input_vocab_size, self.cfg.hidden_size)
        self.conv_padding = (
            self.cfg.conv_padding
            if self.cfg.conv_padding is not None
            else (self.cfg.conv_dilation * (self.cfg.conv_kernel_size - 1)) // 2
        )
        self.conv1 = nn.Conv1d(
            self.cfg.hidden_size,
            self.cfg.hidden_size,
            kernel_size=self.cfg.conv_kernel_size,
            stride=self.cfg.conv_stride,
            dilation=self.cfg.conv_dilation,
            padding=self.conv_padding,
            bias=self.cfg.conv_bias,
        )
        self.activation = nn.LeakyReLU(negative_slope=self.cfg.leaky_relu_negative_slope)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.conv2 = nn.Conv1d(
            self.cfg.hidden_size,
            self.cfg.label_target_size,
            kernel_size=self.cfg.conv_kernel_size,
            stride=self.cfg.conv_stride,
            dilation=self.cfg.conv_dilation,
            padding=self.conv_padding,
            bias=self.cfg.conv_bias,
        )
        self.register_buffer("lm_table", self._load_lm_table(self.cfg.lm_table_path), persistent=False)

    @staticmethod
    def _get_path(path):
        try:
            from returnn.util.basic import cf

            return cf(path)
        except Exception:
            return str(path)

    def _load_lm_table(self, path):
        table = torch.load(self._get_path(path), map_location="cpu")
        if isinstance(table, dict):
            for key in ("log_probs", "lm_table", "scores", "data"):
                if key in table:
                    table = table[key]
                    break
        table = torch.as_tensor(table, dtype=torch.float32)
        expected_shape = (self.cfg.lm_vocab_size ** self.cfg.lm_context_length, self.cfg.lm_vocab_size)
        if tuple(table.shape) != expected_shape:
            raise ValueError(f"Expected LM table shape {expected_shape}, got {tuple(table.shape)}")
        return table

    def _apply_conv_length(self, lengths: torch.Tensor) -> torch.Tensor:
        return torch.div(
            lengths
            + 2 * self.conv_padding
            - self.cfg.conv_dilation * (self.cfg.conv_kernel_size - 1)
            - 1,
            self.cfg.conv_stride,
            rounding_mode="floor",
        ) + 1

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        output_lengths = input_lengths.to(dtype=torch.long)
        output_lengths = self._apply_conv_length(output_lengths)
        output_lengths = torch.clamp(output_lengths, min=0)
        output_lengths = self._apply_conv_length(output_lengths)
        return torch.clamp(output_lengths, min=0)

    def forward(self, input_ids: torch.Tensor, input_ids_len: torch.Tensor | None = None):
        input_ids = input_ids.to(dtype=torch.long)
        if input_ids.numel() > 0:
            min_id = int(input_ids.min().detach().cpu())
            max_id = int(input_ids.max().detach().cpu())
            if min_id < 0 or max_id >= self.cfg.input_vocab_size:
                raise ValueError(
                    f"input ids out of range: min={min_id}, max={max_id}, input_vocab_size={self.cfg.input_vocab_size}"
                )

        x = self.embedding(input_ids).transpose(1, 2)
        x = self.conv1(x)
        x = self.dropout(self.activation(x))
        logits = self.conv2(x).transpose(1, 2)
        if self.cfg.min_logit is not None:
            logits = self.cfg.min_logit + torch.nn.functional.softplus(logits - self.cfg.min_logit)
        log_probs = torch.nn.functional.logsigmoid(logits)
        if input_ids_len is None:
            return log_probs
        return log_probs, self._get_output_lengths(input_ids_len)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    input_ids = data["data"].to(torch.long)
    input_ids_len = data["data:size1"].to(torch.long)
    with torch.enable_grad():
        log_probs, output_len = model(input_ids, input_ids_len)
        output_len = torch.clamp(output_len, max=log_probs.shape[1])

        dp_loss = -one_to_one_full_sum(
            log_probs=log_probs.float(),
            seq_len=output_len,
            lm_table=model.lm_table,
            lm_vocab_size=model.cfg.lm_vocab_size,
            lm_context_length=model.cfg.lm_context_length,
            k=model.cfg.beam_size,
            use_lm_silence_score=True,
            lm_scale=model.cfg.lm_scale,
            am_scale=model.cfg.am_scale,
        )
        soft_target = -torch.autograd.grad(dp_loss, log_probs, retain_graph=True)[0].detach()
        loss = generative_nce(
            log_probs,
            soft_target.to(log_probs.dtype),
            sampling_type=model.cfg.sampling_type,
            seq_len=output_len,
            sampling_ratio=model.cfg.sampling_ratio,
            share_samples=model.cfg.share_samples,
            ratio_corrector=model.cfg.ratio_corrector,
        )
    inv_norm_factor = torch.clamp(output_len.sum().to(dtype=torch.float32), min=1.0)
    run_ctx.mark_as_loss(
        name="generative_nce_one_to_one",
        loss=loss.sum(),
        inv_norm_factor=inv_norm_factor,
    )
