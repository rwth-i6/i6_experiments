import torch
from torch import nn

from .vector_sequence_phmm_generative_cfg import ModelConfig
from ...loss.generative_loss import generative_nce
from ...loss.unsupervised_full_sum_loss import no_concecutive_full_sum


class Model(nn.Module):
    """
    Vector-input generative pHMM model.

    Input is a sequence of feature vectors. The features directly replace the
    wav2vec2 hidden states, and the head follows wav2vec2_hf_phmm_generative:
    optional time batchnorm/residual linear -> dropout -> Conv1d generator ->
    log-sigmoid.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.cfg.input_dim}")
        if self.cfg.label_target_size <= 0:
            raise ValueError(f"label_target_size must be positive, got {self.cfg.label_target_size}")
        if self.cfg.conv_kernel_size <= 0:
            raise ValueError(f"conv_kernel_size must be positive, got {self.cfg.conv_kernel_size}")
        if self.cfg.conv_stride <= 0:
            raise ValueError(f"conv_stride must be positive, got {self.cfg.conv_stride}")
        if self.cfg.conv_dilation <= 0:
            raise ValueError(f"conv_dilation must be positive, got {self.cfg.conv_dilation}")

        if self.cfg.input_time_batch_norm:
            self.input_time_batch_norm = nn.BatchNorm1d(self.cfg.input_dim)
            nn.init.constant_(self.input_time_batch_norm.weight, self.cfg.input_time_batch_norm_affine_init)
            nn.init.zeros_(self.input_time_batch_norm.bias)
        else:
            self.input_time_batch_norm = None
        if self.cfg.input_residual_linear:
            self.input_residual_linear = nn.Linear(self.cfg.input_dim, self.cfg.input_dim)
        else:
            self.input_residual_linear = None
        self.conv_padding = (
            self.cfg.conv_padding
            if self.cfg.conv_padding is not None
            else (self.cfg.conv_dilation * (self.cfg.conv_kernel_size - 1)) // 2
        )
        self.generator = nn.Conv1d(
            self.cfg.input_dim,
            self.cfg.label_target_size,
            kernel_size=self.cfg.conv_kernel_size,
            stride=self.cfg.conv_stride,
            dilation=self.cfg.conv_dilation,
            padding=self.conv_padding,
            bias=self.cfg.conv_bias,
        )
        self.dropout = nn.Dropout(self.cfg.dropout)
        if (self.cfg.lm_table_path is None) == (self.cfg.lm_checkpoint_path is None):
            raise ValueError("Exactly one of lm_table_path or lm_checkpoint_path must be set")
        if self.cfg.lm_table_path is not None:
            self.register_buffer("lm_table", self._load_lm_table(self.cfg.lm_table_path), persistent=False)
            self.lm_network = None
        else:
            self.lm_table = None
            self.lm_network = self._load_lm_network(
                checkpoint_path=self.cfg.lm_checkpoint_path,
                model_config_dict=self.cfg.lm_model_config_dict,
            )

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
        table = table.clone()
        table[:, 0] = -50000.0
        return table

    def _load_lm_network(self, checkpoint_path, model_config_dict):
        if model_config_dict is None:
            raise ValueError("lm_model_config_dict must be set when lm_checkpoint_path is used")
        from .ngram_conv_lm_v3 import Model as NgramConvLmV3

        lm_network = NgramConvLmV3(model_config_dict=model_config_dict)
        checkpoint = torch.load(self._get_path(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if any(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
        lm_network.load_state_dict(state_dict, strict=True)
        lm_network.eval()
        for parameter in lm_network.parameters():
            parameter.requires_grad_(False)
        if lm_network.cfg.vocab_size != self.cfg.lm_vocab_size:
            raise ValueError(f"LM vocab mismatch: config has {self.cfg.lm_vocab_size}, LM has {lm_network.cfg.vocab_size}")
        if lm_network.context_length != self.cfg.lm_context_length:
            raise ValueError(
                f"LM context mismatch: config has {self.cfg.lm_context_length}, LM has {lm_network.context_length}"
            )
        return lm_network

    def _get_output_lengths(self, input_lengths: torch.Tensor) -> torch.Tensor:
        input_lengths = input_lengths.to(dtype=torch.long)
        output_lengths = torch.div(
            input_lengths
            + 2 * self.conv_padding
            - self.cfg.conv_dilation * (self.cfg.conv_kernel_size - 1)
            - 1,
            self.cfg.conv_stride,
            rounding_mode="floor",
        ) + 1
        return torch.clamp(output_lengths, min=0)

    def forward(self, features: torch.Tensor, features_len: torch.Tensor | None = None):
        features = features.to(dtype=torch.float32)
        if features.shape[-1] != self.cfg.input_dim:
            raise ValueError(f"Expected input dim {self.cfg.input_dim}, got {features.shape[-1]}")

        x = features
        if self.input_time_batch_norm is not None:
            x = self.input_time_batch_norm(x.transpose(1, 2)).transpose(1, 2)
        if self.input_residual_linear is not None:
            x = x + self.input_residual_linear(x)
        x = self.dropout(x)
        logits = self.generator(x.transpose(1, 2)).transpose(1, 2)
        if self.cfg.min_logit is not None:
            logits = self.cfg.min_logit + torch.nn.functional.softplus(logits - self.cfg.min_logit)
        log_probs = torch.nn.functional.logsigmoid(logits)
        if features_len is None:
            return log_probs
        return log_probs, self._get_output_lengths(features_len)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    from returnn.config import get_global_config

    config = get_global_config()
    loop_penalty = config.float("loop_penalty", 1.0)
    features = data["data"]
    features_len = data["data:size1"].to(torch.long)
    with torch.enable_grad():
        log_probs, output_len = model(features, features_len)
        output_len = torch.clamp(output_len, max=log_probs.shape[1])

        dp_loss = -no_concecutive_full_sum(
            log_probs=log_probs.float(),
            seq_len=output_len,
            lm_table=model.lm_table,
            lm_network=model.lm_network,
            lm_vocab_size=model.cfg.lm_vocab_size,
            lm_context_length=model.cfg.lm_context_length,
            k=model.cfg.beam_size,
            lm_scale=model.cfg.lm_scale,
            am_scale=model.cfg.am_scale,
            loop_penalty=loop_penalty,
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
        name="generative_nce_no_consecutive",
        loss=loss.sum(),
        inv_norm_factor=inv_norm_factor,
    )
    run_ctx.mark_as_loss(
        name="dp_loss_no_consecutive",
        loss=dp_loss.detach(),
        inv_norm_factor=inv_norm_factor,
        as_error=True,
    )
