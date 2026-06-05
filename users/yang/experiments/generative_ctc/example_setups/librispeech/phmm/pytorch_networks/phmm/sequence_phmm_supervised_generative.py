import torch
from torch import nn

from i6_models.parts.rasr_fsa import RasrFsaBuilderV2

from .sequence_phmm_supervised_generative_cfg import ModelConfig
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.phmm.loss.generative_loss import (
    generative_nce,
)


class RasrFsaBuilderOrth(RasrFsaBuilderV2):
    def build_single(self, orth: str):
        return self.builder.build_by_orthography(orth)


class Model(nn.Module):
    """
    HDF-input supervised generative pHMM model.

    The model input comes from an HDF stream, while the loss uses raw orthography
    labels from the paired OggZip stream to build RASR FSAs.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        if self.cfg.input_type not in {"cluster", "vector"}:
            raise ValueError(f"input_type must be 'cluster' or 'vector', got {self.cfg.input_type!r}")
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

        if self.cfg.input_type == "cluster":
            if self.cfg.input_vocab_size is None or self.cfg.input_vocab_size <= 0:
                raise ValueError("input_vocab_size must be set and positive for cluster input")
            self.embedding = nn.Embedding(self.cfg.input_vocab_size, self.cfg.hidden_size)
            input_channels = self.cfg.hidden_size
            self.input_time_batch_norm = None
            self.input_residual_linear = None
        else:
            if self.cfg.input_dim is None or self.cfg.input_dim <= 0:
                raise ValueError("input_dim must be set and positive for vector input")
            self.embedding = None
            input_channels = self.cfg.input_dim
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
        self.input_dropout = nn.Dropout(self.cfg.input_dropout)
        self.generator = nn.Conv1d(
            input_channels,
            self.cfg.hidden_size,
            kernel_size=self.cfg.conv_kernel_size,
            stride=self.cfg.conv_stride,
            dilation=self.cfg.conv_dilation,
            padding=self.conv_padding,
            bias=self.cfg.conv_bias,
        )
        self.pre_output_dropout = nn.Dropout(self.cfg.pre_output_dropout)
        self.output_projection = nn.Linear(self.cfg.hidden_size, self.cfg.label_target_size)

        self.return_layers = self.cfg.aux_loss_layers or [0]
        self.scales = self.cfg.aux_loss_scales or [1.0]
        if len(self.return_layers) != len(self.scales):
            raise ValueError("aux_loss_layers and aux_loss_scales must have the same length")

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

    def _prepare_input(self, model_input: torch.Tensor) -> torch.Tensor:
        if self.cfg.input_type == "cluster":
            input_ids = model_input.to(dtype=torch.long)
            if input_ids.numel() > 0:
                min_id = int(input_ids.min().detach().cpu())
                max_id = int(input_ids.max().detach().cpu())
                if min_id < 0 or max_id >= self.cfg.input_vocab_size:
                    raise ValueError(
                        f"input ids out of range: min={min_id}, max={max_id}, "
                        f"input_vocab_size={self.cfg.input_vocab_size}"
                    )
            return self.embedding(input_ids)

        features = model_input.to(dtype=torch.float32)
        if features.shape[-1] != self.cfg.input_dim:
            raise ValueError(f"Expected input dim {self.cfg.input_dim}, got {features.shape[-1]}")
        if self.input_time_batch_norm is not None:
            features = self.input_time_batch_norm(features.transpose(1, 2)).transpose(1, 2)
        if self.input_residual_linear is not None:
            features = features + self.input_residual_linear(features)
        return features

    def forward(self, model_input: torch.Tensor, model_input_len: torch.Tensor):
        x = self._prepare_input(model_input)
        x = self.input_dropout(x)
        hidden = self.generator(x.transpose(1, 2)).transpose(1, 2)
        hidden = self.pre_output_dropout(hidden)
        logits = self.output_projection(hidden)
        if self.cfg.min_logit is not None:
            logits = self.cfg.min_logit + torch.nn.functional.softplus(logits - self.cfg.min_logit)
        log_probs = torch.nn.functional.logsigmoid(logits)
        output_lengths = self._get_output_lengths(model_input_len)
        output_lengths = torch.clamp(output_lengths, max=log_probs.shape[1])
        return [log_probs for _ in self.return_layers], output_lengths

    def get_log_probs_by_layer(self, log_probs_list, decode_layer_index=None):
        if decode_layer_index is None:
            return log_probs_list[-1]
        decode_layer_pos = self.return_layers.index(decode_layer_index)
        return log_probs_list[decode_layer_pos]


class GenPhmmTrainStep:
    def __init__(
        self,
        fsa_exporter_config_path: str,
        transition_scale: float = 1.0,
        zero_infinity: bool = True,
        label_smoothing_scale: float = 0.0,
    ):
        self.fsa_builder = RasrFsaBuilderOrth(fsa_exporter_config_path, transition_scale)
        self.zero_infinity = zero_infinity
        self.label_smoothing_scale = label_smoothing_scale

    def __call__(self, *, model: Model, data, run_ctx, **kwargs):
        from i6_native_ops.fbw2 import fbw2_loss

        model_input = data[model.cfg.input_key]
        model_input_len = data[f"{model.cfg.input_key}:size1"].to(torch.long)

        labels_raw = data["labels"]
        labels_len = data["labels:size1"]
        labels = [
            bytes(labels_raw[i, : labels_len[i]].cpu().tolist()).decode("utf8") + " "
            for i in range(labels_raw.shape[0])
        ]

        seq_tags = data.get("seq_tag")
        if seq_tags is not None:
            if hasattr(seq_tags, "tolist"):
                seq_tags = seq_tags.tolist()
            seq_tags = [
                tag.decode("utf8") if isinstance(tag, (bytes, bytearray)) else str(tag)
                for tag in seq_tags
            ]

        with torch.enable_grad():
            logprobs_list, output_len = model(model_input=model_input, model_input_len=model_input_len)
            output_len_for_loss = output_len.to(dtype=torch.int32).contiguous()

            for logprobs, layer_index, scale in zip(logprobs_list, model.return_layers, model.scales):
                target_fsa = self.fsa_builder.build_batch(labels).to(logprobs.device)
                logprobs = logprobs.float()
                ml_loss = fbw2_loss(logprobs, target_fsa, output_len_for_loss)
                inv_norm_factor = torch.sum(output_len)

                inf_mask = torch.isinf(ml_loss)
                if torch.any(inf_mask):
                    inf_indices = torch.nonzero(inf_mask, as_tuple=False).flatten().tolist()
                    print(
                        f"phmm_loss_layer{layer_index}: detected {len(inf_indices)} inf losses "
                        f"in batch of size {len(labels)}"
                    )
                    for idx in inf_indices:
                        msg = f"  seq_idx={idx} model_frames={int(output_len[idx])}"
                        if seq_tags is not None:
                            msg += f" seq_tag={seq_tags[idx]!r}"
                        msg += f" orth={labels[idx]!r}"
                        print(msg)

                if self.zero_infinity:
                    valid_mask = ~inf_mask
                    ml_loss = torch.where(valid_mask, ml_loss, torch.zeros_like(ml_loss))
                    inv_norm_factor = torch.sum(output_len[valid_mask])
                else:
                    valid_mask = torch.ones_like(inf_mask, dtype=torch.bool)

                ml_loss = torch.sum(ml_loss)
                soft_target = -torch.autograd.grad(ml_loss, logprobs)[0].detach()
                valid_seq_len = torch.where(
                    valid_mask,
                    output_len,
                    torch.zeros_like(output_len),
                )
                loss = generative_nce(
                    logprobs,
                    soft_target.to(logprobs.dtype),
                    sampling_type=model.cfg.sampling_type,
                    seq_len=valid_seq_len,
                    sampling_ratio=model.cfg.sampling_ratio,
                    share_samples=model.cfg.share_samples,
                    ratio_corrector=model.cfg.ratio_corrector,
                )
                run_ctx.mark_as_loss(
                    name=f"phmm_loss_layer{layer_index}",
                    loss=loss.sum(),
                    scale=scale,
                    inv_norm_factor=inv_norm_factor,
                )


_phmm_train_step = None


def train_step(*, model: Model, data, run_ctx, **kwargs):
    global _phmm_train_step
    if _phmm_train_step is None:
        _phmm_train_step = GenPhmmTrainStep(
            fsa_exporter_config_path=kwargs["fsa_exporter_config_path"],
            transition_scale=kwargs.get("transition_scale", 1.0),
            zero_infinity=kwargs.get("zero_infinity", True),
            label_smoothing_scale=kwargs.get("label_smoothing_scale", 0.0),
        )
    _phmm_train_step(model=model, data=data, run_ctx=run_ctx, **kwargs)
