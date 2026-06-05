from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_type: str  # "cluster" or "vector"
    label_target_size: int
    input_key: str = "data"
    input_vocab_size: int | None = None
    input_dim: int | None = None
    hidden_size: int = 256
    conv_kernel_size: int = 3
    conv_stride: int = 1
    conv_dilation: int = 1
    conv_bias: bool = True
    conv_padding: int | None = None
    input_dropout: float = 0.0
    pre_output_dropout: float = 0.0
    input_time_batch_norm: bool = False
    input_time_batch_norm_affine_init: float = 30.0
    input_residual_linear: bool = False
    min_logit: float | None = -80.0
    aux_loss_layers: list[int] | None = None
    aux_loss_scales: list[float] | None = None
    sampling_type: str = "batch"
    sampling_ratio: float = 0.3
    share_samples: bool = False
    ratio_corrector: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
