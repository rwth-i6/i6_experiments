from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_vocab_size: int
    label_target_size: int
    lm_table_path: str | None = None
    lm_checkpoint_path: str | None = None
    lm_model_config_dict: dict | None = None
    hidden_size: int = 256
    conv_kernel_size: int = 3
    conv_stride: int = 1
    conv_dilation: int = 1
    conv_bias: bool = True
    conv_padding: int | None = None
    dropout: float = 0.0
    min_logit: float | None = -80.0
    lm_vocab_size: int = 41
    lm_context_length: int = 3
    beam_size: int = 200
    lm_scale: float = 0.6
    am_scale: float = 1.0
    sampling_type: str = "batch"
    sampling_ratio: float = 0.3
    share_samples: bool = False
    ratio_corrector: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
