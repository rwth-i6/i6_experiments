from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    label_target_size: int
    hf_model_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True
    freeze_feature_encoder: bool = True
    freeze_encoder: bool = False
    apply_spec_augment: bool = False
    final_dropout: float = 0.0
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    feat_proj_dropout: float = 0.0
    activation_dropout: float = 0.0
    layerdrop: float = 0.0
    mask_time_prob: float = 0.0
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10
    gradient_checkpointing: bool = False
    aux_loss_layers: Optional[List[int]] = None
    aux_loss_scales: Optional[List[float]] = None
    freeze_output_layers: bool = False
    viterbi_training: bool = True
    sampling_type: str = "batch"
    sampling_ratio: float = 0.1
    share_samples: bool = False
    ratio_corrector: float = 1.0
    input_time_batch_norm: bool = False
    input_time_batch_norm_affine_init: float = 30.0
    input_residual_linear: bool = False
    generator_kernel: int = 9
    generator_stride: int = 1
    generator_dilation: int = 1
    generator_bias: bool = True


    @classmethod
    def from_dict(cls, d):
        return cls(**d)
