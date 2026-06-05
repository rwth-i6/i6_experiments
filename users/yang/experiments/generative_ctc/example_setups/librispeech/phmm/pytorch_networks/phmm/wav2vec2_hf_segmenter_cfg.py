from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    hf_model_name: str = "facebook/wav2vec2-base"
    pretrained: bool = True
    freeze_feature_encoder: bool = True
    freeze_encoder: bool = True
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
    return_layer: int = -1
    segmenter_num_blocks: int = 1
    segmenter_channels: int = 256
    segmenter_kernel_size: int = 4
    segmenter_stride: int = 1
    segmenter_dilation: int = 1
    segmenter_padding: Optional[int] = None
    segmenter_bias: bool = True
    segmenter_dropout: float = 0.0
    leaky_relu_negative_slope: float = 0.01
    contrastive_num_samples: int = 5
    contrastive_temperature: float = 1.0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
