from dataclasses import dataclass


@dataclass
class ModelConfig:
    hf_model_name: str = "facebook/wav2vec2-large"
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
    return_layer: int = 15

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
