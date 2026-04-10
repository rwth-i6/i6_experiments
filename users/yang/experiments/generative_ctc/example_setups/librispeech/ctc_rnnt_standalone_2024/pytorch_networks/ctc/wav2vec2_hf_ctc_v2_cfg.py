#compare to v1: add options to do variance normalization, PCA, and l2-norm

from dataclasses import dataclass
from typing import List, Optional

from sisyphus import tk
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
    ctc_loss_reduction: str = "sum"
    pad_token_id: Optional[int] = None
    blank_index: Optional[int] = None
    gradient_checkpointing: bool = False
    aux_ctc_loss_layers: Optional[List[int]] = None
    aux_ctc_loss_scales: Optional[List[float]] = None
    variance_normalize: bool = False
    variance_path: Optional[tk.Path] = None
    pca_dim: Optional[int] = None
    pca_state_path: Optional[tk.Path] = None
    update_pca_during_training: bool = True
    freeze_output_layers: bool = False
    generative_model: bool = False
    l2_norm: bool = False


    @classmethod
    def from_dict(cls, d):
        return cls(**d)
