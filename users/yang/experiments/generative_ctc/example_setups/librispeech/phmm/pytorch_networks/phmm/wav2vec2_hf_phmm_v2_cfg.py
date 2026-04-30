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
    gradient_checkpointing: bool = False
    aux_loss_layers: Optional[List[int]] = None
    aux_loss_scales: Optional[List[float]] = None
    variance_normalize: bool = False
    variance_path: Optional[tk.Path] = None
    pca_dim: Optional[int] = None
    pca_state_path: Optional[tk.Path] = None
    update_pca_during_training: bool = True
    freeze_output_layers: bool = False
    generative_model: bool = False
    l2_norm: bool = False
    gaussian_mixture_model: bool = False
    num_gaussian_mixtures: int = 1
    gaussian_precision_type: str = "diagonal"  # "diagonal" or "full"
    gaussian_class_dependent_variance: bool = True
    gaussian_init_precision: float = 1.0
    gaussian_precision_floor: float = 1e-6
    gaussian_mean_init_scale: float = 1.0
    gaussian_init_from_variance_stats: bool = False
    gaussian_class_stats_path: Optional[tk.Path] = None
    gaussian_init_from_class_stats: bool = False
    gaussian_init_class_means_only: bool = False
    gaussian_class_stats_mixture_perturb_scale: float = 0.05
    gaussian_use_posterior_training: bool = False
    freeze_gaussian_precision: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
