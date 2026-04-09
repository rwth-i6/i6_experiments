"""
"""

from dataclasses import dataclass, asdict

import torch
from torch import nn
from typing import Callable, List, Literal, Optional, Union

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from i6_experiments.users.yang.experiments.generative_ctc.example_setups.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_0924.generative_ctc_preload import (
    ConformerCTCConfig,
)


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
    activation_str: str = ""
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation_str = d.pop("activation_str")
        if activation_str == "ReLU":
            from torch.nn import ReLU

            activation = ReLU()
        else:
            assert False, "Unsupported activation %s" % d["activation_str"]
        d["activation"] = activation
        return VGG4LayerActFrontendV1Config(**d)

@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class ConformerPosEmbConfig(ModelConfiguration):
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float


@dataclass
class ModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    pos_emb_config: ConformerPosEmbConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    mhsa_with_bias: bool
    conv_kernel_size: int
    final_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]
    num_mixtures: Optional[int] = 1
    norm_vector: Optional[bool] = False
    freeze_encoder: Optional[bool] = True
    blank_pen: Optional[float] = -1e4

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        return ModelConfig(**d)


def convert_conformer_ctc_to_model_config(cfg: ConformerCTCConfig) -> ModelConfig:
    """Convert a ConformerCTCConfig (preload) to the ModelConfig style used here."""
    # Feature extraction maps 1:1
    feature_extraction_config = cfg.feature_extraction_cfg

    # Conformer-related configs
    conformer_cfg = cfg.conformer_cfg
    block_cfg = conformer_cfg.block_cfg
    ff_cfg = block_cfg.ff_cfg
    mhsa_cfg = block_cfg.mhsa_cfg
    conv_cfg = block_cfg.conv_cfg

    # Frontend may be wrapped in ModuleFactoryV1; try to get the underlying cfg
    frontend_config = getattr(conformer_cfg.frontend, "cfg", conformer_cfg.frontend)
    from torch.nn import ReLU
    if isinstance(frontend_config.activation, ReLU):
        frontend_config_dict = asdict(frontend_config)
        frontend_config_dict['activation'] = None
        frontend_config_dict['activation_str'] = "ReLU"
        frontend_config = VGG4LayerActFrontendV1Config_mod(**frontend_config_dict)
    else:
        assert False, "Unsupported activation %s" % frontend_config.activation

    # SpecAug: best-effort mapping from the dict used in preload to compact SpecaugConfig
    specaug_args = getattr(cfg, "specaug_args", {}) or {}
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=specaug_args.get("time_max_mask_per_n_frames", 25),
        max_dim_time=specaug_args.get("time_mask_max_size", 20),
        num_repeat_feat=specaug_args.get("freq_max_num_masks", specaug_args.get("freq_min_num_masks", 2)),
        max_dim_feat=specaug_args.get("freq_mask_max_size", 16),
    )

    # Positional embedding subset from MHSA config
    pos_emb_config = ConformerPosEmbConfig(
        learnable_pos_emb=getattr(mhsa_cfg, "learnable_pos_emb", False),
        rel_pos_clip=getattr(mhsa_cfg, "rel_pos_clip", None),
        with_linear_pos=getattr(mhsa_cfg, "with_linear_pos", True),
        with_pos_bias=getattr(mhsa_cfg, "with_pos_bias", True),
        separate_pos_emb_per_head=getattr(mhsa_cfg, "separate_pos_emb_per_head", True),
        pos_emb_dropout=getattr(mhsa_cfg, "pos_emb_dropout", 0.0),
    )

    # Aux CTC loss layers/scales
    aux_losses = getattr(cfg, "aux_losses", {}) or {}
    if isinstance(aux_losses, dict):
        # keys are 1-based layer indices as strings in the preload file
        sorted_items = sorted(((int(k), v) for k, v in aux_losses.items()), key=lambda x: x[0])
        aux_layers = [k for k, _ in sorted_items]
        aux_scales = [float(v) for _, v in sorted_items]
    else:
        aux_layers, aux_scales = None, None

    # Module list from block config; scales default to Conformer convention
    module_list = list(getattr(block_cfg, "modules", ["ff", "mhsa", "conv", "ff"]))
    default_scale_by_module = {"ff": 0.5}
    module_scales = [default_scale_by_module.get(m, 1.0) for m in module_list]

    # Dropout broadcast axes if present at top-level; otherwise leave None
    dropout_broadcast_axes = getattr(cfg, "dropout_broadcast_axes", None)

    return ModelConfig(
        feature_extraction_config=feature_extraction_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        pos_emb_config=pos_emb_config,
        specauc_start_epoch=getattr(cfg, "specauc_start_epoch", 0),
        label_target_size=cfg.target_size,
        conformer_size=getattr(ff_cfg, "input_dim"),
        num_layers=getattr(conformer_cfg, "num_layers"),
        num_heads=getattr(mhsa_cfg, "num_att_heads"),
        ff_dim=getattr(ff_cfg, "hidden_dim"),
        att_weights_dropout=getattr(mhsa_cfg, "att_weights_dropout", 0.0),
        conv_dropout=getattr(conv_cfg, "dropout", 0.0),
        ff_dropout=getattr(ff_cfg, "dropout", 0.0),
        mhsa_dropout=getattr(mhsa_cfg, "dropout", 0.0),
        mhsa_with_bias=getattr(mhsa_cfg, "with_bias", True),
        conv_kernel_size=getattr(conv_cfg, "kernel_size"),
        final_dropout=getattr(cfg, "final_dropout", 0.0),
        dropout_broadcast_axes=dropout_broadcast_axes,
        module_list=module_list,
        module_scales=module_scales,
        aux_ctc_loss_layers=aux_layers,
        aux_ctc_loss_scales=aux_scales,
        norm_vector=getattr(cfg, "norm_vector", False),
        freeze_encoder=getattr(cfg, "freeze_encoder", True)
    )
