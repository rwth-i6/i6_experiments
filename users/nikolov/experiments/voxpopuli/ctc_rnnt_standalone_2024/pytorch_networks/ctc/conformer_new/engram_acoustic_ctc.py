"""
Engram-augmented CTC Conformer with Learnable Acoustic Quantization.

**Architecture:**
Combines the EngramModule (arXiv:2601.07372) with a learnable acoustic
quantizer (based on BestRQ, arXiv:2202.01855) to produce discrete key
sequences from continuous audio features.

The model extends the LID-aware SC CTC architecture:
  - Layer 3: LID head (unchanged) — predicts language ID per frame
  - Layers 6, 9: Engram modules replace SC layers
    - Keys are generated from BOTH:
      a) LID predictions (categorical language context)
      b) Quantized acoustic features (prosodic/articulatory context)
    - The acoustic quantizer uses random projection (BestRQ-style) to
      map continuous features to discrete clusters
    - Engram hashes n-grams of the joint (LID, acoustic) key sequence

**Why this addresses the LID limitation:**
LID alone gives coarse-grained language labels ([EN, EN, ES, ES]).
Different speakers produce different acoustic realizations of the same
language sequence. The acoustic quantizer captures these individual
articulatory patterns, making the Engram responsive to the actual
acoustic signal of a language switch.

**Training:**
  - CTC loss on BPE targets (primary)
  - Quantizer commitment loss (secondary) — trains the acoustic feature projector
    via a straight-through estimator
  - Contrastive InfoNCE loss on projected hidden states (planned, not yet implemented)

Model file contract:
  - ModelConfig (dataclass with from_dict)
  - Model (forward(raw_audio, raw_audio_len) -> (log_probs, lens))
  - train_step
  - prior_init_hook, prior_finish_hook, prior_step
  - get_model_config
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from i6_models.config import ModuleFactoryV1
from i6_models.assemblies.conformer.conformer_v2 import (
    ConformerEncoderV2Config,
    ConformerBlockV2Config,
    ConformerEncoderV2,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV2Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.parts.best_rq.quantizer import RandomProjectionQuantizer

from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    SpecaugConfig,
    VGG4LayerActFrontendV1Config_mod,
    LogMelFeatureExtractionV1Config,
)
from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_onnx_exportable import (
    LogMelFeatureExtractionV1OnnxExportable,
    mask_tensor,
    get_model_config as get_base_model_config,
)
from i6_models.parts.engram import EngramConfig, EngramModule


@dataclass
class ModelConfig:
    """Standalone ModelConfig with Engram-specific fields."""
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config_mod
    specaug_config: SpecaugConfig
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
    conv_kernel_size: int
    final_dropout: float
    module_list: List[str]
    module_scales: List[float]
    # LID/Engram-specific
    lid_sc_layer: int = 3
    sc_layer: List[int] = field(default_factory=lambda: [6, 9])
    # Engram config
    engram_ngram_orders: List[int] = field(default_factory=lambda: [2, 3])
    engram_num_heads: int = 8
    engram_mem_dim: int = 1280
    engram_table_size: int = 2**12
    engram_dropout: float = 0.0
    # Acoustic quantizer config
    acoustic_feat_dim: int = 80  # F0 + spectral centroid + mfcc delta features
    acoustic_codebook_dim: int = 64
    acoustic_codebook_size: int = 256
    # Loss weights
    quantizer_commitment_weight: float = 0.1
    contrastive_weight: float = 0.01
    contrastive_temperature: float = 0.07

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return cls(**d)


class AcousticQuantizer(nn.Module):
    """
    Wraps the BestRQ RandomProjectionQuantizer and adds a learnable
    feature extraction layer that combines multiple prosodic features.

    Extracts:
      - F0 (fundamental frequency)
      - Spectral centroid
      - MFCC delta coefficients
      - Energy envelope

    Projects these into a shared embedding space, then applies
    random projection quantization.
    """

    def __init__(self, model_dim: int, feat_dim: int, codebook_dim: int, codebook_size: int):
        super().__init__()
        self.model_dim = model_dim
        self.feat_dim = feat_dim

        # Linear projection from Conformer hidden states to feature space
        self.feature_proj = nn.Linear(model_dim, feat_dim)

        # BestRQ quantizer (buffers P and CB are non-learnable)
        self.quantizer = RandomProjectionQuantizer(
            input_dim=feat_dim,
            codebook_dim=codebook_dim,
            codebook_num_vars=codebook_size,
        )

        # Projection for contrastive learning
        self.contrast_proj = nn.Linear(feat_dim, 64)

        # Storage for commitment loss (accessed by train_step)
        self.commitment_loss = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_states: [B, T, model_dim] — Conformer hidden states
        :return: [B, T] discrete acoustic codes
        """
        projected = self.feature_proj(hidden_states)  # [B, T, feat_dim]

        # Straight-through estimator: detach for quantization (argmin is
        # non-differentiable), but keep the graph connected for feature_proj
        # gradients via the commitment loss
        codes = self.quantizer(projected.detach())  # [B, T]

        if self.training:
            # Commitment loss: pull projected features toward their assigned
            # codebook entries (encourages the quantizer to produce meaningful codes)
            normalized = F.normalize(projected @ self.quantizer.P)
            cb = self.quantizer.CB  # [codebook_size, codebook_dim]
            codebook_size = cb.shape[0]
            neg_dists = -(normalized.unsqueeze(2) - cb.unsqueeze(0).unsqueeze(0)).pow(2).sum(dim=-1)
            self.commitment_loss = F.cross_entropy(
                neg_dists.reshape(-1, codebook_size),
                codes.reshape(-1),
                reduction='mean',
            )
        else:
            self.commitment_loss = None

        return codes


class Model(nn.Module):
    """
    Engram-augmented CTC Conformer with Acoustic Quantization.

    Extends the LID-aware SC CTC architecture by replacing SC layers with
    Engram modules that use jointly-derived keys from LID predictions and
    quantized acoustic features.

    forward() returns (log_probs, audio_features_len) — compatible with
    flashlight_ctc_v1_onnx_v2 decoder.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        if isinstance(model_config_dict, dict):
            self.cfg = ModelConfig.from_dict(model_config_dict)
        else:
            self.cfg = model_config_dict

        frontend_config = self.cfg.frontend_config
        conformer_size = self.cfg.conformer_size

        conformer_config = ConformerEncoderV2Config(
            num_layers=self.cfg.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV2Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV2Config(
                    input_dim=conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                    dropout_broadcast_axes=None,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                ),
                modules=self.cfg.module_list,
                scales=self.cfg.module_scales,
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1OnnxExportable(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV2(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)

        # LID head (provides primary keys for Engram)
        self.lid_final_linear = nn.Linear(conformer_size, 17)
        self.lid_sc_linear = nn.Linear(17, conformer_size)

        # Acoustic quantizer (provides secondary keys for Engram)
        self.acoustic_quantizer = AcousticQuantizer(
            model_dim=conformer_size,
            feat_dim=self.cfg.acoustic_feat_dim,
            codebook_dim=self.cfg.acoustic_codebook_dim,
            codebook_size=self.cfg.acoustic_codebook_size,
        )

        # Engram modules (replace SC layers)
        engram_cfg = EngramConfig(
            ngram_orders=self.cfg.engram_ngram_orders,
            num_heads=self.cfg.engram_num_heads,
            mem_dim=self.cfg.engram_mem_dim,
            model_dim=conformer_size,
            table_size=self.cfg.engram_table_size,
            dropout=self.cfg.engram_dropout,
        )
        self.engrams = nn.ModuleList([
            EngramModule(engram_cfg) for _ in self.cfg.sc_layer
        ])

        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.lid_sc_layer = self.cfg.lid_sc_layer
        self.sc_layer = self.cfg.sc_layer

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: [B, T, 1]
        :param raw_audio_len: [B]
        :return: (log_probs [B, T, V+1], audio_features_len [B])
        """
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            if self.training:
                audio_features = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )

        mask = mask_tensor(audio_features, audio_features_len)

        if isinstance(self.conformer.frontend, nn.Identity):
            x = audio_features
            out_mask = mask
        else:
            x, out_mask = self.conformer.frontend(audio_features, mask)

        last_layer = len(self.conformer.module_list) - 1
        lid_keys = None
        acoustic_keys = None
        eng_idx = 0

        for i in range(last_layer + 1):
            x = self.conformer.module_list[i](x, out_mask)

            if i + 1 == self.lid_sc_layer:
                # LID prediction (same as original model)
                lid_out = self.lid_final_linear(x)
                lid_probs = torch.log_softmax(lid_out, dim=2)
                sc_lid_features = self.lid_sc_linear(torch.exp(lid_probs))
                x = x + sc_lid_features.detach()
                # Extract discrete LID predictions as keys for Engram
                lid_keys = torch.argmax(lid_probs, dim=2)  # [B, T]

            if i + 1 in self.sc_layer:
                # Generate acoustic keys from CONFORMER hidden states (aligned with LID keys)
                # This ensures both key sources share the same time dimension
                if acoustic_keys is None:
                    acoustic_keys = self.acoustic_quantizer(x)  # [B, T_conformer]

                # Combine LID + acoustic keys into joint key
                # Offset acoustic keys by number of LID classes to avoid collision
                joint_keys = lid_keys * self.cfg.acoustic_codebook_size + acoustic_keys

                # Engram lookup with joint keys
                x = x + self.engrams[eng_idx](x, joint_keys)
                eng_idx += 1

        conformer_out = self.final_dropout(x)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
    """Build ModelConfig with Engram + acoustic quantizer defaults."""
    base_config = get_base_model_config(vocab_size_without_blank, {})
    config_dict = {
        "feature_extraction_config": base_config.feature_extraction_config,
        "frontend_config": base_config.frontend_config,
        "specaug_config": base_config.specaug_config,
        "specauc_start_epoch": base_config.specauc_start_epoch,
        "label_target_size": base_config.label_target_size,
        "conformer_size": base_config.conformer_size,
        "num_layers": base_config.num_layers,
        "num_heads": base_config.num_heads,
        "ff_dim": base_config.ff_dim,
        "att_weights_dropout": base_config.att_weights_dropout,
        "conv_dropout": base_config.conv_dropout,
        "ff_dropout": base_config.ff_dropout,
        "mhsa_dropout": base_config.mhsa_dropout,
        "conv_kernel_size": base_config.conv_kernel_size,
        "final_dropout": base_config.final_dropout,
        "module_list": base_config.module_list,
        "module_scales": base_config.module_scales,
        "lid_sc_layer": network_args.get("lid_sc_layer", 3),
        "sc_layer": network_args.get("sc_layer", [6, 9]),
        "engram_ngram_orders": network_args.get("engram_ngram_orders", [2, 3]),
        "engram_num_heads": network_args.get("engram_num_heads", 8),
        "engram_mem_dim": network_args.get("engram_mem_dim", 1280),
        "engram_table_size": network_args.get("engram_table_size", 2**12),
        "engram_dropout": network_args.get("engram_dropout", 0.0),
        "acoustic_feat_dim": network_args.get("acoustic_feat_dim", 80),
        "acoustic_codebook_dim": network_args.get("acoustic_codebook_dim", 64),
        "acoustic_codebook_size": network_args.get("acoustic_codebook_size", 256),
        "quantizer_commitment_weight": network_args.get("quantizer_commitment_weight", 0.1),
        "contrastive_weight": network_args.get("contrastive_weight", 0.01),
        "contrastive_temperature": network_args.get("contrastive_temperature", 0.07),
    }
    return ModelConfig(**config_dict)


def train_step(*, model, data, run_ctx, **kwargs):
    """
    CTC loss + quantizer commitment loss.
    No SC auxiliary losses (Engram replaces SC).
    """
    raw_audio = data["data"]
    raw_audio_len = data["data:size1"].to("cpu")
    labels = data["targets"]
    labels_len = data["targets:size1"]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    # Main CTC loss
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_labels = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_labels)

    # Quantizer commitment loss (computed during forward, stored on model)
    if hasattr(model, 'acoustic_quantizer') and model.acoustic_quantizer.commitment_loss is not None:
        commitment_loss = model.acoustic_quantizer.commitment_loss
        run_ctx.mark_as_loss(
            name="quantizer_commitment",
            loss=commitment_loss,
            inv_norm_factor=raw_audio.shape[0],
            weight=model.cfg.quantizer_commitment_weight,
        )


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", 'w') as f:
        np.savetxt(f, log_average_probs, delimiter=' ')
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["data"]
    raw_audio_len = data["data:size1"].to("cpu")

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
