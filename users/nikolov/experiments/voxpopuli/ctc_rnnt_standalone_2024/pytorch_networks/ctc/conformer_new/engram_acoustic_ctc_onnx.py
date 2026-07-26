"""
ONNX-exportable recognition wrapper for the Engram-acoustic CTC model.

Architecture is identical to engram_acoustic_ctc.Model so that checkpoints
load correctly (same state_dict keys).

forward() returns (log_probs, audio_features_len) — exactly 2 outputs —
for compatibility with the flashlight_ctc_v1_onnx_v2 decoder.

Both LID conditioning (at lid_sc_layer) and Engram injection
(at each sc_layer) are preserved during inference, matching training-time
computation.
"""

import numpy as np
from librosa import filters
import torch
from torch import nn
from typing import Tuple, List
from dataclasses import dataclass

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v2 import (
    ConformerEncoderV2Config,
    ConformerBlockV2Config,
    ConformerEncoderV2,
)
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV2Config
from i6_models.parts.best_rq.quantizer import RandomProjectionQuantizer

from i6_experiments.users.nikolov.experiments.voxpopuli.ctc_rnnt_standalone_2024.pytorch_networks.ctc.conformer_new.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    SpecaugConfig,
    VGG4LayerActFrontendV1Config_mod,
    LogMelFeatureExtractionV1Config,
)
from i6_models.parts.engram import EngramConfig, EngramModule


# ---------------------------------------------------------------------------
# ModelConfig — must match engram_acoustic_ctc.ModelConfig exactly
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
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
    lid_sc_layer: int
    sc_layer: List[int]
    engram_ngram_orders: List[int]
    engram_num_heads: int
    engram_mem_dim: int
    engram_table_size: int
    engram_dropout: float
    acoustic_feat_dim: int
    acoustic_codebook_dim: int
    acoustic_codebook_size: int
    quantizer_commitment_weight: float
    contrastive_weight: float
    contrastive_temperature: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return ModelConfig(**d)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)
    return torch.less(r[None, :], seq_len[:, None])


class LogMelFeatureExtractionV1OnnxExportable(nn.Module):
    """Librosa-compatible log-mel (log10). ONNX-traceable."""

    def __init__(self, cfg: LogMelFeatureExtractionV1Config):
        super().__init__()
        self.center = cfg.center
        self.hop_length = int(cfg.hop_size * cfg.sample_rate)
        self.min_amp = cfg.min_amp
        self.n_fft = cfg.n_fft
        self.win_length = int(cfg.win_size * cfg.sample_rate)
        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=cfg.n_fft,
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                )
            ),
        )
        self.register_buffer("window", torch.hann_window(self.win_length))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        power_spectrum = torch.sum(
            torch.stft(
                raw_audio.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window,
                center=self.center,
                pad_mode="constant",
                return_complex=False,
            ) ** 2,
            dim=-1,
        )
        if len(power_spectrum.size()) == 2:
            power_spectrum = torch.unsqueeze(power_spectrum, 0)
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)
        log_melspec = torch.log10(torch.clamp(melspec, min=self.min_amp))
        feature_data = torch.transpose(log_melspec, 1, 2)
        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1
        return feature_data, length.int()


class AcousticQuantizer(nn.Module):
    """Same as training model — extracts prosodic features and quantizes."""

    def __init__(self, model_dim: int, feat_dim: int, codebook_dim: int, codebook_size: int):
        super().__init__()
        self.model_dim = model_dim
        self.feat_dim = feat_dim
        self.feature_proj = nn.Linear(model_dim, feat_dim)
        self.quantizer = RandomProjectionQuantizer(
            input_dim=feat_dim,
            codebook_dim=codebook_dim,
            codebook_num_vars=codebook_size,
        )
        self.contrast_proj = nn.Linear(feat_dim, 64)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        :param hidden_states: [B, T, model_dim] — Conformer hidden states
        :return: [B, T] discrete acoustic codes
        """
        projected = self.feature_proj(hidden_states)
        codes = self.quantizer(projected)
        return codes


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Model(torch.nn.Module):
    """
    ONNX-exportable Engram-acoustic CTC Conformer.

    State_dict keys match engram_acoustic_ctc.Model exactly.
    """

    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
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

        self.feature_extraction = LogMelFeatureExtractionV1OnnxExportable(
            cfg=self.cfg.feature_extraction_config
        )
        self.conformer = ConformerEncoderV2(cfg=conformer_config)
        self.final_linear = nn.Linear(conformer_size, self.cfg.label_target_size + 1)

        # LID head
        self.lid_final_linear = nn.Linear(conformer_size, 17)
        self.lid_sc_linear = nn.Linear(17, conformer_size)

        # Acoustic quantizer
        self.acoustic_quantizer = AcousticQuantizer(
            model_dim=conformer_size,
            feat_dim=self.cfg.acoustic_feat_dim,
            codebook_dim=self.cfg.acoustic_codebook_dim,
            codebook_size=self.cfg.acoustic_codebook_size,
        )

        # Engram modules
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
        self.lid_sc_layer = self.cfg.lid_sc_layer
        self.sc_layer = self.cfg.sc_layer
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T, 1]
        :param raw_audio_len: [B]
        :return: (log_probs [B, T, V+1], audio_features_len [B])
        """
        squeezed_features = torch.squeeze(raw_audio, dim=-1).float()
        audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

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
                lid_out = self.lid_final_linear(x)
                lid_probs = torch.log_softmax(lid_out, dim=2)
                sc_lid_features = self.lid_sc_linear(torch.exp(lid_probs))
                x = x + sc_lid_features.detach()
                lid_keys = torch.argmax(lid_probs, dim=2)

            if i + 1 in self.sc_layer:
                if acoustic_keys is None:
                    acoustic_keys = self.acoustic_quantizer(x)  # [B, T_conformer]

                joint_keys = lid_keys * self.cfg.acoustic_codebook_size + acoustic_keys
                x = x + self.engrams[eng_idx](x, joint_keys)
                eng_idx += 1

        conformer_out = self.final_dropout(x)
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


# ---------------------------------------------------------------------------
# RETURNN hooks
# ---------------------------------------------------------------------------

def train_step(**kwargs):
    raise NotImplementedError("engram_acoustic_ctc_onnx is for recognition only.")


def prior_init_hook(run_ctx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
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


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def get_model_config(vocab_size_without_blank: int, network_args: dict) -> ModelConfig:
    specauc_start_epoch = network_args.get("specauc_start_epoch", 1)
    lid_sc_layer = network_args.get("lid_sc_layer", 3)
    sc_layer = network_args.get("sc_layer", [6, 9])

    fe_config = LogMelFeatureExtractionV1Config(
        sample_rate=16000,
        win_size=0.025,
        hop_size=0.01,
        f_min=60,
        f_max=7600,
        min_amp=1e-10,
        num_filters=80,
        center=True,
    )
    specaug_config = SpecaugConfig(
        repeat_per_n_frames=25,
        max_dim_time=20,
        max_dim_feat=8,
        num_repeat_feat=5,
    )
    frontend_config = VGG4LayerActFrontendV1Config_mod(
        in_features=80,
        conv1_channels=32,
        conv2_channels=64,
        conv3_channels=64,
        conv4_channels=32,
        conv_kernel_size=(3, 3),
        conv_padding=None,
        pool1_kernel_size=(1, 2),
        pool1_stride=None,
        pool1_padding=None,
        pool2_kernel_size=(1, 2),
        pool2_stride=(4, 1),
        pool2_padding=None,
        activation_str="ReLU",
        out_features=512,
        activation=None,
    )
    return ModelConfig(
        feature_extraction_config=fe_config,
        frontend_config=frontend_config,
        specaug_config=specaug_config,
        label_target_size=vocab_size_without_blank,
        conformer_size=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        att_weights_dropout=0.1,
        conv_dropout=0.1,
        ff_dropout=0.1,
        mhsa_dropout=0.1,
        conv_kernel_size=31,
        final_dropout=0.1,
        specauc_start_epoch=specauc_start_epoch,
        module_list=["ff", "conv", "mhsa", "ff"],
        module_scales=[0.5, 1.0, 1.0, 0.5],
        lid_sc_layer=lid_sc_layer,
        sc_layer=sc_layer,
        engram_ngram_orders=network_args.get("engram_ngram_orders", [2, 3]),
        engram_num_heads=network_args.get("engram_num_heads", 8),
        engram_mem_dim=network_args.get("engram_mem_dim", 1280),
        engram_table_size=network_args.get("engram_table_size", 2**12),
        engram_dropout=network_args.get("engram_dropout", 0.0),
        acoustic_feat_dim=network_args.get("acoustic_feat_dim", 80),
        acoustic_codebook_dim=network_args.get("acoustic_codebook_dim", 64),
        acoustic_codebook_size=network_args.get("acoustic_codebook_size", 256),
        quantizer_commitment_weight=network_args.get("quantizer_commitment_weight", 0.1),
        contrastive_weight=network_args.get("contrastive_weight", 0.01),
        contrastive_temperature=network_args.get("contrastive_temperature", 0.07),
    )
