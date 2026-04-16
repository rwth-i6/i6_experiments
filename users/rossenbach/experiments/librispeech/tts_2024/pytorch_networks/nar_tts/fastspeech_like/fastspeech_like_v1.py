from dataclasses import dataclass
import torch
from torch import nn
import math
import os
from typing import Any, Dict



from ...tts_shared.tts_base_model.base_model_v1 import BaseTTSModelV1
from ...tts_shared import DbMelFeatureExtractionConfig
from ...tts_shared.encoder.transformer import TTSTransformerTextEncoderV1Config, ConvLayerNorm, DualConv
from ...tts_shared.encoder.duration_predictor import SimpleConvDurationPredictorV1Config
from ...tts_shared.encoder.rel_mhsa import GlowTTSMultiHeadAttentionV1, GlowTTSMultiHeadAttentionV1Config
from ...tts_shared.util import sequence_mask


@dataclass
class FastSpeechDecoderConfig():
    """
        Args:
            target_channels: Number of feature- and latent space channels
            hidden_dim: Internal dimension of the decoder
            kernel_size: Kernel Size for convolutions in coupling blocks
            dropout: Dropout probability for CNN in coupling blocks.
    """
    target_channels: int
    basic_dim: int
    conv_dim: int
    conv_kernel_size: int
    num_layers: int
    dropout: float
    mhsa_config: GlowTTSMultiHeadAttentionV1Config
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["mhsa_config"] = GlowTTSMultiHeadAttentionV1Config(**d["mhsa_config"])
        return FastSpeechDecoderConfig(**d)


class FastSpeechDecoder(torch.nn.Module):

    def __init__(self, cfg: FastSpeechDecoderConfig, encoder_hidden_size: int, speaker_embedding_size: int):
        super().__init__()
        self.cfg = cfg


        # linear as conv because of B,F,T format
        self.input_linear = nn.Conv1d(
            in_channels=encoder_hidden_size + speaker_embedding_size,
            out_channels=cfg.basic_dim,
            kernel_size=1,
            padding="same",
        )


        self.drop = nn.Dropout(cfg.dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.attn_layers.append(
                GlowTTSMultiHeadAttentionV1(cfg=cfg.mhsa_config, features_last=False))
            self.norm_layers_1.append(ConvLayerNorm(channels=cfg.basic_dim))
            self.ffn_layers.append(
                DualConv(cfg.basic_dim, cfg.conv_dim, cfg.conv_kernel_size, p_dropout=cfg.dropout))
            self.norm_layers_2.append(ConvLayerNorm(channels=cfg.basic_dim))

        self.output_linear = nn.Conv1d(
            in_channels=cfg.basic_dim,
            out_channels=cfg.target_channels,
            kernel_size=1,
            padding="same"
        )


    def forward(self, h_upsampled, s_mask, speaker_embedding):
        """

        :param h_upsampled: [B, encoder base_dim, T]
        :param s_mask: [B, 1, T]
        :param speaker_embedding: [B, speaker embedding size, 1]
        :return: target features: [B, target_channels, T]
        """
        spk_extended = speaker_embedding.expand(-1, -1, h_upsampled.size(-1))  # [B, emb] -> [B, emb, T]
        s = self.input_linear(torch.concat([h_upsampled, spk_extended], dim=1))
        
        for i in range(self.cfg.num_layers):
            s = s * s_mask
            att = self.attn_layers[i](s, s, s_mask)
            att = self.drop(att)
            s = self.norm_layers_1[i](s + att)

            ff = self.ffn_layers[i](s, s_mask)
            ff = self.drop(ff)
            s = self.norm_layers_2[i](s + ff)
        s = s * s_mask

        out_features = self.output_linear(s) * s_mask

        return out_features


@dataclass
class Config():
    """
        Args:
            num_speakers: input size of the speaker embedding matrix
            speaker_embedding_size: output size of the speaker embedding matrix
            mean_only: if true, standard deviation for latent z estimation is fixed to 1
    """
    feature_extraction_config: DbMelFeatureExtractionConfig
    encoder_config: TTSTransformerTextEncoderV1Config
    duration_predictor_config: SimpleConvDurationPredictorV1Config
    decoder_config: FastSpeechDecoderConfig

    num_speakers: int
    speaker_embedding_size: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig(**d["feature_extraction_config"])
        d["encoder_config"] = TTSTransformerTextEncoderV1Config.from_dict(d["encoder_config"])
        d["duration_predictor_config"] = SimpleConvDurationPredictorV1Config(**d["duration_predictor_config"])
        d["decoder_config"] = FastSpeechDecoderConfig.from_dict(d["decoder_config"])
        return Config(**d)


class Model(BaseTTSModelV1):
    """
    Flow-based TTS model based on GlowTTS Structure
    Following the definition from https://arxiv.org/abs/2005.11129
    and code from https://github.com/jaywalnut310/glow-tts
    """

    def __init__(
        self,
        config: Dict[str, Any],
        **kwargs,
    ):
        # build config from nested dict
        config = Config.from_dict(config)
        self.config = config

        super().__init__(
            feature_extraction_config=config.feature_extraction_config,
            encoder_config=config.encoder_config,
            duration_predictor_config=config.duration_predictor_config,
            num_speakers=config.num_speakers,
            speaker_embedding_size=config.speaker_embedding_size,
        )

        self.decoder = FastSpeechDecoder(
            cfg=config.decoder_config,
            encoder_hidden_size=config.encoder_config.basic_dim,
            speaker_embedding_size=config.speaker_embedding_size,
        )

    def forward(
        self, phon_labels, labels_lengths, speaker_label, raw_audio=None, raw_audio_lengths=None, target_durations=None
    ):
        """

        :param phon_labels:
        :param labels_lengths:
        :param speaker_label:
        :param raw_audio:
        :param raw_audio_lengths:
        :param target_durations:
        :return:
          - output_features as [B, F, T]
          - target features as [B, F, T]
          - feature lengths as [B] (of T)
          - log_durations as [B, N]
        """
        if raw_audio is not None:
            target_features, y_lengths = self.extract_features(
                raw_audio=raw_audio.squeeze(-1),
                raw_audio_lengths=raw_audio_lengths
            )
        else:
            target_features, y_lengths = (None, None)

        h, h_mask, spk, log_durations = self.forward_encoder(
            phon_labels=phon_labels,
            labels_lengths=labels_lengths,
            speaker_label=speaker_label,
        )


        if target_durations is not None:
            # target durations can be considered masked (= padding area is zero)
            upsampler_durations = target_durations
        else:
            # create masked absolute durations in the desired shape
            masked_durations = torch.ceil(torch.exp(log_durations)) * h_mask  # [B, 1, N]
            masked_durations = masked_durations.squeeze(1)  # [B, N]
            if self.training:
                raise ValueError("target_durations need to be provided for training")
            upsampler_durations = masked_durations

        if raw_audio is not None:
            feature_lengths = y_lengths
        else:
            feature_lengths = torch.sum(upsampler_durations, dim=1)

        t_mask = sequence_mask(feature_lengths).unsqueeze(1)  # B, 1, T
        attn_mask = h_mask.unsqueeze(-1) * t_mask.unsqueeze(2)  # B, 1, N, T
        path = self.generate_path(upsampler_durations, attn_mask.squeeze(1))

        # path as [B, T, N]  x h as [B, N, F] -> [B, T, F] -> [B, F, T]
        upsampled_h = torch.matmul(path.transpose(1, 2), h.transpose(1, 2)).transpose(1, 2)

        output_features = self.decoder(h_upsampled=upsampled_h, s_mask=t_mask, speaker_embedding=spk)

        return output_features, target_features, feature_lengths, log_durations.squeeze(1)
        

def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_samples = data["raw_audio"]  # [B, T, 1]
    audio_features_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B, N]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    durations = data["durations"]  # [B, N]
    durations_len = data["durations:size1"]  # [B]

    output_features, target_features, features_lengths, log_durations = model(
        phonemes, phonemes_len, speaker_labels, raw_samples, audio_features_len, durations,
    )


    log_duration_targets = torch.log(torch.clamp_min(durations, 1))
    l_l1 = torch.sum(torch.abs(output_features - target_features)) / model.config.decoder_config.target_channels
    l_dp = torch.sum((log_durations - log_duration_targets) ** 2)

    run_ctx.mark_as_loss(name="l1", loss=l_l1, inv_norm_factor=torch.sum(features_lengths))
    run_ctx.mark_as_loss(name="dp", loss=l_dp, inv_norm_factor=torch.sum(phonemes_len))

