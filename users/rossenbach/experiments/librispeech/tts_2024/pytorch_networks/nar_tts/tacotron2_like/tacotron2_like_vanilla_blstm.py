from dataclasses import dataclass
import torch
from torch import nn
from typing import Any, Dict



from ...tts_shared.tts_base_model.base_model_v1 import BaseTTSModelV1
from ...tts_shared import DbMelFeatureExtractionConfig
from ...tts_shared.encoder.transformer import TTSTransformerTextEncoderV1Config, ConvLayerNorm, DualConv
from ...tts_shared.encoder.duration_predictor import SimpleConvDurationPredictorV1Config
from ...tts_shared.util import sequence_mask

from ...tts_shared.espnet_tacotron import decoder_init, Prenet, Postnet


@dataclass
class NarTacotronDecoderConfig():
    """
        Args:
            target_channels: Number of feature- and latent space channels
            hidden_dim: Internal dimension of the decoder
            kernel_size: Kernel Size for convolutions in coupling blocks
            dropout: Dropout probability for CNN in coupling blocks.
    """
    target_channels: int
    basic_dim: int
    num_lstm_layers: int
    lstm_dropout: float
    post_conv_dim: int
    post_conv_kernel_size: int
    post_conv_num_layers: int
    post_conv_dropout: float
    reduction_factor: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return NarTacotronDecoderConfig(**d)


class NarTacotronDecoder(torch.nn.Module):

    def __init__(self, cfg: NarTacotronDecoderConfig, encoder_hidden_size: int, speaker_embedding_size: int):
        super().__init__()
        self.cfg = cfg

        self.lstm_stack = nn.LSTM(
            input_size=(encoder_hidden_size * cfg.reduction_factor) + speaker_embedding_size,
            hidden_size=cfg.basic_dim,
            num_layers=cfg.num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=cfg.lstm_dropout
        )

        self.output_linear = nn.Linear(
            in_features=cfg.basic_dim * 2,
            out_features=cfg.target_channels * cfg.reduction_factor,
        )

        self.postnet = Postnet(
            odim=cfg.target_channels,
            n_layers=cfg.post_conv_num_layers,
            n_chans=cfg.post_conv_dim,
            n_filts=cfg.post_conv_kernel_size,
            dropout_rate=cfg.post_conv_dropout
        )

    def forward(self, h_upsampled, h_lengths, speaker_embedding):
        """

        :param h_upsampled: [B, T, encoder base_dim]
        :param h_lengths: [B]
        :param speaker_embedding: [B, 1, speaker embedding size]
        :return: target features: [B, target_channels, T]
        """

        # this is basically in order to achieve a "zip longest, the last y is not needed anyways"
        missing_frames = h_upsampled.size(1) % self.cfg.reduction_factor
        h_t = torch.nn.functional.pad(h_upsampled, [0, 0, 0, missing_frames])
        t_mask = sequence_mask(h_lengths, max_length=h_t.size(1)).unsqueeze(2)

        stacked_h_t = h_t.view(h_t.size(0), -1, h_t.size(2) * self.cfg.reduction_factor)  # [B, T/2, 2F]
        # ceiled lengths
        stacked_lengths = (h_lengths + self.cfg.reduction_factor - 1) // self.cfg.reduction_factor

        spk_extended = speaker_embedding.expand(-1, stacked_h_t.size(1), -1)  # [B, emb] -> [B, T/2, emb]
        stacked_h_t = torch.concat([stacked_h_t, spk_extended], dim=2)

        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(stacked_h_t, stacked_lengths.to("cpu"), batch_first=True, enforce_sorted=False)
        blstm_packed_out, _ = self.lstm_stack(blstm_packed_in)
        blstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out, padding_value=0.0, batch_first=True
        )  # [B, T, F]

        out_features = self.output_linear(blstm_out)

        # unpack features
        out_features = out_features.view(h_t.size(0), -1, self.cfg.target_channels) * t_mask  # 2 because of "B"LSTM

        out_postnet = self.postnet(out_features.transpose(1, 2)).transpose(1, 2)  # conv operates in [B, F, T]

        final_features = (out_features + out_postnet) * t_mask

        return out_features, final_features


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
    decoder_config: NarTacotronDecoderConfig

    num_speakers: int
    speaker_embedding_size: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig(**d["feature_extraction_config"])
        d["encoder_config"] = TTSTransformerTextEncoderV1Config.from_dict(d["encoder_config"])
        d["duration_predictor_config"] = SimpleConvDurationPredictorV1Config(**d["duration_predictor_config"])
        d["decoder_config"] = NarTacotronDecoderConfig.from_dict(d["decoder_config"])
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

        self.decoder = NarTacotronDecoder(
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
                raw_audio_lengths=raw_audio_lengths,
                time_last=False
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

        # path as [B, T, N]  x h as [B, N, F] -> [B, T, F]
        upsampled_h = torch.matmul(path.transpose(1, 2), h.transpose(1, 2))

        output_features_before, output_features_after = self.decoder(h_upsampled=upsampled_h, h_lengths=feature_lengths, speaker_embedding=spk.transpose(1, 2))

        # frames might be drop depending on reduction factor, so adapt
        max_length = torch.max(torch.sum(upsampler_durations, dim=1)).cpu().int()
        output_features_before = output_features_before[:, :max_length, :]
        output_features_after = output_features_after[:, :max_length, :]

        return output_features_before, output_features_after, target_features, feature_lengths, log_durations.squeeze(1)
        

def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_samples = data["raw_audio"]  # [B, T, 1]
    audio_features_len = data["raw_audio:size1"]  # [B]
    phonemes = data["phonemes"]  # [B, N] (sparse)
    phonemes_len = data["phonemes:size1"]  # [B, N]
    speaker_labels = data["speaker_labels"]  # [B, 1] (sparse)
    durations = data["durations"]  # [B, N]
    durations_len = data["durations:size1"]  # [B]

    output_features_before, output_features_after, target_features, features_lengths, log_durations = model(
        phonemes, phonemes_len, speaker_labels, raw_samples, audio_features_len, durations,
    )

    log_duration_targets = torch.log(torch.clamp_min(durations, 1))
    l_l1_before = torch.sum(torch.abs(output_features_before - target_features)) / model.config.decoder_config.target_channels
    l_l1_after = torch.sum(torch.abs(output_features_after - target_features)) / model.config.decoder_config.target_channels
    l_dp = torch.sum((log_durations - log_duration_targets) ** 2)

    run_ctx.mark_as_loss(name="l1_before", loss=l_l1_before, inv_norm_factor=torch.sum(features_lengths))
    run_ctx.mark_as_loss(name="l1_after", loss=l_l1_after, inv_norm_factor=torch.sum(features_lengths))
    run_ctx.mark_as_loss(name="dp", loss=l_dp, inv_norm_factor=torch.sum(phonemes_len))

