"""
Contains the first version of a reference base model containing the FastSpeech/GlowTTS/GradTTS base encoder and duration
model as well as loss helpers
"""
import torch
from torch import nn
from torch.nn import functional as F


from .. import DbMelFeatureExtraction, DbMelFeatureExtractionConfig
from ..encoder.transformer import TTSTransformerTextEncoderV1Config, TTSTransformerTextEncoderV1
from ..encoder.duration_predictor import SimpleConvDurationPredictorV1, SimpleConvDurationPredictorV1Config
from ..util import sequence_mask, convert_pad_shape


class BaseTTSModelV1(nn.Module):

    def __init__(
            self,
            feature_extraction_config: DbMelFeatureExtractionConfig,
            encoder_config: TTSTransformerTextEncoderV1Config,
            duration_predictor_config: SimpleConvDurationPredictorV1Config,
            num_speakers: int,
            speaker_embedding_size: int,
    ):
        """

        :param encoder_config:
        :param dp_config:
        """
        super().__init__()

        self.num_speakers = num_speakers
        self.speaker_embedding_size = speaker_embedding_size

        self.feature_extraction_config = feature_extraction_config
        self.encoder_config = encoder_config
        self.duration_predictor_config = duration_predictor_config

        self.feature_extraction = DbMelFeatureExtraction(config=feature_extraction_config)
        self.encoder = TTSTransformerTextEncoderV1(cfg=encoder_config)
        self.duration_predictor = SimpleConvDurationPredictorV1(
            cfg=duration_predictor_config,
            input_dim=encoder_config.basic_dim + speaker_embedding_size
        )

        self.spk_emb = nn.Embedding(num_speakers, speaker_embedding_size)

    def extract_features(self, raw_audio, raw_audio_lengths, time_last=True):
        """

        :param raw_audio: [B, T, 1]
        :param raw_audio_lengths: [B], length of T
        :return:
        """
        with torch.no_grad():
            squeezed_audio = torch.squeeze(raw_audio, dim=-1)
            y, y_lengths = self.feature_extraction(
                squeezed_audio, raw_audio_lengths
            )  # [B, T, F]
            if time_last:
                y = y.transpose(1, 2)  # [B, F, T]
        return y, y_lengths

    def forward_encoder(self, phon_labels, labels_lengths, speaker_label):
        """
        :param phon_labels: [B, N]
        :param labels_lengths: [B], length of N
        :param speaker_label: [B], speaker index
        :return: h as [B, hidden_dim, N], h_mask [B, 1, N], spk as [B, F, 1] and log_durations as [B, 1, N]
        """
        h, h_mask = self.encoder(phon_labels=phon_labels, labels_lengths=labels_lengths)
        spk = nn.functional.normalize(self.spk_emb(speaker_label.squeeze(-1))).unsqueeze(-1)

        spk_exp = spk.expand(-1, -1, phon_labels.size(-1))  # [B, emb] -> [B, emb, N]
        # print(f"Dimension of input in Text Encoder: x.shape: {x.shape}; g: {g.shape}, g_exp: {g_exp.shape}")
        h_dp = torch.cat([torch.detach(h), spk_exp], 1)

        log_durations = self.duration_predictor(h_dp, h_mask)

        return h, h_mask, spk, log_durations

    @staticmethod
    def generate_path(duration, mask):
        """
        :param duration: [B, N]
        :param mask: [B, N, T]
        :return: path as [B, N, T] containing 1s for bijective aligned positions of single N to multiple of T
        """
        b, t_x, t_y = mask.shape
        cum_duration = torch.cumsum(duration, 1)

        cum_duration_flat = cum_duration.view(b * t_x)
        path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
        path = path.view(b, t_x, t_y)
        path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
        path = path * mask
        return path
