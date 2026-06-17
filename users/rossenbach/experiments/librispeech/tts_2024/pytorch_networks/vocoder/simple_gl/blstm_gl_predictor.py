from dataclasses import dataclass
import torch
from torch import nn

from librosa import filters
from typing import Tuple

from i6_models.config import ModelConfiguration

from ...tts_shared import DbMelFeatureExtraction, DbMelFeatureExtractionConfig

@dataclass
class BlstmGLPredictorConfig(ModelConfiguration):
    hidden_size: int
    feature_extraction_config: DbMelFeatureExtractionConfig


    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = DbMelFeatureExtractionConfig.from_dict(d["feature_extraction_config"])
        return BlstmGLPredictorConfig(**d)


class DbMelDualFeatureExtraction(DbMelFeatureExtraction):
    """
    Also returns linear output
    """

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return mel features as [B,T,F], linear features as [B, T, F2] and length in frames [B]
        """

        linear = torch.abs(torch.stft(
            raw_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        ))
        S = linear**2
        if len(S.size()) == 2:
            # For some reason torch.stft "eats" batch sizes of 1, so we need to add it again if needed
            S = torch.unsqueeze(S, 0)
        melspec = torch.einsum("...ft,mf->...mt", S, self.mel_basis)
        melspec = 20 * torch.log10(torch.max(self.min_amp, melspec))
        feature_data = torch.transpose(melspec, 1, 2)

        if self.apply_norm:
            feature_data = (feature_data - self.norm_mean) / self.norm_std_dev

        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, linear, length.int()



class Model(torch.nn.Module):


    def __init__(self, config, **kwargs):
        """

        :param config:
        """
        super().__init__()
        self.config = BlstmGLPredictorConfig.from_dict(config)
        self.feature_extraction = DbMelDualFeatureExtraction(config=self.config.feature_extraction_config)
        self.bypass_linear = nn.Linear(self.config.feature_extraction_config.num_filters, self.config.hidden_size)
        self.blstm_1 = nn.LSTM(
            input_size=self.config.feature_extraction_config.num_filters,
            num_layers=1,
            hidden_size=self.config.hidden_size,
            bidirectional=True,
            batch_first=True
        )
        self.blstm_2 = nn.LSTM(
            input_size=self.config.hidden_size * 2,
            num_layers=1,
            hidden_size=self.config.hidden_size,
            bidirectional=True,
            batch_first=True
        )

        self.combine_linear = nn.Linear(
            self.config.hidden_size * 3,
            int(self.config.feature_extraction_config.win_size * self.config.feature_extraction_config.sample_rate) // 2 + 1
        )


    def forward(self, log_mel, lengths):
        """

        :param log_mel: [B, T, F1]
        :param linear: [B, T, F2]
        :param lengths: [B]
        """
        blstm_in = log_mel

        # Sequences are sorted by decoder length, so here we do no sorting
        blstm_packed_in = nn.utils.rnn.pack_padded_sequence(blstm_in, lengths.to("cpu"), batch_first=True,
                                                            enforce_sorted=False)
        blstm_packed_out_1, _ = self.blstm_1(blstm_packed_in)
        blstm_packed_out_2, _ = self.blstm_2(blstm_packed_out_1)
        blstm_out_1, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out_1, padding_value=0.0, batch_first=True
        )  # [B, T, lstm_size*2]
        blstm_out_2, _ = nn.utils.rnn.pad_packed_sequence(
            blstm_packed_out_2, padding_value=0.0, batch_first=True
        )  # [B, T, lstm_size*2]
        lstm_combined = blstm_out_1 + blstm_out_2

        bypass_transformed = torch.nn.functional.relu(self.bypass_linear(log_mel))
        log_output = self.combine_linear(torch.concatenate([lstm_combined, bypass_transformed], dim=-1))

        output = torch.pow(10, log_output)

        return log_output, output



def train_step(*, model: Model, data, run_ctx, **kwargs):
    samples = data["raw_audio"]  # [B, T', 1]
    samples_len = data["raw_audio:size1"]  # [B]

    mel, linear, length = model.feature_extraction(torch.squeeze(samples), samples_len)
    linear = linear.transpose(1, 2)
    log_output, output = model(mel, length)

    log_target = torch.log10(linear + 1e-10)

    log_loss = torch.nn.functional.mse_loss(log_output, log_target, reduction="sum") / 401
    loss = torch.nn.functional.mse_loss(output, linear, reduction="sum") / 401

    length_sum = torch.sum(length)
    run_ctx.mark_as_loss(name="mse", loss=loss, inv_norm_factor=length_sum, scale=0.05)
    run_ctx.mark_as_loss(name="log_mse", loss=log_loss, inv_norm_factor=length_sum, scale=1.0)
