from dataclasses import dataclass
import torch
from torch import nn
from typing import Tuple, Optional

from librosa import filters


@dataclass
class DbMelFeatureExtractionConfig():
    """

    :param sample_rate: audio sample rate in Hz
    :param win_size: window size in seconds
    :param hop_size: window shift in seconds
    :param f_min: minimum mel filter frequency in Hz
    :param f_max: maximum mel fitler frequency in Hz
    :param min_amp: minimum amplitude for safe log
    :param num_filters: number of mel windows
    :param center: centered STFT with automatic padding
    :param norm: tuple optional of mean & std_dev for feature normalization
    """
    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    norm: Optional[Tuple[float, float]] = None

    @classmethod
    def from_dict(cls, d):
        return DbMelFeatureExtractionConfig(**d)


class DbMelFeatureExtraction(nn.Module):

    def __init__(
            self,
            config: DbMelFeatureExtractionConfig
    ):
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(int(config.win_size * config.sample_rate)))
        self.register_buffer("hop_length", torch.tensor(int(config.hop_size * config.sample_rate)))
        self.register_buffer("min_amp", torch.tensor(config.min_amp))
        self.center = config.center
        if config.norm is not None:
            self.apply_norm = True
            self.register_buffer("norm_mean", torch.tensor(config.norm[0]))
            self.register_buffer("norm_std_dev", torch.tensor(config.norm[1]))
        else:
            self.apply_norm = False

        self.register_buffer("mel_basis", torch.tensor(filters.mel(
            sr=config.sample_rate,
            n_fft=int(config.sample_rate * config.win_size),
            n_mels=config.num_filters,
            fmin=config.f_min,
            fmax=config.f_max)))
        self.register_buffer("window", torch.hann_window(int(config.win_size * config.sample_rate)))

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """

        S = torch.abs(torch.stft(
            raw_audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            pad_mode="constant",
            return_complex=True,
        )) ** 2
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

        return feature_data, length.int()
