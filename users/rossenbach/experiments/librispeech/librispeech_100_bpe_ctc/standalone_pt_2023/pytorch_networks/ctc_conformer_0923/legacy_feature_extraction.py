__all__ = ["LogMelFeatureExtractionV1", "LogMelFeatureExtractionV1Config"]

from dataclasses import dataclass
from typing import Optional, Tuple

from librosa import filters
import torch
from torch import nn

from i6_models.config import ModelConfiguration


@dataclass
class LogMelFeatureExtractionV1Config(ModelConfiguration):
    """
    Attributes:
        sample_rate: audio sample rate in Hz
        win_size: window size in seconds
        hop_size: window shift in seconds
        f_min: minimum filter frequency in Hz
        f_max: maximum filter frequency in Hz
        min_amp: minimum amplitude for safe log
        num_filters: number of mel windows
        center: centered STFT with automatic padding
    """

    sample_rate: int
    win_size: float
    hop_size: float
    f_min: int
    f_max: int
    min_amp: float
    num_filters: int
    center: bool
    n_fft: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.f_max <= self.sample_rate // 2, "f_max can not be larger than half of the sample rate"
        assert self.f_min > 0 and self.f_max > 0 and self.sample_rate > 0, "frequencies need to be positive"
        assert self.win_size > 0 and self.hop_size > 0, "window settings need to be positive"
        assert self.num_filters > 0, "number of filters needs to be positive"
        assert self.hop_size <= self.win_size, "using a larger hop size than window size does not make sense"
        if self.n_fft is None:
            # if n_fft is not given, set n_fft to the window size (in samples)
            self.n_fft = int(self.win_size * self.sample_rate)
        else:
            assert self.n_fft >= self.win_size * self.sample_rate, "n_fft cannot to be smaller than the window size"


class LogMelFeatureExtractionV1(nn.Module):
    """
    Librosa-compatible log-mel feature extraction using log10. Does not use torchaudio.

    Using it wrapped with torch.no_grad() is recommended if no gradient is needed
    """

    def __init__(self, cfg: LogMelFeatureExtractionV1Config):
        super().__init__()
        self.register_buffer("n_fft", torch.tensor(cfg.n_fft))
        self.register_buffer("window", torch.hann_window(int(cfg.win_size * cfg.sample_rate)))
        self.register_buffer("hop_length", torch.tensor(int(cfg.hop_size * cfg.sample_rate)))
        self.register_buffer("min_amp", torch.tensor(cfg.min_amp))
        self.center = cfg.center
        self.register_buffer(
            "mel_basis",
            torch.tensor(
                filters.mel(
                    sr=cfg.sample_rate,
                    n_fft=int(cfg.sample_rate * cfg.win_size),
                    n_mels=cfg.num_filters,
                    fmin=cfg.f_min,
                    fmax=cfg.f_max,
                )
            ),
        )

    def forward(self, raw_audio, length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length in samples: [B]
        :return features as [B,T,F] and length in frames [B]
        """
        power_spectrum = (
            torch.abs(
                torch.stft(
                    raw_audio,
                    n_fft=self.n_fft.cpu(),
                    hop_length=self.hop_length,
                    window=self.window,
                    center=self.center,
                    pad_mode="constant",
                    return_complex=True,
                )
            )
            ** 2
        )
        if len(power_spectrum.size()) == 2:
            # For some reason torch.stft removes the batch axis for batch sizes of 1, so we need to add it again
            power_spectrum = torch.unsqueeze(power_spectrum, 0)
        melspec = torch.einsum("...ft,mf->...mt", power_spectrum, self.mel_basis)
        log_melspec = torch.log10(torch.max(self.min_amp, melspec))
        feature_data = torch.transpose(log_melspec, 1, 2)

        if self.center:
            length = (length // self.hop_length) + 1
        else:
            length = ((length - self.n_fft) // self.hop_length) + 1

        return feature_data, length.int()