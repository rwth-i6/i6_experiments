from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
from torch import nn

from .scf import FeatureExtractionConfig


@dataclass
class StftFeatureExtractionV1Config(FeatureExtractionConfig):
    """
    Attributes:
        window_size: window size in samples
        window_shift: window shift in samples
        center: centered STFT with automatic padding
    """

    window_size: int
    window_shift: int
    center: bool
    magnitude: bool
    n_fft: Optional[int] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return StftFeatureExtractionV1Config(**d)


class StftFeatureExtractionV1(nn.Module):
    """
    Like V1, but with additional convolutions to reduce the feature dimension after the envelope extraction.
    """

    def __init__(self, cfg: StftFeatureExtractionV1Config):
        super().__init__()
        if cfg.n_fft is None:
            cfg.n_fft = cfg.window_size
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(self.cfg.window_size))

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T'',F']
        """
        feature_data = torch.stft(
            raw_audio,
            n_fft=self.cfg.n_fft,
            hop_length=self.cfg.window_shift,
            win_length=self.cfg.window_size,
            window=self.window,
            center=self.cfg.center,
            pad_mode="constant",
            return_complex=True,
        )
        if self.cfg.magnitude:
            feature_data = torch.abs(feature_data)
        feature_data = torch.transpose(feature_data, 1, 2)

        if self.cfg.center:
            length = (length // self.cfg.window_shift) + 1
        else:
            length = ((length - self.cfg.n_fft) // self.cfg.window_shift) + 1

        return feature_data, length

