from dataclasses import dataclass
from typing import Union, Tuple, Optional
import torch
from torch import nn

from .scf import FeatureExtractionConfig


@dataclass
class ConvFeatureExtractionV1Config(FeatureExtractionConfig):
    wave_norm: bool
    out_channels: int
    kernel_size: int
    stride: int
    bias: bool
    init: Optional[str]
    activation: Optional[Union[str, nn.Module]]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation = d.pop("activation")
        if activation is None:
            pass
        elif activation == "ReLU":
            from torch.nn import ReLU
            activation = ReLU()
        else:
            assert False, f"Unsupported activation {activation}"
        d["activation"] = activation
        return ConvFeatureExtractionV1Config(**d)


class ConvFeatureExtractionV1(nn.Module):
    """
    Feature extraction front-end that just runs Conv1d
    (and optionally normalizes the waveform before and adds an activation after).
    """

    def __init__(self, cfg: ConvFeatureExtractionV1Config):
        super().__init__()
        self.cfg = cfg
        self.wave_norm = nn.Conv1d(1, 1, 1) if cfg.wave_norm else None
        self.conv = nn.Conv1d(1, cfg.out_channels, cfg.kernel_size, stride=cfg.stride, bias=cfg.bias)
        if cfg.init == "gammatone":
            from .gammatone import GammatoneFilterbank
            gt_fbank = GammatoneFilterbank(cfg.out_channels, cfg.kernel_size / 16000)
            gt_fbank_tensor = torch.from_numpy(gt_fbank.get_gammatone_filterbank()).float()  # [T, C]
            gt_fbank_tensor = gt_fbank_tensor.transpose(0, 1).unsqueeze(1)  # [C, 1, T]
            self.conv.weight = nn.Parameter(gt_fbank_tensor)
        self.activation = cfg.activation

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T',F]
        """
        raw_audio = torch.unsqueeze(raw_audio, 1)  # [B,1,T]

        if self.wave_norm is not None:
            raw_audio = raw_audio - torch.mean(raw_audio, dim=-1)[:, None]
            raw_audio = raw_audio * torch.rsqrt(torch.var(raw_audio, dim=-1)[:, None] + 1e-6)
            raw_audio = self.wave_norm(raw_audio)

        feature_data = self.conv(raw_audio)  # [B,F,T']
        if self.activation is not None:
            assert isinstance(self.activation, nn.Module)
            feature_data = self.activation(feature_data)
        feature_data = feature_data.transpose(1, 2)  # [B,T',F]

        length = ((length - self.conv.kernel_size[-1]) / self.conv.stride[-1] + 1).int()

        return feature_data, length

