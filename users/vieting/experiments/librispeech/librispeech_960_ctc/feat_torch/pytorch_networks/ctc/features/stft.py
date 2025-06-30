from dataclasses import dataclass
from typing import Union, Tuple, Optional, Type
import torch
from torch import nn

from i6_models.config import ModelConfiguration
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
    Feature extraction front-end that just runs the STFT.
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


@dataclass
class StftFeatureExtractionV2Config(FeatureExtractionConfig):
    """
    Attributes:
        window_size: window size in samples
        window_shift: window shift in samples
        center: centered STFT with automatic padding
    """
    proc_config: ModelConfiguration
    proc_module: Union[str, Type[nn.Module]]
    window_size: int
    window_shift: int
    center: bool
    n_fft: Optional[int] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        if isinstance(d["proc_config"], dict):
            assert isinstance(d["proc_module"], str), "not implemented otherwise"
            # just infer config class from module class
            from ..conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2_cfg import (
                VGGNLayerActFrontendV1Config,
                VGGNLayerActFrontendV2Config,
            )
            d["proc_config"] = locals()[d["proc_module"] + "Config"](**d["proc_config"])
        if isinstance(d["proc_module"], str):
            from ..conformer_0924.i6models_relposV1_VGGNLayerActFrontendV1_feat_v2 import (
                VGGNLayerActFrontendV1,
                VGGNLayerActFrontendV2,
            )
            d["proc_module"] = locals()[d["proc_module"]]
        return StftFeatureExtractionV2Config(**d)


class StftFeatureExtractionV2(nn.Module):
    """
    Feature extraction front-end that runs the STFT and some further layer(s) to process real and imaginary part
    separately before summing them up.
    """

    def __init__(self, cfg: StftFeatureExtractionV2Config):
        super().__init__()
        if cfg.n_fft is None:
            cfg.n_fft = cfg.window_size
        self.cfg = cfg
        self.register_buffer("window", torch.hann_window(self.cfg.window_size))
        self.proc_re = cfg.proc_module(cfg.proc_config)
        self.proc_im = cfg.proc_module(cfg.proc_config)

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
        feature_data = torch.transpose(feature_data, 1, 2)

        if self.cfg.center:
            length = (length // self.cfg.window_shift) + 1
        else:
            length = ((length - self.cfg.n_fft) // self.cfg.window_shift) + 1

        from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1 import mask_tensor
        mask = mask_tensor(feature_data, length)
        feature_data_re, mask_re = self.proc_re(torch.real(feature_data), mask)
        feature_data_im, mask_im = self.proc_im(torch.imag(feature_data), mask)

        assert torch.equal(mask_re, mask_im)
        feature_data = feature_data_re + feature_data_im
        length = mask_re.sum(dim=1)

        return feature_data, length

