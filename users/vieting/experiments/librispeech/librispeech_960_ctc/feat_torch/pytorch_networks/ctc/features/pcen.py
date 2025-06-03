# Based on https://github.com/google-research/leaf-audio/blob/master/leaf_audio/postprocessing.py
# and https://github.com/TeaPoly/asr_frontend/blob/main/pcen.py

import torch
from dataclasses import dataclass, asdict
from typing import Optional

from .scf import FeatureExtractionConfig


@dataclass
class PcenV1Config(FeatureExtractionConfig):
    """
    Args:
      feat_dim: int, feature dims
      alpha: float, exponent of EMA smoother
      smooth_coefficient: float, smoothing coefficient of EMA
      delta: float, bias added before compression
      root: float, one over exponent applied for compression (r in the paper)
      floor: float, offset added to EMA smoother
    """
    feat_dim: int
    activation: Optional[str] = None
    alpha: float = 0.96
    smooth_coefficient: float = 0.04
    delta: float = 2.0
    root: float = 2.0
    floor: float = 1e-6


class PcenV1(torch.nn.Module):
    """
    Trainable per-channel energy normalization (PCEN) as presented in https://arxiv.org/abs/1607.05666.
    We also train the smoothing coefficient as done in LEAF:
    https://github.com/google-research/leaf-audio/blob/master/leaf_audio/postprocessing.py
    """

    def __init__(self, cfg: PcenV1Config):
        super().__init__()

        self.floor = cfg.floor

        # We need positive inputs. Therefore, apply activation upfront
        if cfg.activation is None:
            self.activation = None
        elif cfg.activation in vars(torch):
            self.activation = torch.__dict__[cfg.activation]
        else:
            raise NotImplementedError(f"Unknown activation function: {cfg.activation}")

        # Smoothing coefficient controls strength of history in EMA
        self.smooth = torch.nn.Parameter(torch.Tensor(cfg.feat_dim))
        torch.nn.init.constant_(self.smooth, cfg.smooth_coefficient)

        # The AGC strength (or gain normalization strength) is controlled by the parameter alpha in [0, 1]
        self.alpha = torch.nn.Parameter(torch.Tensor(cfg.feat_dim))
        torch.nn.init.constant_(self.alpha, cfg.alpha)

        # A stabilized root compression to further reduce the dynamic range offset delta and exponent r
        self.delta = torch.nn.Parameter(torch.Tensor(cfg.feat_dim))
        torch.nn.init.constant_(self.delta, cfg.delta)

        # Root for compression
        self.root = torch.nn.Parameter(torch.Tensor(cfg.feat_dim))
        torch.nn.init.constant_(self.root, cfg.root)

    def apply_iir(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implements a first order Infinite Impulse Response (IIR) forward filter initialized using the input values.
        :param x: Batch of (mel-) spectrograms. Shape: [..., Frequency, Time]
        :return M: Low-pass filtered version of the input spectrograms.
        """
        s = torch.clamp(self.smooth, min=0.0, max=1.0)

        ema = [x[..., 0]]
        for t in range(1, x.size(-1)):
            m = (1. - s) * ema[-1] + s * x[..., t]
            ema.append(m)
        ema = torch.stack(ema, dim=-1)

        return ema

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: Input tensor (#batch, feat, time).
        :return: (#batch, feat, time).
        """
        alpha = torch.min(self.alpha, torch.ones(self.alpha.size(), device=self.alpha.device)).unsqueeze(1)
        root = torch.max(self.root, torch.ones(self.root.size(), device=self.root.device)).unsqueeze(1)
        delta = self.delta.unsqueeze(1)

        if self.activation is not None:
            tensor = self.activation(tensor)
        ema_smoother = self.apply_iir(tensor)
        one_over_root = 1. / root
        tensor = (
            ((tensor / ((self.floor + ema_smoother) ** alpha) + delta) ** one_over_root)
            - (delta ** one_over_root)
        )

        return tensor
