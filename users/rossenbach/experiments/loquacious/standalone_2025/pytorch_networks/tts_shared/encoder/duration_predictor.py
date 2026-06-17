from dataclasses import dataclass
import torch
from torch import nn

from .norm import ConvLayerNorm

from collections import OrderedDict


class Conv1DBlock(torch.nn.Module):
    """
    A 1D-Convolution with ReLU, batch-norm and non-broadcasted p_dropout
    Will pad to the same output length
    """

    def __init__(self, in_size, out_size, filter_size, p_dropout, norm="layer"):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param p_dropout: dropout probability
        """
        super().__init__()
        assert filter_size % 2 == 1, "Only odd filter sizes allowed"
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size // 2)
        self.ln = ConvLayerNorm(channels=out_size)
        self.p_dropout = p_dropout

    def forward(self, x_with_mask):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x, x_mask = x_with_mask
        x = self.conv(x * x_mask)
        x = nn.functional.relu(x)
        x = self.ln(x) # Layer normalization
        x = nn.functional.dropout(x, p=self.p_dropout, training=self.training)
        return (x, x_mask)


@dataclass
class SimpleConvDurationPredictorV1Config():
    num_convs: int
    hidden_dim: int
    kernel_size: int
    dropout: float


class SimpleConvDurationPredictorV1(nn.Module):
    """
    Duration Predictor module using a stack of convolutional layers
    """

    def __init__(self, cfg: SimpleConvDurationPredictorV1Config, input_dim: int):
        """

        :param cfg:
        :param input_dim: determined by the encoder output
        """
        super().__init__()

        self.cfg = cfg
        self.input_dim = input_dim

        self.convs = nn.Sequential(
            OrderedDict(
                [(
                    "conv_%i" % i,
                    Conv1DBlock(
                        in_size=input_dim if i == 0 else cfg.hidden_dim,
                        out_size=cfg.hidden_dim,
                        filter_size=cfg.kernel_size,
                        p_dropout=cfg.dropout,
                )) for i in range(cfg.num_convs)]
            )
        )
        self.proj = nn.Conv1d(in_channels=cfg.hidden_dim, out_channels=1, kernel_size=1)

    def forward(self, h, h_mask):
        """
        :param h: [B, F, N]
        :param h_mask: [B, 1, N]
        :return: some kind of durations (e.g. log durations) as [B, 1, N]
        """
        h_with_mask = (h, h_mask)
        (h, h_mask) = self.convs(h_with_mask)
        out = self.proj(h * h_mask)
        return out * h_mask

