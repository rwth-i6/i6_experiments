"""
Tacotron-2 / Transformer TTS / GlowTTS style Convolutional Pre-Net

Transformer TTS: https://arxiv.org/pdf/1809.08895.pdf
512-dim, 3 layers, unknown filter size, batch norm, ReLU

GlowTTS:
Additional Residual connection for pre-net


"""
from dataclasses import dataclass
import torch
from torch import nn

from i6_models.config import ModelConfiguration
from .norm import ConvLayerNorm


@dataclass
class TTSEncoderPreNetV1Config(ModelConfiguration):
    """
    if include_residual, it has to be
    input_embedding_size = hidden_dimension = output_dimension

    Args:
        input_embedding_size:
        hidden_dimension:
        kernel_size:
        output_dimension:
        num_layers:
        dropout:
    """
    input_embedding_size: int
    hidden_dimension: int
    kernel_size: int
    output_dimension: int
    num_layers: int
    dropout: float


class TTSEncoderPreNetV1(torch.nn.Module):
    """

    """
    def __init__(
            self,
            config: TTSEncoderPreNetV1Config
        ):
        super().__init__()

        self.config = config
        self.conv_layers = torch.nn.ModuleList()
        self.norm_layers = torch.nn.ModuleList()

        for i in range(config.num_layers):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=config.input_embedding_size if i == 0 else config.hidden_dimension,
                    out_channels=config.hidden_dimension,
                    kernel_size=config.kernel_size,
                    padding="same",
                )
            )
            self.norm_layers.append(
            ConvLayerNorm(config.hidden_dimension)
            )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=config.dropout)

        # Is FF, but use Conv1d for B,C,N format
        self.out_ff = nn.Conv1d(config.hidden_dimension, config.output_dimension, kernel_size=1, bias=True)

        # Glow TTS used zeros for out init, is this reasonable?
        torch.nn.init.zeros_(self.out_ff.weight)
        torch.nn.init.zeros_(self.out_ff.bias)

    def forward(self, embedding: torch.Tensor, encoder_mask: torch.Tensor) -> torch.Tensor:
        """
        :param embedding: [B, input_embedding_size, N]
        :param encoder_mask: [B, 1, N]
        :return [B, output_dimension, N]
        """
        x_residual = embedding
        x = embedding
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x * encoder_mask)
            x = norm_layer(x)
            x = self.dropout(self.relu(x))
        out = (self.out_ff(x) + x_residual) * encoder_mask
        return out







