from dataclasses import dataclass
import torch
from torch import nn
import math
from typing import Tuple


from .prenet import TTSEncoderPreNetV1Config, TTSEncoderPreNetV1
from .rel_mhsa import GlowTTSMultiHeadAttentionV1, GlowTTSMultiHeadAttentionV1Config
from .norm import ConvLayerNorm
from ..util import sequence_mask


@dataclass
class TTSTransformerTextEncoderV1Config():
    num_layers: int
    vocab_size: int
    basic_dim: int
    conv_dim: int
    conv_kernel_size: int
    dropout: float
    mhsa_config: GlowTTSMultiHeadAttentionV1Config
    prenet_config: TTSEncoderPreNetV1Config
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["mhsa_config"] = GlowTTSMultiHeadAttentionV1Config(**d["mhsa_config"])
        d["prenet_config"] = TTSEncoderPreNetV1Config(**d["prenet_config"])
        return TTSTransformerTextEncoderV1Config(**d)





class DualConv(nn.Module):
    def __init__(self, base_channels, hidden_channels, kernel_size, p_dropout=0., activation=None):
        super().__init__()
        self.base_channels = base_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation

        self.conv_1 = nn.Conv1d(base_channels, hidden_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(hidden_channels, base_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        """

        :param x: [B, F, T]
        :param x_mask: [B, 1, T]
        :return:
        """
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
          x = x * torch.sigmoid(1.702 * x)
        else:
          x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class TTSTransformerTextEncoderV1(nn.Module):
    """
    Text Encoder model for TTS models (e.g. FastSpeech, GlowTTS, GradTTS, adapted from GlowTTS code

    Based on a "convolution" Transformer architecture, where FFN was replaced by convolutions
    """

    def __init__(
        self,
        cfg: TTSTransformerTextEncoderV1Config,
    ):
        """
        Text Encoder Model based on Multi-Head Self-Attention combined with FF-CCNs
        """
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.prenet_config.input_embedding_size)
        nn.init.normal_(self.emb.weight, 0.0, cfg.basic_dim**-0.5)

        self.pre_net = TTSEncoderPreNetV1(cfg.prenet_config)

        self.drop = nn.Dropout(cfg.dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(cfg.num_layers):
            self.attn_layers.append(
                GlowTTSMultiHeadAttentionV1(cfg=cfg.mhsa_config, features_last=False))
            self.norm_layers_1.append(ConvLayerNorm(channels=cfg.basic_dim))
            self.ffn_layers.append(
                DualConv(cfg.basic_dim, cfg.conv_dim, cfg.conv_kernel_size, p_dropout=cfg.dropout))
            self.norm_layers_2.append(ConvLayerNorm(channels=cfg.basic_dim))


    def forward(self, phon_labels, labels_lengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        :param phon_labels: [B, N]
        :param labels_lengths: [B]
        :return Tuple of time axis last encoder states [B, basic_dim, N] and corresponding encoder mask as [B, 1, N]
        """
        h = self.emb(phon_labels) * math.sqrt(self.cfg.prenet_config.input_embedding_size)  # [B, N, input_embedding_size]
        h = torch.transpose(h, 1, -1)  # [b, h, t]
        h_mask = torch.unsqueeze(sequence_mask(labels_lengths, h.size(2)), 1).to(h.dtype)

        h = self.pre_net(h, h_mask)

        for i in range(self.cfg.num_layers):
            h = h * h_mask
            att = self.attn_layers[i](h, h, h_mask)
            att = self.drop(att)
            h = self.norm_layers_1[i](h + att)

            ff = self.ffn_layers[i](h, h_mask)
            ff = self.drop(ff)
            h = self.norm_layers_2[i](h + ff)
        h = h * h_mask

        return h, h_mask