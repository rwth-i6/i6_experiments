from typing import Optional, Iterable
import torch
from torch import nn
import torch.nn.functional as F
import whisper
from whisper.model import MultiHeadAttention, LayerNorm, Linear, Tensor, Conv1d, sinusoids


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, cross_attention: bool = False, dropout: float = 0.0):
        super().__init__()

        self.attn = MultiHeadAttention(n_state, n_head)
        self.attn_ln = LayerNorm(n_state)

        self.cross_attn = MultiHeadAttention(n_state, n_head) if cross_attention else None
        self.cross_attn_ln = LayerNorm(n_state) if cross_attention else None

        n_mlp = n_state * 4
        self.mlp = nn.Sequential(Linear(n_state, n_mlp), nn.GELU(), Linear(n_mlp, n_state))
        self.dropout = nn.Dropout(p=dropout) if dropout != 0.0 else None
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        xa: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[dict] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, kv_cache=kv_cache)[0]
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, kv_cache=kv_cache)[0]
        y = self.mlp_ln(x)
        y = self.mlp[0](y)
        y = self.mlp[1](y)
        if self.dropout:
            y = self.dropout(y)
        y = self.mlp[2](y)
        x = x + y
        return x


class Whisper(nn.Module):
    def __init__(self, dims: whisper.ModelDimensions, dropout: float):
        super().__init__()
        self.dims = dims
        self.encoder = AudioEncoder(
            self.dims.n_mels,
            self.dims.n_audio_ctx,
            self.dims.n_audio_state,
            self.dims.n_audio_head,
            self.dims.n_audio_layer,
            dropout=dropout,
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.encoder(mel)


class AudioEncoder(nn.Module):
    def __init__(self, n_mels: int, n_ctx: int, n_state: int, n_head: int, n_layer: int, dropout: float):
        super().__init__()
        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.register_buffer("positional_embedding", sinusoids(n_ctx, n_state))

        self.blocks: Iterable[ResidualAttentionBlock] = nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, dropout=dropout) for _ in range(n_layer)]
        )
        self.ln_post = LayerNorm(n_state)

    def forward(self, x: Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        x = (x + self.positional_embedding).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x
