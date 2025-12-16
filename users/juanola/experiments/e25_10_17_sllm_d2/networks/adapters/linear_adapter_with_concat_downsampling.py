import torch
from torch import nn, Tensor
import torch.nn.functional as F

# used in SLAM (https://arxiv.org/pdf/2402.08846)
class LinearAdapterWithConcatDownsampling(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            *,
            downsampling_factor: int = 1, # TODO: move
            with_bias: bool = True,
    ):
        super().__init__()

        self.downsampling_factor = downsampling_factor
        self.speech_to_llm_proj = torch.nn.Linear(in_dim * downsampling_factor, out_dim, bias=with_bias)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: input of shape [B, T, F]
        :return:
        """
        batch, time, feat = x.shape  # noqa

        # make sure T is even
        mod = time % self.downsampling_factor
        if mod != 0:
            x = F.pad(x, (0, 0, 0, self.downsampling_factor - mod), "constant", value=0.0)
            time = x.size(1)

        # reshape into groups of downsampling_factor along the time axis
        x = x.view(batch, time // self.downsampling_factor, self.downsampling_factor, feat)  # (B, T//down, down, F)

        # flatten the last two dims
        x = x.view(batch, time // self.downsampling_factor, feat * self.downsampling_factor)  # (B, T//down, down*F)

        x = self.speech_to_llm_proj(x)
        return x