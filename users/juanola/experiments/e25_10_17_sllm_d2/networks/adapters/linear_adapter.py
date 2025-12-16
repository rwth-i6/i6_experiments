import torch
from torch import nn, Tensor


class LinearAdapter(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        with_bias: bool = True,
    ):
        super().__init__()

        self.speech_to_llm_proj = torch.nn.Linear(in_dim, out_dim, bias=with_bias)

    def forward(self, x: Tensor) -> Tensor:
        """

        :param x: input of shape [B, T, F]
        :return:
        """
        x = self.speech_to_llm_proj(x)
        return x
