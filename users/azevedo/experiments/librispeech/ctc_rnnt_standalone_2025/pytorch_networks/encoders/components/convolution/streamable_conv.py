import torch
from torch import nn
from typing import Optional, List, Callable, Union
import math
from copy import deepcopy
from dataclasses import dataclass

# from i6_models.parts.conformer import ConformerConvolutionV1Config
from i6_models.parts.conformer.norm import LayerNormNC

from ..streamable_layernorm import StreamableLayerNormV1
from ....streamable_module import StreamableModule
from ....base_config import BaseConfig



@dataclass(kw_only=True)
class StreamableConformerConvolutionV1Config(BaseConfig):
    """
    Attributes:
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    channels: int
    kernel_size: int
    dropout: float
    activation: Union[str, nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    # norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        # TODO: change for more flexibility
        assert d["activation"].lower() == "silu"
        d["activation"] = torch.nn.functional.silu
        return StreamableConformerConvolutionV1Config(**d)


class StreamableConformerConvolutionV1(StreamableModule):
    def __init__(self, model_cfg: StreamableConformerConvolutionV1Config, dual_mode: bool):
        super().__init__()
        model_cfg.check_valid()

        self.pointwise_conv1 = nn.Linear(in_features=model_cfg.channels, out_features=2 * model_cfg.channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
        )
        self.pointwise_conv2 = nn.Linear(in_features=model_cfg.channels, out_features=model_cfg.channels)
        self.layer_norm = StreamableLayerNormV1(model_cfg.channels, dual_mode=dual_mode)
        self.norm = LayerNormNC(model_cfg.channels)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward_offline(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B, T, F]

        :return: torch.Tensor of shape [B, T, F]
        """

        tensor = self.layer_norm(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.depthwise_conv(tensor)

        tensor = self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pointwise_conv2(tensor)

        return self.dropout(tensor)

    def forward_streaming(self, tensor: torch.Tensor, lookahead_sz: int, carry_over_size: int) -> torch.Tensor:
        """
        :param tensor: [B, N, C, F]
        :param lookahead_sz: number of future frames in chunk
        :param carry_over_size: number of past chunks we can convolve over

        :return: [B, N, C, F]
        """
        assert tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = tensor.shape
        kernel_radius = self.depthwise_conv.kernel_size[0] // 2

        conv_in = torch.zeros(
            bsz, num_chunks, kernel_radius + chunk_sz, tensor.size(-1),
            device=tensor.device
        )

        # conv convolves over multiple past chunks w/o their future-acoustic-context (fac)
        tensor = tensor.flatten(1, 2)  # [B, N*C, F]
        chunks_no_fac = tensor.unfold(
            1, chunk_sz-lookahead_sz, chunk_sz
        ).swapaxes(-2, -1)  # [B, N, C-R, F]

        for i in range(num_chunks):
            if i > 0:
                # calc how many past chunks needed for conv
                conv_carry = math.ceil(kernel_radius / (chunk_sz - lookahead_sz))
                # don't go over predefined carryover
                conv_carry = min(carry_over_size, conv_carry)
                carry_no_fac = chunks_no_fac[:, max(0, i-conv_carry): i].flatten(1, 2)
                carry_no_fac = carry_no_fac[:, :kernel_radius]

                conv_in[:, i, -chunk_sz-carry_no_fac.size(1):-chunk_sz] = carry_no_fac
                    
            t_step = i * chunk_sz
            # add chunk itself
            conv_in[:, i, -chunk_sz:] = tensor[:, t_step: t_step+chunk_sz]

        conv_in = conv_in.flatten(0, 1)  # [B*N, KRN//2 + C, F]

        out = self.forward_offline(conv_in)
        out = out[:, -chunk_sz:]  # [B*N, C, F]
        out = out.view(bsz, num_chunks, chunk_sz, -1)

        return out

    def infer(self, x: torch.Tensor, states: Optional[List[torch.Tensor]], chunk_sz: int, lookahead_sz: int) -> torch.Tensor:
        if states is not None:
            states_no_fac = [layer_out[:-lookahead_sz] for layer_out in states]
            x = torch.cat((*states_no_fac, x), dim=0).unsqueeze(0)
            x = self.forward_offline(x)[:, -chunk_sz:]  # [1, C+R, F]
        else:
            x = x.unsqueeze(0)
            x = self.forward_offline(x)  # [1, C+R, F]

        return x.squeeze(0)
