from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
from torch import nn

from .scf import FeatureExtractionConfig


class Fp32GroupNorm(nn.GroupNorm):
    """
    Copied from fairseq
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = torch.nn.functional.group_norm(
            input.float(),
            self.num_groups,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class Fp32LayerNorm(nn.LayerNorm):
    """
    Copied from fairseq
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = torch.nn.functional.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)


class TransposeLast(nn.Module):
    """
    Copied from fairseq
    """
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


@dataclass
class Wav2vecFeatureExtractionV1Config(FeatureExtractionConfig):
    conv_layers: List[Tuple[int, int, int]]
    dropout: float = 0.0
    mode: str = "default"
    conv_bias: bool = False
    activation: Optional[Union[str, nn.Module]] = "GELU"

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation = d.pop("activation")
        if activation is None:
            pass
        elif activation == "GELU":
            from torch.nn import GELU
            activation = GELU()
        elif activation == "ReLU":
            from torch.nn import ReLU
            activation = ReLU()
        else:
            assert False, f"Unsupported activation {activation}"
        d["activation"] = activation
        return cls(**d)


class Wav2vecFeatureExtractionV1(nn.Module):
    """
    Mostly copy of ConvFeatureExtractionModel from fairseq
    """
    def __init__(
        self,
        cfg: Wav2vecFeatureExtractionV1Config,
    ):
        super().__init__()

        assert cfg.mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=cfg.dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    cfg.activation,
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=cfg.dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    cfg.activation,
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=cfg.dropout), cfg.activation)

        in_d = 1
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(cfg.conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=cfg.mode == "layer_norm",
                    is_group_norm=cfg.mode == "default" and i == 0,
                    conv_bias=cfg.conv_bias,
                )
            )
            in_d = dim

    def forward(self, raw_audio: torch.Tensor, length: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T]
        :param length: in samples [B]
        :return features as [B,T',F]
        """
        x = torch.unsqueeze(raw_audio, 1)  # [B,1,T]
        for conv in self.conv_layers:
            x = conv(x)
            length = ((length - conv[0].kernel_size[0]) / conv[0].stride[0] + 1).int()
        x = x.transpose(1, 2)  # [B,T',F]

        return x, length
