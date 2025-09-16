import torch
from typing import Optional, Tuple, Literal, Union, Any, Callable
from dataclasses import dataclass

from ...base_config import BaseConfig

from .base_joiner import Joiner
from ...streamable_module import StreamableModule



@dataclass(kw_only=True)
class StreamableJoinerConfig(BaseConfig):
    input_dim: int
    output_dim: Union[int, Any] 
    activation: Union[str, torch.nn.Module, Callable[[torch.Tensor], torch.Tensor]] = "relu" 
    dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None
    dual_mode: bool
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return cls(**d)
    
    def module(self):
        return StreamableJoinerV1


class StreamableJoinerV1(StreamableModule):
    r"""Streamable RNN-T joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(self, cfg: StreamableJoinerConfig) -> None:
        super().__init__()
        out_dim = cfg.output_dim  # + 1  # NOTE: because of blank

        self.joiner_off = Joiner(
            input_dim=cfg.input_dim,
            output_dim=out_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            dropout_broadcast_axes=cfg.dropout_broadcast_axes
        )
        self.joiner_on = Joiner(
            input_dim=cfg.input_dim,
            output_dim=out_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            dropout_broadcast_axes=cfg.dropout_broadcast_axes
        ) if cfg.dual_mode else self.joiner_off

    def forward_offline(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_off(source_encodings, source_lengths, target_encodings, target_lengths)
    
    def forward_streaming(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_on(source_encodings, source_lengths, target_encodings, target_lengths)