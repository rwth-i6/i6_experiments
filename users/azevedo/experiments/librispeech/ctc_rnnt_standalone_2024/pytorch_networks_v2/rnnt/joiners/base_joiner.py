import torch
from typing import Optional, Tuple, Literal
from dataclasses import dataclass

from ...base_config import BaseConfig

from ....pytorch_networks.rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1 import Joiner
from ...streamable_module import StreamableModule



@dataclass(kw_only=True)
class StreamableJoinerConfig(BaseConfig):
    input_dim: int
    output_dim: int 
    activation: str = "relu" 
    dropout: float = 0.0 
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None
    dual_mode: bool = True
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return StreamableJoinerConfig(**d)
    
    def module():
        return StreamableJoiner

# TODO
class StreamableJoiner(StreamableModule):
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