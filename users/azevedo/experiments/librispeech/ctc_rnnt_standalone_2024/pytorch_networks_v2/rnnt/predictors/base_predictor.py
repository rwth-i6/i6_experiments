import torch
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from ...base_config import BaseConfig



@dataclass(kw_only=True)
class PredictorConfig(BaseConfig):
    symbol_embedding_dim: int
    emebdding_dropout: float
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_dropout: float

    label_target_size: int
    output_dim: int
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PredictorConfig(**d)
    
    def module():
        return Predictor
    

class Predictor(torch.nn.Module):
    def __init__(
        self,
        cfg: PredictorConfig,
    ) -> None:
        raise NotImplementedError

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        raise NotImplementedError