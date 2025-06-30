import torch
from torch import nn
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple, Union, Any

from i6_models.parts.dropout import BroadcastDropout
from ...base_config import BaseConfig



@dataclass(kw_only=True)
class LSTMPredictorConfig(BaseConfig):
    symbol_embedding_dim: int
    emebdding_dropout: float
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_dropout: float

    label_target_size: Union[int, Any]
    output_dim: int
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return LSTMPredictorConfig(**d)
    
    def module(self):
        return LSTMPredictor
    

class LSTMPredictor(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Taken from torchaudio
    """

    def __init__(self, cfg: LSTMPredictorConfig) -> None:
        """
        :param cfg: model configuration for the predictor
        :param label_target_size: shared value from model
        :param output_dim: shared value from model
        """
        super().__init__()

        label_target_size = cfg.label_target_size
        output_dim = cfg.output_dim
        dropout_broadcast_axes = cfg.dropout_broadcast_axes

        self.embedding = torch.nn.Embedding(label_target_size, cfg.symbol_embedding_dim)
        self.embedding_dropout = BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=cfg.emebdding_dropout)
        self.input_layer_norm = torch.nn.LayerNorm(cfg.symbol_embedding_dim)
        self.lstm_layers = torch.nn.ModuleList(
            [
                nn.LSTM(
                    input_size=cfg.symbol_embedding_dim if idx == 0 else cfg.lstm_hidden_dim,
                    hidden_size=cfg.lstm_hidden_dim,
                )
                for idx in range(cfg.num_lstm_layers)
            ]
        )
        self.dropout = BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=cfg.lstm_dropout)
        self.linear = torch.nn.Linear(cfg.lstm_hidden_dim, output_dim)
        self.output_layer_norm = torch.nn.LayerNorm(output_dim)

        self.lstm_dropout = cfg.lstm_dropout

    def forward(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        state: Optional[List[List[torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.

        Args:
            input (torch.Tensor): target sequences, with shape `(B, U)` and each element
                mapping to a target symbol, i.e. in range `[0, num_symbols)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.
            state (List[List[torch.Tensor]] or None, optional): list of lists of tensors
                representing internal state generated in preceding invocation
                of ``forward``. (Default: ``None``)

        Returns:
            (torch.Tensor, torch.Tensor, List[List[torch.Tensor]]):
                torch.Tensor
                    output encoding sequences, with shape `(B, U, output_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid elements for i-th batch element in output encoding sequences.
                List[List[torch.Tensor]]
                    output states; list of lists of tensors
                    representing internal state generated in current invocation of ``forward``.
        """
        input_tb = input.permute(1, 0)
        embedding_out = self.embedding(input_tb)
        embedding_out = self.embedding_dropout(embedding_out)
        input_layer_norm_out = self.input_layer_norm(embedding_out)

        lstm_out = input_layer_norm_out
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(lstm_out, None if state is None else [s.permute(1, 0, 2) for s in state[layer_idx]])
            lstm_out = self.dropout(lstm_out)
            state_out.append([s.permute(1, 0, 2) for s in lstm_state_out])

        linear_out = self.linear(lstm_out)
        output_layer_norm_out = self.output_layer_norm(linear_out)
        return output_layer_norm_out.permute(1, 0, 2), lengths, state_out