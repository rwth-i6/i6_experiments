import torch
from torch import nn 
from dataclasses import dataclass
from typing import Optional, Literal, Tuple, List, Union, Any

from i6_models.parts.dropout import BroadcastDropout
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from ...base_config import BaseConfig


@dataclass(kw_only=True)
class FFNNPredictorConfig(BaseConfig):
    symbol_embedding_dim: int
    emebdding_dropout: float
    context_size: int
    num_layers: int
    hidden_dim: int
    prediction_dropout: float
    prediction_act: Union[str, nn.Module]
    label_target_size: Union[int, Any]
    output_dim: int
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()

        if d["prediction_act"].lower() == "relu":
            d["prediction_act"] = torch.nn.ReLU()
        elif d["prediction_act"].lower() == "tanh":
            d["prediction_act"] = torch.nn.Tanh()
        else:
            raise NotImplementedError(
                "Activation %s not available for FFNNPredictor." % d["prediction_act"]
            )
        
        return FFNNPredictorConfig(**d)
    
    def module(self):
        return FFNNPredictor


class FFNNPredictor(nn.Module):
    def __init__(self, cfg: FFNNPredictorConfig) -> None:
        super().__init__()
        label_target_size = cfg.label_target_size
        output_dim = cfg.output_dim
        dropout_broadcast_axes = cfg.dropout_broadcast_axes

        self.embedding = torch.nn.Embedding(label_target_size, cfg.symbol_embedding_dim)
        self.embedding_dropout = BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=cfg.emebdding_dropout)
        self.input_layer_norm = torch.nn.LayerNorm(cfg.symbol_embedding_dim)
        self.context_size = cfg.context_size
        
        layers = []
        prev_size = cfg.symbol_embedding_dim * cfg.context_size  # `context_size` embed vecs contribute to one out
        for _ in range(cfg.num_layers):
            layers.append(torch.nn.Dropout(cfg.prediction_dropout))
            layers.append(torch.nn.Linear(prev_size, cfg.hidden_dim))
            layers.append(cfg.prediction_act)
            prev_size = cfg.hidden_dim

        self.layers = nn.ModuleList(layers)

        self.final_linear = nn.Linear(prev_size, output_dim)

    def add_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        B: batch size;
        U: maximum sequence length in batch;
        D: feature dimension of each input sequence element.
        H: context size = self.context_size

        :param context: (B, U, E)

        :return: (B, U, H, E) 
        """
        if self.context_size == 1:
            return context.unsqueeze(-2)  # [B, U, H, E] = [B, U, 1, E]

        context_list = [context]
        for _ in range(self.context_size - 1):
            context_list.append(
                torch.nn.functional.pad(context_list[-1][:, :-1, :], pad=(0, 0, 1, 0), value=0)
            )  # [B, U, E] shifted by < context_size
        context_stack = torch.stack(context_list, dim=-2)  # [B, U, H, E]

        return context_stack  # [B, U, H, E]

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
        # print(f"{input.shape = }")
        state_out = []
        # NOTE: during training there is no state
        if state and self.context_size > 1:
            # print(f"{state[0][0].shape = }")
            input = torch.stack((*state[0], input), dim=1).squeeze(-1)  # (B, U)
            state_out = list(input.unbind(dim=1))  # [(B), (B), ..., (B)]
            state_out = [[s.unsqueeze(-1) for s in state_out[-self.context_size+1:]]]
        else:
            state_out = [[input]]

        # print(f"after stacking: {input.shape}")
        embedding_out = self.embedding(input)  # (B, U, E)
        embedding_context = self.add_context(embedding_out)  # (B, U, H, E)
        embedding_context = self.embedding_dropout(embedding_context)
        embedding_context_norm = self.input_layer_norm(embedding_context)

        x = embedding_context_norm.flatten(-2, -1)  # (B, U, H*E)
        for layer in self.layers:
            x = layer(x)  # (B, U, F)

        x = self.final_linear(x)  # (B, U, P)
        if state:
            x = x[:, [-1]]

        # print(f"{x.shape = }")

        return x, lengths, state_out
