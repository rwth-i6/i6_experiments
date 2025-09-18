import torch
from typing import Optional, Tuple, Literal, Union, Any, Callable
from dataclasses import dataclass

from i6_models.parts.dropout import BroadcastDropout
from ...base_config import BaseConfig
from ...streamable_module import StreamableModule



class MonotonicJoinerV2(torch.nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            activation: str = "relu",
            dropout: float = 0.0,
            dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None,
    ) -> None:
        super().__init__()

        if activation.lower() == "relu":
            joiner_activation = torch.nn.ReLU()
        elif activation.lower() == "tanh":
            joiner_activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

        self.joiner = torch.nn.Sequential(
            BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=dropout),
            torch.nn.Linear(input_dim, hidden_dim, bias=True),
            joiner_activation,
            BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=dropout),
            torch.nn.Linear(hidden_dim, output_dim, bias=True),
        )
        
    def forward(
        self,
        source_encodings: torch.Tensor,  # [B, T, E]
        source_lengths: torch.Tensor,  # [B]
        target_encodings: torch.Tensor,  # [B, S, P]
        target_lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # final logits [T_1 * (S_1) + T_2 * (S_2) + ... + T_B * (S_B), C]
        # taken from example_setups/seq2seq_rasr_2025/model_pipelines/bpe_ffnn_transducer/pytorch_modules.py
        
        source_encodings = source_encodings.to(dtype=torch.float32)
        target_encodings = target_encodings.to(dtype=torch.float32)
        batch_tensors = []
        for b in range(source_encodings.size(0)):
            valid_enc = source_encodings[b, : source_lengths[b], :]  # [T_b, E]
            valid_pred = target_encodings[b, : target_lengths[b], :]  # [S_b, P]  NOTE: changed to S_b from S_b+1

            expanded_enc = valid_enc.unsqueeze(1).expand(-1, valid_pred.size(0), -1)  # [T_b, S_b, E]
            expanded_pred = valid_pred.unsqueeze(0).expand(
                valid_enc.size(0), -1, -1
            )  # [T_b, S_b, P]

            combination = torch.concat([expanded_enc, expanded_pred], dim=-1)  # [T_b, S_b, E+P]

            batch_tensors.append(combination.reshape(-1, combination.size(2)))  # [T_b*(S_b), E+P]

        joint_input = torch.concat(batch_tensors, dim=0)  # [T_1*(S_1) + ... + T_B*(S_B), E+P]
        joint_output = self.joiner.forward(joint_input)  # [T_1*(S_1) + ... + T_B*(S_B), V]

        return joint_output, source_lengths, target_lengths


@dataclass(kw_only=True)
class StreamableMonotonicJoinerV2Config(BaseConfig):
    input_dim: int
    hidden_dim: int
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
        return StreamableMonotonicJoinerV2


class StreamableMonotonicJoinerV2(StreamableModule):
    r"""Streamable RNN-T joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(self, cfg: StreamableMonotonicJoinerV2Config) -> None:
        super().__init__()
        out_dim = cfg.output_dim  # + 1  # NOTE: because of blank

        self.joiner_off = MonotonicJoinerV2(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=out_dim,
            activation=cfg.activation,
            dropout=cfg.dropout,
            dropout_broadcast_axes=cfg.dropout_broadcast_axes
        )
        self.joiner_on = MonotonicJoinerV2(
            input_dim=cfg.input_dim,
            hidden_dim=cfg.hidden_dim,
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
