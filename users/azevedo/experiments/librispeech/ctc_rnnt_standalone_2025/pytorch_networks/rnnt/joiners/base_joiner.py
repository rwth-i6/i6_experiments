import torch
from typing import Optional, Tuple, Literal
from i6_models.parts.dropout import BroadcastDropout



class Joiner(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            activation: str = "relu",
            dropout: float = 0.0,
            dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None,
    ) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
        self.dropout = BroadcastDropout(dropout_broadcast_axes=dropout_broadcast_axes, p=dropout)
        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation {activation}")

    def forward(
        self,
        source_encodings: torch.Tensor,
        source_lengths: torch.Tensor,
        target_encodings: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass for training.

        B: batch size;
        T: maximum source sequence length in batch;
        U: maximum target sequence length in batch;
        D: dimension of each source and target sequence encoding.

        Args:
            source_encodings (torch.Tensor): source encoding sequences, with
                shape `(B, T, D)`.
            source_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``source_encodings``.
            target_encodings (torch.Tensor): target encoding sequences, with shape `(B, U, D)`.
            target_lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                valid sequence length of i-th batch element in ``target_encodings``.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor):
                torch.Tensor
                    joint network output, with shape `(B, T, U, output_dim)`.
                torch.Tensor
                    output source lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 1 for i-th batch element in joint network output.
                torch.Tensor
                    output target lengths, with shape `(B,)` and i-th element representing
                    number of valid elements along dim 2 for i-th batch element in joint network output.
        """
        joint_encodings = source_encodings.unsqueeze(2).contiguous() + target_encodings.unsqueeze(1).contiguous()
        joint_encodings = self.dropout(joint_encodings)
        activation_out = self.activation(joint_encodings)
        output = self.linear(activation_out)
        return output, source_lengths, target_lengths