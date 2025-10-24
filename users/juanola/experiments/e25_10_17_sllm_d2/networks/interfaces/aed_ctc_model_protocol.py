__all__ = ["AedCtcModelProtocol"]

from abc import abstractmethod
from typing import List, Protocol, Tuple

import returnn.frontend as rf
from returnn.tensor import Tensor as ReturnnTensor
from returnn.tensor import TensorDict
from torch import Tensor

class AedCtcModelProtocol(Protocol):
    blank_idx: int
    bos_idx: int
    eos_idx: int

    @abstractmethod
    def forward(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> Tuple[Tensor, List[Tensor], Tensor, Tensor]:
        """
        Forward the data through the encoder.

        :return: tuple of:
            - encoder output
            - list of CTC aux logits
            - length tensor of the output
            - mask tensor of the output
        """

    @abstractmethod
    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_mask: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterward.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        :param encoder_output: output of the encoder
        :param encoder_output_mask: padding mask of the encoder output
        """