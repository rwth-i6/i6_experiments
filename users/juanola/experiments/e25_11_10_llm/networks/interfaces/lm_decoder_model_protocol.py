from abc import abstractmethod
from typing import Protocol

from returnn_common.nn import Tensor


class LmDecoderModelProtocol(Protocol):

    @abstractmethod
    def decode_seq_lm(self, x: Tensor, x_lens: Tensor) -> Tensor:
        """
        Forward the decoder for the entire sequence in `x`, discarding any intermediate state afterward.

        :param x: current sequence to be decoded
        :param x_lens: length of the seqs in x
        """