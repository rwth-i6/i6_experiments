__all__ = ["BaseEncoderDecoderModel"]

from abc import abstractmethod
from typing import Generic

from torch import Tensor

from .label_scorer_protocol import LabelScorerProtocol, State

class BaseEncoderDecoderModel(LabelScorerProtocol[State], Generic[State]):
    """
    Interface for an encoder and a decoder that scores labels.

    Can encode acoustic data into a higher level representation and, as part of that, generate
    an initial decoder state.
    This state also stores the higher level representations.
    """

    @abstractmethod
    def forward_encoder(self, raw_audio: Tensor, raw_audio_lens: Tensor) -> State:
        """
        Forward the raw audio data through the encoder and initialize decoder state from it.

        :param raw_audio: audio data, shape [B,T,1]
        :param raw_audio_lens: lengths of the audio in `raw_audio`, shape [B,]
        :return: decoder state initialized by passing the `raw_audio` through the encoder and
            initializing a fresh decoder state with it.
        """
        raise NotImplementedError
