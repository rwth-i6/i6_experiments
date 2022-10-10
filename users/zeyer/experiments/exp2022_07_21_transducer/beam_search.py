"""
Beam search for recog
"""

from __future__ import annotations
from typing import Protocol, Optional, Tuple
from returnn_common import nn


def beam_search(decoder: IDecoder, *, beam_size: int = 12) -> nn.Tensor:
    """
    Beam search.

    :return: alignment label sequences including beam
    """
    loop = nn.Loop(axis=decoder.target_spatial_dim)
    loop.max_seq_len = decoder.max_seq_len()
    loop.state.decoder = decoder.initial_state()
    loop.state.target = decoder.bos_label()
    with loop:
        log_prob, loop.state.decoder = decoder(loop.state.target, state=loop.state.decoder)
        loop.state.target = nn.choice(
            log_prob, input_type="log_prob",
            target=None, search=True, beam_size=beam_size,
            length_normalization=False)
        if not decoder.target_spatial_dim:
            loop.end(decoder.end(loop.state.target, state=loop.state.decoder), include_eos=decoder.include_eos)
        found = loop.stack(loop.state.target)
    return found


class IDecoder(Protocol):
    """
    Decoder interface.
    If there is any encoder output, this would already be computed and stored in here,
    as well as other necessary information, such as batch dims.
    """
    target_spatial_dim: Optional[nn.Dim] = None  # must be given when loop.unstack is used
    include_eos: bool  # eg true for transducer, false for AED

    def max_seq_len(self) -> nn.Tensor:
        """max seq len"""
        raise NotImplementedError

    def initial_state(self) -> nn.LayerState:
        """initial state"""
        raise NotImplementedError

    def bos_label(self) -> nn.Tensor:
        """begin-of-sequence (BOS) label"""
        raise NotImplementedError

    def __call__(self, prev_target: nn.Tensor, *, state: nn.LayerState) -> Tuple[nn.Tensor, nn.LayerState]:
        """return log_prob, new_state"""
        raise NotImplementedError

    def end(self, target: nn.Tensor, *, state: nn.LayerState) -> nn.Tensor:
        """
        Checks whether the (new) target, or (new) state reached the end-of-sequence (EOS).
        Only called when target_spatial_dim is not defined, i.e. when the length is dynamic.

        :return: bool tensor, True if EOS reached
        """
        raise nn.OptionalNotImplementedError
