"""Structural type for the decoder objects the search loops drive."""

from typing import Any, Protocol, Tuple, TypeVar

import torch

StateT = TypeVar("StateT")


class SearchDecoder(Protocol[StateT]):
    """Autoregressive interface the greedy and beam search loops require."""

    def text_embed(self, text_codes: torch.Tensor) -> torch.Tensor:
        ...

    def initial_state(self, *, batch_size: int, device: torch.device) -> StateT:
        ...

    def step(self, combined: torch.Tensor, state: StateT) -> Tuple[torch.Tensor, StateT]:
        ...

    def reorder_state(self, state: StateT, beam_backrefs: torch.Tensor) -> StateT:
        ...


class LabelSyncDecoder(Protocol[StateT]):
    """Decoder the label-synchronous search steps, conditioned on everything it reads when its state is built."""

    def initial_state(self, enc: torch.Tensor, enc_lens: torch.Tensor, beam: int) -> StateT:
        ...

    def step(self, label: torch.Tensor, state: StateT) -> Tuple[torch.Tensor, StateT]:
        ...

    def reorder_state(self, state: StateT, beam_backrefs: torch.Tensor) -> StateT:
        ...


class LabelSyncSearchModel(Protocol):
    """What the label-synchronous search reads of a model, its start and end ids and its stepping decoder."""

    @property
    def bos_id(self) -> int:
        ...

    @property
    def eos_id(self) -> int:
        ...

    @property
    def decoder(self) -> LabelSyncDecoder[Any]:
        ...


class MaskedStepDecoder(Protocol[StateT]):
    """Search decoder whose step leaves the rows with ``active`` False untouched, so histories may differ in length."""

    def text_embed(self, text_codes: torch.Tensor) -> torch.Tensor:
        ...

    def initial_state(self, *, batch_size: int, device: torch.device) -> StateT:
        ...

    def step(self, combined: torch.Tensor, state: StateT, active: torch.Tensor) -> Tuple[torch.Tensor, StateT]:
        ...

    def reorder_state(self, state: StateT, beam_backrefs: torch.Tensor) -> StateT:
        ...
