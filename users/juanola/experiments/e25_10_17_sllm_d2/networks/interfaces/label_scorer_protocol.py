__all__ = ["State", "LabelScorerProtocol"]


from abc import abstractmethod

from typing import Generic, Protocol, Tuple, TypeVar


from torch import Tensor

State = TypeVar("State")

class LabelScorerProtocol(Protocol, Generic[State]):
    """
    Interface for scoring labels.

    Given existing labels (initial: BOS) and the recurrent state, scores labels.

    Generic over a recurrent decoder state of type `State`.
    `State` can be any PyTree (nested structure composed of the primitive containers list, dict and tuple).

    All tensors in the state are expected to have shape [Batch, Beam, ...Features].
    The data is reprocessed during decoding when e.g. selecting beam backrefs.
    To store beam-independent data like the encoder output, the Beam entry can also be set to 1.
    In that case the values are treated as broadcast over all beams and left untouched.
    """

    bos_idx: int
    """Index of the beginning-of-sentence label."""
    eos_idx: int
    """Index of the end-of-sentence label."""
    num_labels: int
    """Number of labels including BOS/EOS."""

    @abstractmethod
    def step_decoder(self, labels: Tensor, state: State) -> Tuple[Tensor, State]:
        """
        Run one decoder step, given the labels and recurrent state.

        :param labels: current labels, shape [Batch, Beam, Time=1],
            sparse dim L
        :param state: recurrent decoder state, where the initial state is obtained
            from `forward_encoder`.
        :return: tuple of:
            - logits of the next labels, shape [Batch, Beam, Time=1, L]
            - decoder state for decoding the next step.
        """
        raise NotImplementedError