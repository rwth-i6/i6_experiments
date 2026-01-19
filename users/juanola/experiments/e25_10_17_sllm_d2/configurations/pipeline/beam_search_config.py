from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BeamSearchConfig:  # TODO: maybe merge with search?
    """
    Configuration options for beam search decoding.

    Attributes:
        beam_size: Number of hypotheses kept during beam search.
    """

    beam_sizes: list[int]

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


def beam_search_baseline() -> BeamSearchConfig:
    return BeamSearchConfig(beam_sizes=[12])


def beam_search_baseline_v2() -> BeamSearchConfig:
    """
    best beam over SLLM_tuned_dropout_v2 setup
    :return:
    """
    return BeamSearchConfig(beam_sizes=[8])


def beam_search_multiple_beams() -> BeamSearchConfig:
    return BeamSearchConfig(beam_sizes=[1, 2, 4, 6, 8, 10, 12])


def greedy() -> BeamSearchConfig:
    return BeamSearchConfig(beam_sizes=[1])


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
