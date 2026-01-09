from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BeamSearchConfig: # TODO: maybe merge with search?
    """
    Configuration options for beam search decoding.

    Attributes:
        beam_size: Number of hypotheses kept during beam search.
    """
    beam_size: int

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


def beam_search_baseline() -> BeamSearchConfig:
    return BeamSearchConfig(beam_size=12)


def greedy() -> BeamSearchConfig:
    return BeamSearchConfig(beam_size=1)

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)