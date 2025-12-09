from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BeamSearchConfig:
    """
    Configuration options for beam search decoding.

    Attributes:
        beam_size: Number of hypotheses kept during beam search.
        lm_weight: Scaling factor for the external language model.
        ilm_weight: Scaling factor for internal LM correction.
    """
    beam_size: int
    lm_weight: Optional[float]
    ilm_weight: Optional[float]
    prior_scale: Optional[float]


def beam_search_baseline() -> BeamSearchConfig:
    return BeamSearchConfig(
        beam_size=12,
        lm_weight=None, # TODO: improve this, it is set at run time, and multiple values are tested...
        ilm_weight=None,
        prior_scale=None,
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)