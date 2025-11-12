from dataclasses import dataclass


@dataclass
class DecoderConfig:
    """
    Configuration options for beam search decoding.

    Attributes:
        beam_size: Number of hypotheses kept during beam search.
        lm_weight: Scaling factor for the external language model. (to be set)
        ilm_weight: Scaling factor for internal LM correction. (to be set)
    """
    beam_size: int = 12
    lm_weight: float = 0.0
    ilm_weight: float = 0.0