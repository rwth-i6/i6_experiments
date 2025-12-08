from dataclasses import dataclass


@dataclass(frozen=True)
class EncoderConfig:
    """
    Encoder configuration base dataclass.

    Can contain default values.
    """
    encoder_dim: int
    num_heads: int

    rel_pos_clip: int = 16
    pos_emb_dropout: float = 0.1
    learnable_pos_emb: bool = True
    with_linear_pos: bool = False
    with_pos_bias: bool = False
    separate_pos_emb_per_head: bool = False


"""
Specific configurations set below.
"""


def encoder_baseline() -> EncoderConfig:
    return EncoderConfig(
        encoder_dim=512,
        num_heads=8,
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
