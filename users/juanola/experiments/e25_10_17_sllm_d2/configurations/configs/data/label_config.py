from dataclasses import dataclass


@dataclass(frozen=True)
class LabelConfig:
    """
    Label configuration base dataclass.

    Can contain default values.
    """

    vocab_size: int
    bos_idx: int
    eos_idx: int


"""
Specific configurations set below.
"""


def label_baseline() -> LabelConfig:
    return LabelConfig(
        vocab_size=10_240,
        bos_idx=1,
        eos_idx=0,
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
