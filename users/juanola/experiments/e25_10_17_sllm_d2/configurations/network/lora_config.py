from dataclasses import dataclass


@dataclass(frozen=True)
class LoraConfig:
    """
    Encoder configuration base dataclass.

    Can contain default values.
    """

    target_modules: list[str]
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    use_rslora: bool

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass

"""
Specific configurations set below.
"""


def decoder_lora_v1() -> LoraConfig:
    return LoraConfig(
        target_modules=["q_proj", "v_proj"],
        r=320,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_rslora=True
    )

def decoder_small_lora_v1() -> LoraConfig:
    return LoraConfig(
        target_modules=["q_proj", "v_proj"],
        r=130,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        use_rslora=True
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
