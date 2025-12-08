from dataclasses import dataclass
from typing import Sequence, Union, Tuple

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.protocols.has_name_protocol import \
    HasNameProtocol


@dataclass(frozen=True)
class EncoderConfig(HasNameProtocol):
    """
    Encoder configuration base dataclass.

    Can contain default values.
    """
    encoder_dim: int
    num_heads: int
    num_enc_layers: int

    aux_loss_layers: Sequence[int]
    # TODO: add loss ctc values

    rel_pos_clip: int = 16
    pos_emb_dropout: float = 0.1
    learnable_pos_emb: bool = True
    with_linear_pos: bool = False
    with_pos_bias: bool = False
    separate_pos_emb_per_head: bool = False

    # Spectrogram Augmentation
    specaug_start: Union[int, Tuple[int, int, int]] = 10,

    @property
    def name(self) -> str:
        return f"Conformer_l{self.num_enc_layers}"
        #return f"Conformer_l{self.num_enc_layers}_h_{self.num_heads}_d{self.encoder_dim}"


"""
Specific configurations set below.
"""


def encoder_baseline() -> EncoderConfig:
    return EncoderConfig(
        encoder_dim=512,
        num_heads=8,
        num_enc_layers=12,

        aux_loss_layers= (4, 8),

        specaug_start=(5_000, 15_000, 25_000),
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
