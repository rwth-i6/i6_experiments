__all__ = ["Model"]

from typing import Callable, Tuple

from torch import Tensor

from i6_models.assemblies.transformer.transformer_decoder_v1 import (
    TransformerDecoderV1,
    TransformerDecoderV1State,
)

from .conformer_aed_discrete_shared_v1 import Model as AEDModel


class Model(AEDModel):
    """
    CTC-only variant of the shared AED model: same Conformer encoder (`ConformerRelPosEncoderV1`)
    and embeddings, but no decoder at all — the output comes from the encoder-side CTC aux heads.
    """

    # no decoder is built at all (see `AEDModel.__init__`), so `print_param_summary`
    # (called at the end of the base `__init__`) already reports encoder-only counts.
    has_decoder = False

    def decode_text_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        raise NotImplementedError

    def decode_audio_seq(
        self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def decode_seq(self, x: Tensor, x_lens: Tensor, encoder_output: Tensor, encoder_output_lens: Tensor) -> Tensor:
        raise NotImplementedError

    def forward_encoder(
        self, indices: Tensor, indices_lens: Tensor, decoder: TransformerDecoderV1, forward_func: Callable
    ) -> TransformerDecoderV1State:
        raise NotImplementedError

    def step_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State, decoder: TransformerDecoderV1
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError

    def step_audio_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError

    def step_text_decoder(
        self, labels: Tensor, state: TransformerDecoderV1State
    ) -> Tuple[Tensor, TransformerDecoderV1State]:
        raise NotImplementedError
