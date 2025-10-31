from typing import TypedDict

from torch import Tensor


class Qwen2DecoderState(TypedDict):
    """Recurrent state of the Qwen2 HF transformers decoder."""

    input_embeds: Tensor
    past_key_values: Tensor