from dataclasses import dataclass, replace
import warnings

from ..protocols.has_name_protocol import HasNameProtocol


@dataclass(frozen=True)
class DecoderConfig(HasNameProtocol):
    """
    Decoder configuration base dataclass.

    Default parameters from HF Qwen-0.5B variant: https://huggingface.co/Qwen/Qwen2-0.5B/blob/main/config.json

    Can contain default values.
    """

    architectures = ["Qwen2ForCausalLM"]
    attention_dropout: float = 0.0
    bos_token_id: int = 151643  # Modified in runtime!
    eos_token_id: int = 151643  # Modified in runtime!
    hidden_act: str = "silu"
    hidden_size: int = 896
    initializer_range: float = 0.02
    intermediate_size: int = 4864
    max_position_embeddings: int = 131072
    max_window_layers: int = 24
    model_type: str = "qwen2"
    num_attention_heads: int = 14
    num_hidden_layers: int = 24
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-06
    rope_theta: float = 1_000_000.0
    # sliding_window: int = 131_072 # It's not being used because of use_sliding_window=False
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.40.1"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936  # Modified in runtime!

    # ADDED
    mlp_dropout: float = 0.0

    def __post_init__(self):
        assert 0 <= self.attention_dropout <= 1, f"attention_dropout ({self.attention_dropout}) should be between 0 and 1"
        assert 0 <= self.mlp_dropout <= 1, f"mlp_dropout ({self.mlp_dropout}) should be between 0 and 1"

    @property
    def name(self) -> str:
        return f"Qwen2_hl_{self.num_hidden_layers}"
"""
Parameter groups
"""

_QWEN_AUDIO_V2_DECODER_KWARGS = dict(
    max_position_embeddings=8192,
    rope_theta=10_000,
    rms_norm_eps=1e-5,
)

_DROPOUT_KWARGS = dict(
    attention_dropout=0.1,
    mlp_dropout=0.1,
)

"""
Specific configurations set below.
"""


def decoder_baseline() -> DecoderConfig:
    return DecoderConfig()


def decoder_dropout() -> DecoderConfig:
    return replace(decoder_baseline(), **_DROPOUT_KWARGS)


def decoder_dropout_tuned_v2() -> DecoderConfig:
    """
    Uses text params from HF Qwen-Audio-7B: https://huggingface.co/Qwen/Qwen2-Audio-7B/blob/main/config.json
    """
    return replace(
        decoder_dropout(),
        **_QWEN_AUDIO_V2_DECODER_KWARGS,
    )

def decoder_v2_tuned() -> DecoderConfig:
    """
    Uses text params from HF Qwen-Audio-7B: https://huggingface.co/Qwen/Qwen2-Audio-7B/blob/main/config.json
    NO DROPOUT!
    """
    return replace(
        decoder_baseline(),
        **_QWEN_AUDIO_V2_DECODER_KWARGS,
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
