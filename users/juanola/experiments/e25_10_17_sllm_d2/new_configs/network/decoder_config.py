from dataclasses import dataclass, replace


@dataclass(frozen=True)
class DecoderConfig:
    """
    Decoder configuration base dataclass.

    Can contain default values.
    """
    architectures = [
        "Qwen2ForCausalLM"
    ]
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
    rope_theta: float = 1000000.0
    sliding_window: int = 131072
    tie_word_embeddings: bool = True
    torch_dtype: str = "bfloat16"
    transformers_version: str = "4.40.1"
    use_cache: bool = True
    use_sliding_window: bool = False
    vocab_size: int = 151936  # Modified in runtime!

    # ADDED
    mlp_dropout: float = 0.0


"""
Specific configurations set below.
"""


def decoder_baseline() -> DecoderConfig:
    return DecoderConfig()


def decoder_dropout() -> DecoderConfig:
    return DecoderConfig(
        attention_dropout=0.1,
        mlp_dropout=0.1,
    )


def decoder_dropout_tuned() -> DecoderConfig:
    return replace(decoder_baseline(),
                   rope_theta=0.1,
                   # TODO: check this...
                   )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
