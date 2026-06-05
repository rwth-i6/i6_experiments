from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 256
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    dropout: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-5
    pad_token_id: int = 0
    output_bias: bool = False
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
