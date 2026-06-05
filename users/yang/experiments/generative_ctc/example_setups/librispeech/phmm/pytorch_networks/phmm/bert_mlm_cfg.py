from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 256
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    unk_token_id: int = 1
    cls_token_id: int = 2
    sep_token_id: int = 3
    mask_token_id: int = 4
    mlm_probability: float = 0.15
    mask_replace_probability: float = 0.8
    random_replace_probability: float = 0.1
    output_bias: bool = False
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
