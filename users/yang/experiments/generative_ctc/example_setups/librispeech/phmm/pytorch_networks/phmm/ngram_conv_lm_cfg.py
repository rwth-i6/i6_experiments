from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    embedding_dim: int = 128
    conv_channels: int = 256
    conv_kernel_size: int = 5
    projection_dim: int = 256
    dropout: float = 0.0
    pad_token_id: int = 0

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
