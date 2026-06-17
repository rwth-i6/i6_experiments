from dataclasses import dataclass

from i6_models.config import ModelConfiguration

@dataclass
class ModelConfig():
    vocab_dim: int
    embed_dim: int
    hidden_dim: int
    n_lstm_layers: int
    init_args: dict
    bias: bool = True
    use_bottle_neck: bool = False
    bottle_neck_dim: int = 512
    dropout: float = 0.0
