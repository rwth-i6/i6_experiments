import math
from dataclasses import dataclass

from i6_models.config import ModelConfiguration



@dataclass
class TransformerLinearConfig(ModelConfiguration):
    input_dim: int
    ff_dim: int
    output_dim: int
    dropout: float=0.0
    batch_first: bool=False

@dataclass
class TransformerMHSAConfig(ModelConfiguration):
    input_dim: int
    num_heads: int
    dropout: float=0.0
    batch_first: bool=False

@dataclass
class TransformerBlockConfig(ModelConfiguration):
    linear_config: TransformerLinearConfig
    mhsa_config: TransformerMHSAConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["linear_config"] = TransformerLinearConfig(**d["linear_config"])
        d["mhsa_config"] = TransformerMHSAConfig(**d["mhsa_config"])
        return cls(**d)

@dataclass
class TransformerLMConfig():
    embed_dim: int
    hidden_dim: int
    vocab_dim: int
    num_layers: int
    block_config: TransformerBlockConfig
    batch_first: bool=True
    dropout: float=0.0

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["block_config"] = TransformerBlockConfig.from_dict(d["block_config"])
        return cls(**d)

