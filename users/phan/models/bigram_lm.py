"""Simple feed-forward bigram LM"""

import torch
from torch import nn
from dataclasses import dataclass

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.phan.utils import init_linear, init_lstm


@dataclass
class BigramLMConfig(ModelConfiguration):
    """
    n_hidden_layers does not count the embedding and the final linear
    """
    vocab_dim: int
    embed_dim: int
    hidden_dim: int
    n_hidden_layers: int # does not count the embedding!
    dropout: float = 0.1

class BigramLM(nn.Module):
    """
    Simple 2-gram FFN LM
    """
    def __init__(self, step: int, cfg: BigramLMConfig, **kwargs):
        super().__init__()
        assert cfg.n_hidden_layers >= 1, "There must be at least one hidden layers"
        self.cfg = cfg
        self.embed = nn.Linear(cfg.vocab_dim, cfg.embed_dim, bias=False)
        self.hidden_layers = nn.ModuleList()
        first_linear = nn.Linear(cfg.embed_dim, cfg.hidden_dim)
        self.hidden_layers.append(first_linear)
        if cfg.dropout > 0.:
            first_dropout = nn.Dropout(cfg.dropout)
            self.hidden_layers.append(first_dropout)
        for i in range(cfg.n_hidden_layers-1):
            next_linear = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=True)
            self.hidden_layers.append(next_linear)
            if cfg.dropout > 0.:
                next_dropout = nn.Dropout(cfg.dropout)
                self.hidden_layers.append(next_dropout)
        self.final_linear = nn.Linear(cfg.hidden_dim, cfg.vocab_dim, bias=True)

        # Init
        init_func = nn.init.normal_
        init_args = {"mean": 0.0, "std": 0.1}
        init_linear(self.embed, init_func, init_args, bias=False)
        init_linear(self.final_linear, init_func, init_args)
        for k in range(len(self.hidden_layers)):
            if isinstance(self.hidden_layers[k], nn.Linear):
                init_linear(self.hidden_layers[k], init_func, init_args)

    def forward(self, x):
        """
        x: (..., F) [1-hot encoding]
        """
        x = self.embed(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.final_linear(x)
        x = x.log_softmax(dim=-1)
        return x

def get_train_serializer(
    model_config: BigramLMConfig,
    train_step_package: str
) -> Collection:
    # pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{BigramLM.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{train_step_package}.train_step"),
        ],
    )
