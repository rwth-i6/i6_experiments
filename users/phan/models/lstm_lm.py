import torch
from torch import nn
from dataclasses import dataclass

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import Import

@dataclass
class LSTMLMConfig(ModelConfiguration):
    vocab_dim: int
    embed_dim: int
    hidden_dim: int
    n_lstm_layers: int
    bias: bool = True
    dropout: float = 0.0

class LSTMLM(nn.Module):
    """
    Simple LSTM LM with an embedding, an LSTM, and a final linear
    """
    def __init__(self, step: int, cfg: LSTMLMConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.vocab_dim, cfg.embed_dim, bias=True)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_lstm_layers,
            bias=cfg.bias,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=False,
        )
        # if cfg.dropout > 0 and cfg.n_lstm_layers == 1:
        #     self.dropout = nn.Dropout(0.1)
        self.final_linear = nn.Linear(cfg.hidden_dim, cfg.vocab_dim, bias=True)

    def forward(self, x):
        """
        Return log probs of each beam at each time step
        x: (B, S, F)
        """
        x = self.embed(x)
        batch_size = x.shape[0]
        h0 = torch.zeros((self.cfg.n_lstm_layers, batch_size, self.cfg.hidden_dim), device=x.device).detach()
        c0 = torch.zeros_like(h0, device=x.device).detach()
        x, _ = self.lstm(x, (h0, c0))
        # if self.dropout:
        #     x = self.dropout(x)
        x = self.final_linear(x)
        x = x.log_softmax(dim=-1)
        return x

def get_train_serializer(
    model_config: LSTMLMConfig,
    train_step_package: str
) -> Collection:
    # pytorch_package = __package__.rpartition(".")[0]
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{LSTMLM.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            Import(f"{train_step_package}.train_step"),
        ],
    )
