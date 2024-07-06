import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import Import, PartialImport
from i6_experiments.users.phan.utils import init_linear, init_lstm


@dataclass
class LSTMLMConfig(ModelConfiguration):
    vocab_dim: int # Synonym to input dim. This naming is dumb though.
    embed_dim: int
    hidden_dim: int
    n_lstm_layers: int
    output_dim: Optional[int] = None
    bias: bool = True
    dropout: float = 0.0
    bidirectional: bool = False

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
            bidirectional=cfg.bidirectional,
        )
        # if cfg.dropout > 0 and cfg.n_lstm_layers == 1:
        #     self.dropout = nn.Dropout(0.1)
        if cfg.output_dim is None:
            output_dim = cfg.vocab_dim
        else:
            output_dim = cfg.output_dim
        if not cfg.bidirectional:
            lstm_out_dim = cfg.hidden_dim
        else:
            lstm_out_dim = 2*cfg.hidden_dim
        self.final_linear = nn.Linear(lstm_out_dim, output_dim, bias=True)
        init_func = nn.init.normal_
        init_args = {"mean": 0.0, "std": 0.1}
        init_lstm(self.lstm, cfg.n_lstm_layers, init_func, init_args)
        init_linear(self.embed, init_func, init_args)
        init_linear(self.final_linear, init_func, init_args)

    def forward(self, x, seq_lengths=None):
        """
        Return log probs of each beam at each time step
        x: (B, S, F)
        seq_lengths: (B,) sequence lengths, needed for bidirectional case
        """
        x = self.embed(x)

        if self.cfg.bidirectional:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x,
                seq_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            x, _ = self.lstm(x)
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, padding_value=0.0) 
        else:
            x, _ = self.lstm(x)

        x = self.final_linear(x)
        x = x.log_softmax(dim=-1)
        return x

def get_train_serializer(
    model_config: LSTMLMConfig,
    train_step_package: str,
    partial_train_step: bool = False,
    partial_kwargs: dict = {},
) -> Collection:
    """
    :param partial_train_step: Is the train step a partial function?
    :param partial_kwargs: Contains two dicts: "hashed_arguments" and "unhashed_arguments"
    """
    # pytorch_package = __package__.rpartition(".")[0]
    if partial_train_step:
        train_step_import = PartialImport(
            code_object_path=f"{train_step_package}.train_step",
            unhashed_package_root=train_step_package,
            import_as="train_step",
            **partial_kwargs,
        )
    else:
        train_step_import = Import(f"{train_step_package}.train_step")
    return get_basic_pt_network_serializer(
        module_import_path=f"{__name__}.{LSTMLM.__name__}",
        model_config=model_config,
        additional_serializer_objects=[
            train_step_import,
        ],
    )
