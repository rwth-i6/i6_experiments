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
    init_args: dict
    bias: bool = True
    use_bottle_neck: bool = False
    bottle_neck_dim: int = 512
    dropout: float = 0.0
    trainable: bool = True
    log_prob_output: bool = False

class LSTMLM(nn.Module):
    """
    Simple LSTM LM with an embedding, an LSTM, and a final linear
    """
    def __init__(self, step: int, cfg: LSTMLMConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        if cfg.dropout > 0:
            self.dropout = nn.Dropout(p=cfg.dropout)
        else:
            self.dropout = None
        self.use_bottle_neck = cfg.use_bottle_neck
        self.use_log_prob_output = cfg.log_prob_output
        #self.embed = nn.Linear(cfg.vocab_dim, cfg.embed_dim, bias=True)
        self.embed = nn.Embedding(cfg.vocab_dim, cfg.embed_dim)
        self.lstm = nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.n_lstm_layers,
            bias=cfg.bias,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=False,
        )
        if cfg.use_bottle_neck:
            self.bottle_neck = nn.Linear(cfg.hidden_dim,cfg.bottle_neck_dim, bias=True)
            self.final_linear = nn.Linear(cfg.bottle_neck_dim, cfg.vocab_dim, bias=True)
        else:
            self.final_linear = nn.Linear(cfg.hidden_dim, cfg.vocab_dim, bias=True)
        self._param_init(**cfg.init_args)



    def _param_init(self, init_args_w=None, init_args_b=None):
        if init_args_w is None:
            init_args_w = {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}
        if init_args_b is None:
            init_args_b = {'func': 'normal', 'arg': {'mean': 0.0, 'std': 0.1}}

        for m in self.modules():

            for name, param in m.named_parameters():
                if 'bias' in name:
                    if init_args_b['func'] == 'normal':
                        init_func = nn.init.normal_
                    else:
                        NotImplementedError
                    hyp = init_args_b['arg']
                else:
                    if init_args_w['func'] == 'normal':
                        init_func = nn.init.normal_
                    else:
                        NotImplementedError
                    hyp = init_args_w['arg']
                init_func(param, **hyp)
    def _param_freeze(self):
        for params in self.parameters():
            params.requires_grad= False

    def forward(self, x):
        """
        Return log probs of each batch at each time step
        x: (B, S)
        """
        x = self.embed(x)
        if self.dropout:
            x = self.dropout(x)
        batch_size = x.shape[0]
        h0 = torch.zeros((self.cfg.n_lstm_layers, batch_size, self.cfg.hidden_dim), device=x.device).detach()
        c0 = torch.zeros_like(h0, device=x.device).detach()
        x, _ = self.lstm(x, (h0, c0))
        if self.dropout:
            x = self.dropout(x)
        if self.use_bottle_neck:
            x = self.bottle_neck(x)
            if self.dropout:
                x = self.dropout(x)
        x = self.final_linear(x)
        if self.use_log_prob_output:
            x = x.log_softmax(dim=-1)
        return x
    def incremental_step(self, target, h0, c0):
        """
        param:
        param:
        """
        # x: (batch * beam,1)
        # input embedding:
        x = self.embed(target) # (batch* beam,1, embd)
        x, (h1, c1) = self.lstm(x, (h0, c0))
        output = self.final_linear(x)
        out_log_prob = output.log_softmax(dim=-1)
        return out_log_prob, (h1, c1)

    def get_default_init_state(self, batch_size, device):
        h0 = torch.zeros((self.cfg.n_lstm_layers, batch_size, self.cfg.hidden_dim), device=device).detach()
        c0 = torch.zeros_like(h0)
        return h0, c0

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
