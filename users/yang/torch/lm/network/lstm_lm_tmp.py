from __future__ import annotations


import torch
from torch import nn
from dataclasses import dataclass
from typing import Tuple

from i6_models.config import ModelConfiguration, ModuleFactoryV1




class LSTMNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # needed: hidden dim, input dim, output dim, num_lstm layers
        self.input_embd = nn.Linear(in_features=cfg.input_dim, out_features=cfg.lstm_hidden_dim)
        self.lstm_layers = nn.LSTM(cfg.lstm_input_dim, cfg.lstm_hidden_dim, num_layers=cfg.num_layers,)
        self.output_logit = nn.Linear(in_features=cfg.lstm_hidden_dim, out_features=cfg.output_dim,bias=False,)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        x = self.input_embd(x)
        x = self.lstm_layers(x)
        x = 0
        return x
