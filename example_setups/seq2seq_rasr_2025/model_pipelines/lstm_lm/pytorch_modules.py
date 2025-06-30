from dataclasses import dataclass
from typing import Tuple

import torch
from i6_models.config import ModelConfiguration


@dataclass
class LstmLmConfig(ModelConfiguration):
    vocab_size: int
    embed_dim: int
    lstm_hidden_size: int
    lstm_layers: int
    dropout: float


class LstmLm(torch.nn.Module):
    def __init__(self, cfg: LstmLmConfig, **_):
        super().__init__()

        self.embed_dim = cfg.embed_dim
        self.embed = torch.nn.Embedding(num_embeddings=cfg.vocab_size, embedding_dim=cfg.embed_dim)

        self.dropout = torch.nn.Dropout(cfg.dropout)

        self.lstm = torch.nn.LSTM(
            input_size=cfg.embed_dim,
            hidden_size=cfg.lstm_hidden_size,
            num_layers=cfg.lstm_layers,
            bias=True,
            batch_first=True,
            dropout=cfg.dropout,
            bidirectional=False,
        )

        self.final_linear = torch.nn.Linear(cfg.lstm_hidden_size, cfg.vocab_size)

        self._param_init()

    def _param_init(self):
        for m in self.modules():
            for name, param in m.named_parameters():
                torch.nn.init.normal_(param, mean=0, std=0.1)

    def forward(
        self,
        targets: torch.Tensor,  # [B, S]
    ) -> torch.Tensor:  # final log_probs [B, S, V]
        embed = self.embed.forward(targets)  # [B, S, E]
        embed = self.dropout.forward(embed)

        lstm_out, _ = self.lstm.forward(embed)  # [B, S, H]
        lstm_out = self.dropout.forward(lstm_out)

        logits = self.final_linear.forward(lstm_out)  # [B, S, V]
        return logits


class LstmLmScorer(LstmLm):
    def forward(
        self,
        lstm_out: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        logits = self.final_linear(lstm_out)  # [B, V]
        scores = -torch.nn.functional.log_softmax(logits, dim=-1)  # [B, V]
        scores = torch.nn.functional.pad(
            scores, [0, 1], value=0
        )  # [B, V+1]  add 0 score for blank in model combination
        return scores


class LstmLmStateInitializer(LstmLm):
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = torch.zeros([1, 1], dtype=torch.int32)  # [1, 1]
        embed = self.embed.forward(token)  # [1, 1, E]
        h_0 = torch.zeros((self.lstm.num_layers, 1, self.lstm.hidden_size), dtype=torch.float32)  # [L, 1, H]
        c_0 = torch.zeros((self.lstm.num_layers, 1, self.lstm.hidden_size), dtype=torch.float32)  # [L, 1, H]

        lstm_out, (h_0, c_0) = self.lstm.forward(embed, (h_0, c_0))  # [1, 1, H], [L, 1, H], [L, 1, H]
        lstm_out = lstm_out.reshape([1, self.lstm.hidden_size])

        return lstm_out, h_0, c_0


class LstmLmStateUpdater(LstmLm):
    def forward(
        self,
        token: torch.Tensor,  # [1]
        lstm_h: torch.Tensor,  # [L, 1, H]
        lstm_c: torch.Tensor,  # [L, 1, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.embed.forward(token)  # [1, E]
        embed = embed.reshape([embed.size(0), 1, embed.size(1)])  # [1, 1, E]

        lstm_out, (new_lstm_h, new_lstm_c) = self.lstm.forward(embed, (lstm_h, lstm_c))  # [1, 1, H] [L, 1, H] [L, 1, H]

        lstm_out = lstm_out.reshape([-1, self.lstm.hidden_size])  # [1, H]

        return lstm_out, new_lstm_h, new_lstm_c
