from dataclasses import dataclass
from typing import Tuple

import torch
from i6_models.config import ModelConfiguration
from apptek_asr.lib.pytorch.networks.lstm import LstmLmModelV1


class LstmLmScorer(LstmLmModelV1):
    def forward(
        self,
        lstm_out: torch.Tensor,  # [B, H]
    ) -> torch.Tensor:
        # final_linear -> output
        logits = self.output(lstm_out)  # [B, V]
        scores = -torch.nn.functional.log_softmax(logits, dim=-1)  # [B, V] negative log probabilities
        scores = torch.nn.functional.pad(
            scores, [0, 1], value=0
        )  # [B, V+1]  add 0 score for blank in model combination
        return scores


class LstmLmStateInitializer(LstmLmModelV1):
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = torch.zeros([1, 1], dtype=torch.int32)  # [1, 1]
        # self.encoder.embedding = nn.Embedding(self.cfg.input_dim, self.cfg.embed_dim)
        embed = self.encoder.embedding.forward(token)  # [1, 1, E] emebed -> encoder.embedding:
        hidden_size = self.cfg.lstm_encoder_cfg.lstm_layers_cfg.hidden_dim
        lstm_layers = self.cfg.lstm_encoder_cfg.lstm_layers_cfg.num_layers
        h_0 = torch.zeros((lstm_layers, 1, hidden_size), dtype=torch.float32)  # [L, 1, H]
        c_0 = torch.zeros((lstm_layers, 1, hidden_size), dtype=torch.float32)  # [L, 1, H]

        # lstm -> encoder.lstm_block.lstm_stack
        # self.encoder.lstm_block.lstm_stack = nn.LSTM()
        lstm_out, (h_0, c_0) = self.encoder.lstm_block.lstm_stack.forward(
            embed, (h_0, c_0)
        )  # [1, 1, H], [L, 1, H], [L, 1, H]
        lstm_out = lstm_out.reshape([1, hidden_size])

        return lstm_out, h_0, c_0


class LstmLmStateUpdater(LstmLmModelV1):
    def forward(
        self,
        token: torch.Tensor,  # [1]
        lstm_h: torch.Tensor,  # [L, 1, H]
        lstm_c: torch.Tensor,  # [L, 1, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # self.encoder.embedding = nn.Embedding(self.cfg.input_dim, self.cfg.embed_dim)
        embed = self.encoder.embedding.forward(token)  # [1, 1, E] emebed -> encoder.embedding:
        embed = embed.reshape([embed.size(0), 1, embed.size(1)])  # [1, 1, E]

        # lstm -> encoder.lstm_block.lstm_stack
        # self.encoder.lstm_block.lstm_stack = nn.LSTM()
        lstm_out, (new_lstm_h, new_lstm_c) = self.encoder.lstm_block.lstm_stack.forward(
            embed, (lstm_h, lstm_c)
        )  # [1, 1, H], [L, 1, H], [L, 1, H]

        lstm_out = lstm_out.reshape([-1, self.cfg.lstm_encoder_cfg.lstm_layers_cfg.hidden_dim])  # [1, H]

        return lstm_out, new_lstm_h, new_lstm_c
