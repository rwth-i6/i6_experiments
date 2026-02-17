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
        logits = self.output(lstm_out)  # [B, V]
        scores = -torch.nn.functional.log_softmax(logits, dim=-1)  # [B, V]
        scores = torch.nn.functional.pad(
            scores, [0, 1], value=0
        )  # [B, V+1]  add 0 score for blank in model combination
        return scores


class LstmLmStateInitializer(LstmLmModelV1):
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        token = torch.zeros([1, 1], dtype=torch.int32)  # [1, 1]
        embed = self.encoder.embedding.forward(token)  # [1, 1, E]

        hidden_size = self.cfg.lstm_encoder_cfg.lstm_layers_cfg.hidden_dim
        lstm_layers = self.cfg.lstm_encoder_cfg.lstm_layers_cfg.num_layers
        h_0 = torch.zeros((lstm_layers, 1, hidden_size), dtype=torch.float32)  # [L, 1, H]
        c_0 = torch.zeros((lstm_layers, 1, hidden_size), dtype=torch.float32)  # [L, 1, H]

        lstm_out, (h_0, c_0) = self.encoder.lstm_block.lstm_stack.forward(
            embed, (h_0, c_0)
        )  # [1, 1, H], [L, 1, H], [L, 1, H]
        lstm_out = lstm_out.reshape([1, hidden_size])

        h_0 = h_0.transpose(0, 1)  # [1, L, H]
        c_0 = c_0.transpose(0, 1)  # [1, L, H]

        return lstm_out, h_0, c_0


class LstmLmStateUpdater(LstmLmModelV1):
    def forward(
        self,
        token: torch.Tensor,  # [B]
        lstm_h: torch.Tensor,  # [B, L, H]
        lstm_c: torch.Tensor,  # [B, L, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.encoder.embedding.forward(token)  # [1, 1, E]
        embed = embed.reshape([embed.size(0), 1, embed.size(1)])  # [B, 1, E]

        lstm_h = lstm_h.transpose(0, 1)  # [L, B, H]
        lstm_c = lstm_c.transpose(0, 1)  # [L, B, H]

        lstm_out, (new_lstm_h, new_lstm_c) = self.encoder.lstm_block.lstm_stack.forward(
            embed, (lstm_h, lstm_c)
        )  # [B, 1, H] [L, B, H] [L, B, H]

        hidden_size = self.cfg.lstm_encoder_cfg.lstm_layers_cfg.hidden_dim
        lstm_out = lstm_out.reshape([-1, hidden_size])  # [B, H]

        new_lstm_h = new_lstm_h.transpose(0, 1)  # [B, L, H]
        new_lstm_c = new_lstm_c.transpose(0, 1)  # [B, L, H]

        return lstm_out, new_lstm_h, new_lstm_c
