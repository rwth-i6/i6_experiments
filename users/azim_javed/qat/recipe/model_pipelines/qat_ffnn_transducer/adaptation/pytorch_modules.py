from typing import Tuple

import torch

from ...common.adapters import IdentityInitializedLinear
from ..mem_inited.pytorch_modules import (
    QATFFNNTransducerModel,
    QATFFNNTransducerRecogConfig,
    QATFFNNTransducerScorer,
)
from .config import MemristorAdapterConfig


class MemristorAdapterModel(QATFFNNTransducerModel):
    def __init__(self, cfg: MemristorAdapterConfig, **kwargs):
        locations = [adapter.location for adapter in cfg.adapters]
        if locations != ["output_log_probs"]:
            raise ValueError(f"Expected one output_log_probs adapter, got {locations!r}")
        super().__init__(cfg=cfg.base_config, **kwargs)
        self.requires_grad_(False)
        self.adapters = torch.nn.ModuleDict(
            {"output_log_probs": IdentityInitializedLinear(self.target_size)}
        )
        self.train()

    def train(self, mode: bool = True):
        super().train(False)
        self.adapters.train(mode)
        return self

    def forward(
        self,
        audio_samples: torch.Tensor,
        audio_samples_size: torch.Tensor,
        targets: torch.Tensor,
        targets_size: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            encoder_states, encoder_states_size = self.forward_encoder(
                audio_samples=audio_samples,
                audio_samples_size=audio_samples_size,
            )
            pred_states = self.forward_prediction_network(targets=targets)
            logits = self.forward_joint_network(
                encoder_states=encoder_states,
                encoder_states_size=encoder_states_size,
                pred_states=pred_states,
                targets_size=targets_size,
            )
            log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        adapted_logits = self.adapters["output_log_probs"](log_probs)
        return adapted_logits, encoder_states_size


class MemristorAdapterScorer(QATFFNNTransducerScorer):
    def __init__(self, cfg: QATFFNNTransducerRecogConfig, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.adapters = torch.nn.ModuleDict(
            {"output_log_probs": IdentityInitializedLinear(self.target_size)}
        )

    def _transform_joint_output(self, logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)
        return self.adapters["output_log_probs"](log_probs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        required = {f"adapters.{key}" for key in self.adapters.state_dict()}
        missing = required - state_dict.keys()
        if missing:
            raise RuntimeError(f"Missing adapter weights in recognition checkpoint: {sorted(missing)}")
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
