__all__ = ["QATConformerCTCConfig", "QATConformerCTCRecogConfig", "QATConformerCTCModel", "QATConformerCTCRecogModel", "QATConformerCTCRecogExportModel"]

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from ..common.assemblies.conformer import ConformerEncoderQuant, ConformerEncoderQuantV1Config

from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from sisyphus import tk

from ..common.pytorch_modules import SpecaugmentByLengthConfig, lengths_to_padding_mask


@dataclass
class QATConformerCTCConfig(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    conformer_cfg: ConformerEncoderQuantV1Config
    dim: int
    target_size: int
    dropout: float
    specaug_cfg: SpecaugmentByLengthConfig

    def __sis_state__(self):
        import dataclasses, torch
        from sisyphus import tk

        def _sanitize(v):
            if isinstance(v, torch.dtype):
                return str(v)
            if isinstance(v, tk.Path):
                return v                 # keep for path extraction
            if dataclasses.is_dataclass(v):
                return {f.name: _sanitize(getattr(v, f.name)) for f in dataclasses.fields(v)}
            if isinstance(v, dict):
                return {k: _sanitize(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return type(v)(_sanitize(x) for x in v)
            return v

        return {f.name: _sanitize(getattr(self, f.name)) for f in dataclasses.fields(self)}

    def __sis_hash__(self):
        return str(type(self))

    def with_replaced(self, **kwargs):
        import dataclasses

        consumed = set()

        def _recurse(obj):
            # 1. Handle lists and tuples
            if isinstance(obj, (list, tuple)):
                new_seq = type(obj)(_recurse(v) for v in obj)
                if any(old is not new for old, new in zip(obj, new_seq)):
                    return new_seq
                return obj

            # 2. Base case: not a dataclass
            if not dataclasses.is_dataclass(obj):
                return obj

            # 3. Handle dataclasses
            changes = {}
            for f in dataclasses.fields(obj):
                val = getattr(obj, f.name)
                if f.name in kwargs:
                    changes[f.name] = kwargs[f.name]
                    consumed.add(f.name)
                else:
                    new_val = _recurse(val)
                    if new_val is not val:
                        changes[f.name] = new_val
            if changes:
                return dataclasses.replace(obj, **changes)
            return obj

        result = _recurse(self)
        unconsumed = set(kwargs) - consumed
        assert not unconsumed, f"with_replaced: keys not found in config tree: {unconsumed}"
        return result


@dataclass
class QATConformerCTCRecogConfig(QATConformerCTCConfig):
    blank_penalty: float
    prior_scale: float
    prior_file: Optional[tk.Path]


def __post_init__(self) -> None:
    assert self.prior_scale == 0 or self.prior_file is not None, "Must specify prior file if prior scale is not 0"


class QATConformerCTCModel(torch.nn.Module):
    def __init__(self, cfg: QATConformerCTCConfig, **_):
        super().__init__()
        self.feature_extraction = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.conformer = ConformerEncoderQuant(cfg.conformer_cfg)
        self.dropout = torch.nn.Dropout(cfg.dropout)
        self.target_size = cfg.target_size
        self.final_linear = torch.nn.Linear(cfg.dim, self.target_size)
        self.specaug_config = cfg.specaug_cfg

    def __sis_state__(self):
        return str(type(self))
    
    def __sis_hash__(self):
        return str(type(self))

    def prep_quant(self):
        self.conformer.prep_quant()

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        with torch.no_grad():
            audio_samples = audio_samples.squeeze(-1)  # [B, T]
            features, features_size = self.feature_extraction(audio_samples, audio_samples_size)  # [B, T, F], [B]
            sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

            if self.training:
                from returnn.torch.context import get_run_ctx  # type: ignore

                if get_run_ctx().epoch >= self.specaug_config.start_epoch:
                    features = specaugment_v1_by_length(
                        audio_features=features,
                        time_min_num_masks=self.specaug_config.time_min_num_masks,
                        time_max_mask_per_n_frames=self.specaug_config.time_max_mask_per_n_frames,
                        time_mask_max_size=self.specaug_config.time_mask_max_size,
                        freq_min_num_masks=self.specaug_config.freq_min_num_masks,
                        freq_max_num_masks=self.specaug_config.freq_max_num_masks,
                        freq_mask_max_size=self.specaug_config.freq_mask_max_size,
                    )  # [B, T, F]

        encoder_states, sequence_mask = self.conformer(features, sequence_mask)  # [B, T, F], [B, T]
        encoder_states = encoder_states[-1]

        encoder_states = self.dropout(encoder_states)  # [B, T, F]
        logits = self.final_linear(encoder_states)  # [B, T, V]
        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, V]

        return log_probs, torch.sum(sequence_mask, dim=1).type(torch.int32)


class QATConformerCTCRecogModel(QATConformerCTCModel):
    def __init__(self, cfg: QATConformerCTCRecogConfig, epoch: int, **_):
        super().__init__(
            cfg=cfg,
            epoch=epoch,
        )
        if cfg.prior_scale != 0:
            assert cfg.prior_file is not None
            self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        else:
            self.scaled_priors = torch.zeros((cfg.target_size,), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        log_probs, _ = super().forward(audio_samples=audio_samples, audio_samples_size=audio_samples_size)  # [B, T, V]
        scores = -log_probs  # [B, T, V]
        scores[:, :, -1] += self.blank_penalty  # [B, T, V]
        return scores + self.scaled_priors.to(device=log_probs.device)  # [B, T, V]


class QATConformerCTCRecogExportModel(QATConformerCTCModel):
    def __init__(self, cfg: QATConformerCTCRecogConfig, epoch: int, **_):
        super().__init__(
            cfg=cfg,
            epoch=epoch,
        )
        if cfg.prior_scale != 0:
            assert cfg.prior_file is not None
            self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        else:
            self.scaled_priors = torch.zeros((cfg.target_size,), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        features: torch.Tensor,  # [B, T, F]
        features_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_mask = lengths_to_padding_mask(features_size)  # [B, T]

        encoder_states, sequence_mask = self.conformer(features, sequence_mask)  # [B, T, F], [B, T]
        encoder_states = encoder_states[-1]

        encoder_states = self.dropout(encoder_states)  # [B, T, F]
        logits = self.final_linear(encoder_states)  # [B, T, V]
        scores = -torch.log_softmax(logits, dim=2)  # [B, T, V]

        scores[:, :, -1] += self.blank_penalty  # [B, T, V]
        return scores + self.scaled_priors.to(device=scores.device), torch.sum(sequence_mask, dim=1).type(
            torch.int32
        )  # [B, T, V], [B]
