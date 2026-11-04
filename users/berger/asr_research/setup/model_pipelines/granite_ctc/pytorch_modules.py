"""
Encoder wrappers around an externally trained HuggingFace CTC model.

The model is not trained in this setup, so there is no RETURNN checkpoint: the weights are loaded from
the HuggingFace repository when the module is constructed, and the recognition and prior jobs run with
``model_checkpoint=None``.

Environment requirements, both of which the RETURNN python of the recognition and prior jobs must meet:

* transformers >= 5.16, since the models declare ``model_type: granite_speech5_ctc``. Older versions
  fail with "model type not recognized" when the model is constructed.
* A working torchaudio matching the installed torch: ``GraniteSpeech5FeatureExtractor`` computes its
  log-mel and delta features with ``torchaudio`` and declares it as a required backend.
"""

__all__ = [
    "GraniteCtcConfig",
    "GraniteCtcModel",
    "GraniteCtcRecogConfig",
    "GraniteCtcRecogModel",
]

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from i6_models.config import ModelConfiguration
from sisyphus import tk


@dataclass
class GraniteCtcConfig(ModelConfiguration):
    huggingface_repo_dir: tk.Path
    blank_index: int = 0
    sample_rate: int = 16000
    dtype: str = "float32"


@dataclass
class GraniteCtcRecogConfig(GraniteCtcConfig):
    blank_penalty: float = 0.0
    prior_scale: float = 0.0
    prior_file: Optional[tk.Path] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.prior_scale == 0 or self.prior_file is not None, "Must specify prior file if prior scale is not 0"


def _load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "rt") as file:
        return json.load(file)


class GraniteCtcModel(torch.nn.Module):
    """
    HuggingFace CTC model behind the encoder interface expected by the RASR search and prior jobs.

    The feature extraction shipped with the model is applied here as well, so the module consumes raw
    audio samples just like the natively trained models of this setup.
    """

    def __init__(self, cfg: GraniteCtcConfig, **_):
        super().__init__()
        from transformers import AutoModelForCTC, AutoProcessor

        repo_dir = cfg.huggingface_repo_dir.get_path()

        processor = AutoProcessor.from_pretrained(repo_dir)
        # The feature extractor is called directly: it takes a list of waveforms and applies the
        # log-mel + delta + frame stacking front-end the model was trained with.
        self.feature_extractor = getattr(processor, "feature_extractor", processor)
        self.model = AutoModelForCTC.from_pretrained(repo_dir, dtype=getattr(torch, cfg.dtype))
        self.model.eval()

        self.blank_index = cfg.blank_index
        self.sample_rate = cfg.sample_rate
        self.hop_length, self.stack_factor, self.subsampling_factor = self._get_framing(repo_dir)

    @staticmethod
    def _get_framing(repo_dir: str) -> Tuple[int, int, int]:
        """
        Framing of the model, used to map audio sample counts onto encoder output frame counts.

        The front-end takes ``samples // hop_length`` mel frames and stacks ``stack_factor`` of them,
        the encoder then halves the time axis once per subsampling block.
        """
        processor_config = _load_json(os.path.join(repo_dir, "processor_config.json"))
        feature_extractor_config = processor_config.get("feature_extractor", processor_config)
        if not feature_extractor_config:
            feature_extractor_config = _load_json(os.path.join(repo_dir, "preprocessor_config.json"))

        hop_length = int(feature_extractor_config.get("hop_length", 160))

        stack_factor = feature_extractor_config.get("stack_factor")
        if stack_factor is None:
            # The stacking factor is implied by how many mel channels end up in one model input frame.
            num_mel_bins = int(feature_extractor_config.get("num_mel_bins", feature_extractor_config.get("n_mels", 80)))
            feature_size = int(feature_extractor_config.get("feature_size", 2 * num_mel_bins))
            num_channels = num_mel_bins * (2 if feature_extractor_config.get("deltas", True) else 1)
            stack_factor = max(feature_size // num_channels, 1)
        stack_factor = int(stack_factor)

        model_config = _load_json(os.path.join(repo_dir, "config.json"))
        encoder_config = model_config.get("encoder_config", model_config)
        subsample_layers = encoder_config.get("subsample_layers", [0, 1])
        subsampling_factor = 2 ** len(subsample_layers)

        return hop_length, stack_factor, subsampling_factor

    def _output_lengths(self, audio_samples_size: torch.Tensor, max_num_frames: int) -> torch.Tensor:
        mel_frames = torch.div(audio_samples_size, self.hop_length, rounding_mode="floor")
        # The front-end pads the trailing partial group instead of dropping it.
        stacked_frames = torch.div(mel_frames + self.stack_factor - 1, self.stack_factor, rounding_mode="floor")
        # Every subsampling block drops a trailing odd frame.
        output_frames = stacked_frames
        factor = self.subsampling_factor
        while factor > 1:
            output_frames = torch.div(output_frames, 2, rounding_mode="floor")
            factor //= 2
        return output_frames.clamp(min=1, max=max_num_frames).to(torch.int32)

    def _extract_features(self, audio_samples: torch.Tensor, audio_samples_size: torch.Tensor) -> dict:
        # The batch is passed as individual clips so that the padding of the incoming batch does not
        # enter the features; the feature extractor re-pads and returns the matching attention mask.
        waveforms: List[torch.Tensor] = [
            audio_samples[b, : int(audio_samples_size[b])].to(dtype=torch.float32, device="cpu")
            for b in range(audio_samples.size(0))
        ]
        inputs = self.feature_extractor(
            waveforms,
            sampling_rate=self.sample_rate,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            key: value.to(device=audio_samples.device) for key, value in inputs.items() if torch.is_tensor(value)
        }

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_samples = audio_samples.squeeze(-1)  # [B, T]

        inputs = self._extract_features(audio_samples, audio_samples_size)
        outputs = self.model(**inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs  # [B, T, V]
        logits = logits.float()

        log_probs = torch.log_softmax(logits, dim=2)  # [B, T, V]
        return log_probs, self._output_lengths(audio_samples_size, logits.size(1))


class GraniteCtcRecogModel(GraniteCtcModel):
    def __init__(self, cfg: GraniteCtcRecogConfig, **_):
        super().__init__(cfg=cfg)

        if cfg.prior_scale != 0:
            assert cfg.prior_file is not None
            self.scaled_priors = cfg.prior_scale * torch.tensor(np.loadtxt(cfg.prior_file), dtype=torch.float32)
        else:
            self.scaled_priors = torch.zeros((self.model.config.vocab_size,), dtype=torch.float32)
        self.blank_penalty = cfg.blank_penalty

    def forward(
        self,
        audio_samples: torch.Tensor,  # [B, T, 1]
        audio_samples_size: torch.Tensor,  # [B]
    ) -> torch.Tensor:
        log_probs, _ = super().forward(audio_samples=audio_samples, audio_samples_size=audio_samples_size)  # [B, T, V]
        scores = -log_probs  # [B, T, V]
        scores[:, :, self.blank_index] += self.blank_penalty  # [B, T, V]
        return scores + self.scaled_priors.to(device=log_probs.device)  # [B, T, V]
