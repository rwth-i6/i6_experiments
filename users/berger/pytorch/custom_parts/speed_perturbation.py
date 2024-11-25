from dataclasses import dataclass

import numpy as np
import torch
from i6_models.config import ModelConfiguration
from i6_models.parts.frontend.vgg_act import Tuple


@dataclass
class SpeedPerturbationModuleV1Config(ModelConfiguration):
    min_speed_factor: float
    max_speed_factor: float


class SpeedPerturbationModuleV1(torch.nn.Module):
    def __init__(self, model_cfg: SpeedPerturbationModuleV1Config):
        super().__init__()

        self.min_speed_factor = model_cfg.min_speed_factor
        self.max_speed_factor = model_cfg.max_speed_factor

    def forward(self, audio_samples: torch.Tensor, sequence_lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        samples_device = audio_samples.device
        lengths_device = sequence_lengths.device

        from scipy.signal import resample

        speed_factor = np.random.uniform(self.min_speed_factor, self.max_speed_factor)

        B, T = audio_samples.shape
        new_T = int(T * speed_factor)

        perturbed_audio_samples_list = []
        new_sequence_lengths = []

        for b in range(B):
            sequence_length_b = sequence_lengths[b].cpu().item()
            audio_samples_b = audio_samples[b, :sequence_length_b].cpu().numpy()

            new_sequence_length_b = int(sequence_length_b * speed_factor)
            new_sequence_lengths.append(new_sequence_length_b)

            perturbed_audio = resample(audio_samples_b, num=new_sequence_length_b)
            perturbed_audio_tensor = torch.tensor(perturbed_audio, dtype=audio_samples.dtype, device=samples_device)

            if new_sequence_length_b < new_T:
                perturbed_audio_tensor = torch.nn.functional.pad(
                    perturbed_audio_tensor, pad=(0, new_T - new_sequence_length_b)
                )

            perturbed_audio_samples_list.append(perturbed_audio_tensor)

        perturbed_audio_samples_tensor = torch.stack(perturbed_audio_samples_list)
        new_sequence_lengths_tensor = torch.tensor(
            new_sequence_lengths, dtype=sequence_lengths.dtype, device=lengths_device
        )

        return perturbed_audio_samples_tensor, new_sequence_lengths_tensor
