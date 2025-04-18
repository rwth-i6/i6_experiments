import torch
from torch import nn
from typing import Tuple
from dataclasses import dataclass

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.config import ModelConfiguration

from ....streamable_module import StreamableModule
from ....common import mask_tensor, num_samples_to_frames
from ....base_config import BaseConfig

from returnn.torch.context import get_run_ctx



@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

@dataclass(kw_only=True)
class StreamableFeatureExtractorV1Config(BaseConfig):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugConfig
    specaug_start_epoch: int

    def module():
        return StreamableFeatureExtractorV1
    
    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["logmel_cfg"] = LogMelFeatureExtractionV1Config(**d["logmel_cfg"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])

        return StreamableFeatureExtractorV1Config(**d)
    

class StreamableFeatureExtractorV1(StreamableModule):
    def __init__(self, cfg: StreamableFeatureExtractorV1Config):
        super().__init__()

        self.logmel = LogMelFeatureExtractionV1(cfg.logmel_cfg)
        self.specaug_config = cfg.specaug_cfg
        self.specaug_start_epoch = cfg.specaug_start_epoch

    def num_samples_to_frames(self, num_samples: int):
        return num_samples_to_frames(
            n_fft=self.logmel.n_fft,
            hop_length=self.logmel.hop_length,
            center=self.logmel.center,
            num_samples=num_samples,
        )

    def prep_streaming_input(self, features: torch.Tensor, mask: torch.Tensor, chunk_sz: int):
        bsz = features.size(0)

        chunk_size_frames = self.num_samples_to_frames(num_samples=int(chunk_sz))
        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -features.size(1) % chunk_size_frames
        # [B, T, *] -> [B, T+time_dim_pad, *] = [B, T', *]
        features = nn.functional.pad(features, (0, 0, 0, time_dim_pad), "constant", 0)
        mask = nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        features = features.view(bsz, -1, chunk_size_frames,
                                         features.size(-1))  # [B, (T'/C), C, F] = [B, N, C, F]
        mask = mask.view(bsz, -1, chunk_size_frames)  # [B, N, C]

        return features, mask

    def forward_offline(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T', <?>] 
        :param raw_audio_len: <= T' (in samples)

        :return: [B, T, F] features
        """
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.logmel(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        out = audio_features_masked_2
        mask = mask_tensor(out, audio_features_len)

        return out, mask
    
    def forward_streaming(self, raw_audio: torch.Tensor, length: torch.Tensor, chunk_sz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        :return: [B, N, C, F], [B, N, C]
        """
        features, mask = self.forward_offline(raw_audio=raw_audio, raw_audio_len=length)
        out, mask = self.prep_streaming_input(features, mask, chunk_sz)

        return out, mask

    def infer(self, input, lengths, chunk_sz_frames):
        audio_features, audio_features_lengths = self.forward_offline(input, lengths)

        time_dim_pad = -audio_features.size(1) % chunk_sz_frames
        audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, time_dim_pad), "constant", 0)

        return audio_features, audio_features_lengths.sum(dim=-1)