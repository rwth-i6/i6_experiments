import torch
from torch import nn
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

from i6_models.util import compat
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config, filters
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import SpecaugConfig
from ..conformer_1124.conf_relpos_streaming_v1 import ConformerRelPosBlockV1COV1Config
from ..conformer_0325.conf_dual_0325_v1 import (
    StreamableModule,
    StreamableConformerEncoderRelPosV2Config,
    StreamableConformerBlockRelPosV1,
    BroadcastDropout
)
from ..auxil.functional import add_lookahead_v2, create_chunk_mask, Mode, mask_tensor
from returnn.torch.context import get_run_ctx



class StreamableFeatureExtractorWithAmplitudePerturbation(StreamableModule):
    def __init__(
            self, cfg: LogMelFeatureExtractionV1Config, 
            specaug_cfg: SpecaugConfig, specaug_start_epoch: int,
    ):
        super().__init__()

        self.logmel = LogMelFeatureExtractionV1(cfg)
        self.specaug_config = specaug_cfg
        self.specaug_start_epoch = specaug_start_epoch

    def num_samples_to_frames(self, num_samples: int):
        if self.logmel.center:
            return (num_samples // self.logmel.hop_length) + 1
        else:
            return ((num_samples - self.logmel.n_fft) // self.logmel.hop_length) + 1

    def prep_streaming_input(self, features: torch.Tensor, mask: torch.Tensor, chunk_sz: int):
        bsz = features.size(0)

        chunk_size_frames = self.num_samples_to_frames(num_samples=int(chunk_sz))
        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -features.size(1) % chunk_size_frames
        # [B, T, *] -> [B, T+time_dim_pad, *] = [B, T', *]
        features = nn.functional.pad(features, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
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
        # amplitude perturbation
        amp_mult = torch.FloatTensor(1).uniform_(1/8, 8).item()
        raw_audio_amp_perturbed = raw_audio * amp_mult

        squeezed_features = torch.squeeze(raw_audio_amp_perturbed, dim=-1)
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