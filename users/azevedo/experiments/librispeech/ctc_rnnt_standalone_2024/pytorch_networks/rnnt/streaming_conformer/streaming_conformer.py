import torch
from torch import nn
from typing import Callable, Optional, Type, Union, Tuple


# config imports
from .streaming_conformer_cfg import ModelConfig
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerBlockV1Config

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from .generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
# stuff that didn't change
from ..conformer_2.conformer_v2 import (
    ConformerEncoderV2
)
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
    mask_tensor,
    
    Predictor,
    Joiner
)

from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx


class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if self.cfg.use_vgg:
            frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config)
        else:
            frontend = ModuleFactoryV1(module_class=GenericFrontendV2, cfg=self.cfg.frontend_config)

        conformer_config = ConformerEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=frontend,
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=self.cfg.conformer_size,
                    hidden_dim=self.cfg.ff_dim,
                    dropout=self.cfg.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=self.cfg.conformer_size,
                    num_att_heads=self.cfg.num_heads,
                    att_weights_dropout=self.cfg.att_weights_dropout,
                    dropout=self.cfg.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=self.cfg.conformer_size,
                    kernel_size=self.cfg.conv_kernel_size,
                    dropout=self.cfg.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                ),
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV2(cfg=conformer_config)
        self.predictor = Predictor(
            cfg=self.cfg.predictor_config,
            label_target_size=self.cfg.label_target_size + 1,  # ctc blank added
            output_dim=self.cfg.joiner_dim,
        )
        self.joiner = Joiner(
            input_dim=self.cfg.joiner_dim,
            output_dim=self.cfg.label_target_size + 1,
            activation=self.cfg.joiner_activation,
            dropout=self.cfg.joiner_dropout,
        )
        self.encoder_out_linear = nn.Linear(self.cfg.conformer_size, self.cfg.joiner_dim)
        if self.cfg.ctc_output_loss > 0:
            self.encoder_ctc = nn.Linear(self.cfg.conformer_size, self.cfg.label_target_size + 1)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch

    def num_samples_to_frames(self, num_samples: int) -> int:
        n_fft = self.feature_extraction.n_fft
        hop_length = self.feature_extraction.hop_length

        if self.feature_extraction.center:
            return (num_samples // hop_length) + 1
        else:
            return ((num_samples - n_fft) // hop_length) + 1

    def get_chunk_size(self, **kwargs) -> int:
        # TODO: add possibility for dynamic chunk_size

        # return self.num_samples_to_frames(num_samples=int(self.cfg.chunk_size))
        return int(self.cfg.chunk_size)

    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, labels: torch.Tensor, labels_len: torch.Tensor
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, S]
        :param labels_len: length of S as [B]
        
        :return: logprobs [B, T + S, #labels + blank]
        """
        batch_size = raw_audio.size(0)

        #
        # compute (log-mel) features of raw_audio
        #
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.cfg.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.cfg.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.cfg.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.cfg.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        #
        # (transcriber compuation): prep conformer input and pass to conformer
        #
        conformer_in = audio_features_masked_2  # (B, T', F)
        mask = mask_tensor(conformer_in, audio_features_len)  # (B, T')

        chunk_size = self.get_chunk_size()
        # convert chunk size in #samples to chunk size in #frames based on feature_extraction pipeline
        chunk_size_frames = self.num_samples_to_frames(num_samples=chunk_size)

        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        conformer_in = conformer_in.view(batch_size, -1, chunk_size_frames, conformer_in.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
        mask = mask.view(batch_size, -1, chunk_size_frames)  # (B, N, C)

        print(f"\n{batch_size = }, {chunk_size_frames = }, {conformer_in.shape = }")
        conformer_out, out_mask = self.conformer(conformer_in, mask)  # (B, N, C', F')
        print(f"{conformer_out.shape = }")

        # merge chunks
        conformer_out = conformer_out.view(batch_size, -1, conformer_out.size(-1))  # (B, C'*N, F')
        out_mask = out_mask.view(batch_size, -1)  # (B, C'*N)

        conformer_joiner_out = self.encoder_out_linear(conformer_out)   # (B, C'*N, F')
        conformer_out_lengths = torch.sum(out_mask, dim=1)  # (B, C'*N) -> (B)

        #
        # predictor and joiner computation 
        #
        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_joiner_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, C'*N, S, #vocab + 1]

        #
        # (joint-transducer) compute ctc output on shared conformer output
        #
        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc(conformer_out), dim=-1)          
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs


def train_step(*, model: Model, data, run_ctx, **kwargs):
    # import for training only, will fail on CPU servers
    from i6_native_ops import warp_rnnt

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B], cpu transfer needed only for Mini-RETURNN

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    prepended_targets = labels.new_empty([labels.size(0), labels.size(1) + 1])
    prepended_targets[:, 1:] = labels
    prepended_targets[:, 0] = model.cfg.label_target_size  # blank is last index
    prepended_target_lengths = labels_len + 1

    logits, audio_features_len, ctc_logprobs = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len, labels=prepended_targets, labels_len=prepended_target_lengths
    )

    logprobs = torch.log_softmax(logits, dim=-1)
    fastemit_lambda = model.cfg.fastemit_lambda

    rnnt_loss = warp_rnnt.rnnt_loss(
        log_probs=logprobs,
        frames_lengths=audio_features_len.to(dtype=torch.int32),
        labels=labels,
        labels_lengths=labels_len.to(dtype=torch.int32),
        blank=model.cfg.label_target_size,
        fastemit_lambda=0.0,#fastemit_lambda if fastemit_lambda is not None else 0.0,
        reduction="sum",
        gather=True,
    )

    num_phonemes = torch.sum(labels_len)

    # currently not being used
    if ctc_logprobs is not None:
        transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes, scale=model.cfg.ctc_output_loss)

    run_ctx.mark_as_loss(name="rnnt", loss=rnnt_loss, inv_norm_factor=num_phonemes)
