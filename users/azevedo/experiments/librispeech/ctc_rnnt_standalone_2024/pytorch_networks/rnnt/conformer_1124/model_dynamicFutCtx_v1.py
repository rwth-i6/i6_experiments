import torch
from torch import nn
from typing import Callable, List, Optional
from enum import Enum

from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx

from .model_dynamicFutCtx_v1_cfg import ModelConfig

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config

from .conformer_dynamicFutCtx_v1 import (
    ConformerLearnedFutCtxEncoderV1,
    ConformerLearnedFutCtxEncoderV1Config,
)

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from ..streaming_conformer.generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from .conformer_dynamicFutCtx_v1 import AggregatorV1Config

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
    mask_tensor,
    
    Predictor,
    Joiner
)

from ..auxil.functional import num_samples_to_frames


class Mode(Enum):
    STREAMING = 0
    OFFLINE = 1


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if self.cfg.use_vgg:
            frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config)
        else:
            frontend = ModuleFactoryV1(module_class=GenericFrontendV2, cfg=self.cfg.frontend_config)

        conformer_config = ConformerLearnedFutCtxEncoderV1Config(
            num_layers=self.cfg.num_layers,
            frontend=frontend,
            aggregator_cfg=AggregatorV1Config(
                input_dim=self.cfg.conformer_size,
                num_lstm_layers=self.cfg.aggr_num_layers,
                lstm_hidden_dim=self.cfg.aggr_lstm_dim,
                lstm_dropout=self.cfg.aggr_lstm_dropout,
                ff_dim=self.cfg.aggr_ff_dim,
                fut_weights_dropout=self.cfg.aggr_weights_dropout,
                num_att_heads=self.cfg.aggr_attn_heads,
                with_bias=self.cfg.aggr_bias,
                noise_std=self.cfg.aggr_noise_std,
            ),
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
        self.conformer = ConformerLearnedFutCtxEncoderV1(cfg=conformer_config)
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

        self.mode: Optional[Mode] = None

        self.carry_over_size = self.cfg.carry_over_size

    def extract_features(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
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

        conformer_in = audio_features_masked_2
        mask = mask_tensor(conformer_in, audio_features_len)

        return conformer_in, mask

    def prep_streaming_input(self, conformer_in, mask):
        batch_size = conformer_in.size(0)

        chunk_size_frames = num_samples_to_frames(
            n_fft=self.feature_extraction.n_fft, 
            hop_length=self.feature_extraction.hop_length,
            center=self.feature_extraction.center,
            num_samples=int(self.cfg.chunk_size)
        )

        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        conformer_in = conformer_in.view(batch_size, -1, chunk_size_frames,
                                         conformer_in.size(-1))  # (B, (T'/C), C, F) = (B, N, C, F)
        mask = mask.view(batch_size, -1, chunk_size_frames)  # (B, N, C)

        return conformer_in, mask

    def merge_chunks(self, conformer_out, out_mask):
        batch_size = conformer_out.size(0)

        conformer_out = conformer_out.view(batch_size, -1, conformer_out.size(-1))  # (B, C'*N, F')
        out_mask = out_mask.view(batch_size, -1)  # (B, C'*N)

        return conformer_out, out_mask

    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor, 
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        assert self.mode is not None
        #print(f"> Currently running in {self.mode}.")

        conformer_in, mask = self.extract_features(raw_audio, raw_audio_len)

        if self.mode == Mode.STREAMING:
            conformer_in, mask = self.prep_streaming_input(conformer_in, mask)

            #assert conformer_in.size(2) == (self.cfg.chunk_size / 0.01)

        conformer_out, out_mask, fut_frame_probs = self.conformer(conformer_in, mask, 
                                                                  carry_over_size=self.carry_over_size)

        if self.mode == Mode.STREAMING:
            conformer_out, out_mask = self.merge_chunks(conformer_out, out_mask)

        conformer_joiner_out = self.encoder_out_linear(conformer_out)
        conformer_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_joiner_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc(conformer_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs, fut_frame_probs


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

    #
    #
    #
    encoder_mode = Mode.STREAMING  # if run_ctx.global_step % 2 == 0 else Mode.OFFLINE

    model.mode = encoder_mode
    logits, audio_features_len, ctc_logprobs, fut_frame_probs = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        labels=prepended_targets, labels_len=prepended_target_lengths,
    )

    logprobs = torch.log_softmax(logits, dim=-1)
    fastemit_lambda = model.cfg.fastemit_lambda

    rnnt_loss = warp_rnnt.rnnt_loss(
        log_probs=logprobs,
        frames_lengths=audio_features_len.to(dtype=torch.int32),
        labels=labels,
        labels_lengths=labels_len.to(dtype=torch.int32),
        blank=model.cfg.label_target_size,
        fastemit_lambda=fastemit_lambda if fastemit_lambda is not None else 0.0,
        reduction="sum",
        gather=True,
    )

    num_phonemes = torch.sum(labels_len)

    scale = 1  # model.cfg.online_model_scale if encoder_mode == Mode.STREAMING else (1 - model.cfg.online_model_scale)
    mode_str = encoder_mode.name.lower()[:3]

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
        run_ctx.mark_as_loss(name="ctc.%s" % mode_str, loss=ctc_loss, inv_norm_factor=num_phonemes,
                                scale=model.cfg.ctc_output_loss * scale)

    run_ctx.mark_as_loss(name="rnnt.%s" % mode_str, loss=rnnt_loss, inv_norm_factor=num_phonemes,
                            scale=scale)
    
    bsz = fut_frame_probs.size(0)
    num_subframes = fut_frame_probs.size(-1)
    num_chunks = fut_frame_probs.size(-2)
    chunk_size = num_subframes // num_chunks

    # frame_indc = torch.arange(1, num_subframes+1, device=fut_frame_probs.device).view(1, num_subframes)
    # chunk_ends = torch.arange(1, num_chunks+1, device=fut_frame_probs.device).view(num_chunks, 1) * chunk_size
    # realisations = (frame_indc - chunk_ends).unsqueeze(0).expand(bsz, -1, -1) # [1, K, T]
    # realisations[realisations < 0] = 0

    # # remove last chunks (have p=1 for last frame)
    # probs_indcs = (fut_frame_probs < 1).all(-1)
    # probs = fut_frame_probs[probs_indcs]
    # realisations = realisations[probs_indcs]
    # # remove empty chunks
    # nonempty_indcs = (probs > 0).any(-1)
    # probs = probs[nonempty_indcs]
    # realisations = realisations[nonempty_indcs]

    # fut_frame_reals = fut_frame_probs * realisations

    chunk_total_lengths = torch.arange(0, num_chunks, device=audio_features_len.device) * chunk_size
    # [B, 1] - [1, K] = [B, K]
    chunk_rem_lengths = audio_features_len.unsqueeze(-1) - chunk_total_lengths.unsqueeze(0)
    chunk_rem_lengths[chunk_rem_lengths < 0] = 0

    latency_targets = chunk_total_lengths + chunk_size + model.cfg.avg_fut_latency
    latency_targets = latency_targets.unsqueeze(0).repeat(bsz, 1)  # [B, K]
    latency_targets = torch.where(
        latency_targets > chunk_rem_lengths,
        audio_features_len.unsqueeze(-1),
        latency_targets
    )

    print(fut_frame_probs[0, 0])

    expected_aggr_lat = (fut_frame_probs.sum(dim=-1) - latency_targets)**2  # [B, K]
    expected_aggr_lat = torch.mean(expected_aggr_lat)  # [1]
    
    run_ctx.mark_as_loss(name="ltc.agr", loss=expected_aggr_lat)