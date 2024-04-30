import torch
from torch import nn
from typing import Callable, List, Optional
from enum import Enum

# frontend imports
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from .generic_frontend_v2 import GenericFrontendV2

# feature extract and conformer module imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1
from i6_models.parts.conformer.norm import LayerNormNC
from ..conformer_2.conformer_v4 import (
    Mode,
    ConformerConvolutionV1Config,
    ConformerBlockV2Config,
    ConformerEncoderV2Config,
    ConformerEncoderV4
)
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    prior_step,
    prior_init_hook,
    prior_finish_hook,
    mask_tensor,
    
    Predictor,
    Joiner
)

from i6_models.primitives.specaugment import specaugment_v1_by_length

from .unified_lookahead_transducer_cfg import ModelConfigV2 as ModelConfig

from returnn.torch.context import get_run_ctx


from torch.utils.checkpoint import checkpoint



class Model(nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        # net_args are passed as a dict to returnn and here the config is retransformed into its dataclass
        self.cfg = ModelConfig.from_dict(model_config_dict)

        if self.cfg.use_vgg:
            frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=self.cfg.frontend_config)
        else:
            frontend = ModuleFactoryV1(module_class=GenericFrontendV2, cfg=self.cfg.frontend_config)

        conformer_config = ConformerEncoderV2Config(
            num_layers=self.cfg.num_layers,
            frontend=frontend,
            block_cfg=ConformerBlockV2Config(
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
                causal_conv_cfg=ConformerConvolutionV1Config(
                    channels=self.cfg.conformer_size,
                    kernel_size=self.cfg.causal_conv_kernel_size,
                    dropout=self.cfg.causal_conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(self.cfg.conformer_size),
                ),
                causal_scale=self.cfg.causal_scale 
            ),
        )

        self.feature_extraction = LogMelFeatureExtractionV1(cfg=self.cfg.feature_extraction_config)
        self.conformer = ConformerEncoderV4(cfg=conformer_config)
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

        self.chunk_size = int(self.cfg.chunk_size)
        self.lookahead_size = int(self.cfg.lookahead_size)

        self.num_split_blocks = self.cfg.num_split_blocks

        self.mode: Optional[Mode] = None


    @staticmethod
    def prep_streaming_input(raw_audio, raw_audio_len, chunk_size):
        raw_audio = torch.squeeze(raw_audio, dim=-1)
        batch_size = raw_audio.size(0)

        # pad audio time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -raw_audio.size(1) % chunk_size
        # (B, T) -> (B, T+time_dim_pad) = (B, T')
        raw_audio = torch.nn.functional.pad(raw_audio, (0, time_dim_pad), "constant", 0)

        # separate chunks to signal the conformer that we are chunking input
        raw_audio_chunked = raw_audio.view(batch_size, -1, chunk_size)  # (B, (T'/C), C) = (B, N, C)

        #
        # figure out raw_audio_len of chunks
        #
        num_chunks = raw_audio_chunked.size(1)  # = N

        sub = chunk_size * torch.arange(0, num_chunks).unsqueeze(0)  # (1, N)
        raw_audio_len_chunked = raw_audio_len.unsqueeze(-1)  # (B, 1)

        raw_audio_len_chunked = raw_audio_len_chunked - sub  # (B, N)
        raw_audio_len_chunked[raw_audio_len_chunked < 0] = 0
        raw_audio_len_chunked[raw_audio_len_chunked > chunk_size] = chunk_size

        # (B, N, C), (B, N)
        return raw_audio_chunked, raw_audio_len_chunked

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

    def streaming_feature_extractor(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, chunk_size: int
    ):

        batch_size = raw_audio.size(0)

        # chunk raw_audio
        raw_audio_chunked, raw_audio_len_chunked = self.prep_streaming_input(raw_audio, raw_audio_len, chunk_size)
        num_chunks = raw_audio_chunked.size(1)

        # future acoustic context
        future_ac_ctx = torch.zeros(batch_size, num_chunks, self.lookahead_size, device=raw_audio.device)
        future_ac_ctx[:, :-1] = raw_audio_chunked[:, 1:, :self.lookahead_size]

        # add future acoustic context to audio chunks and update lengths
        raw_audio_chunked_fac = torch.cat((raw_audio_chunked, future_ac_ctx), dim=2)  # (B, N, C+R)
        
        added_lengths = torch.zeros_like(raw_audio_len_chunked)
        added_lengths[raw_audio_len_chunked > self.lookahead_size] = self.lookahead_size
        raw_audio_len_chunked_fac = raw_audio_len_chunked
        raw_audio_len_chunked_fac[:, :-1] += added_lengths[:, :-1]

        raw_audio_chunked_fac = raw_audio_chunked_fac.view(-1, chunk_size+self.lookahead_size)
        raw_audio_len_chunked_fac = raw_audio_len_chunked_fac.view(-1, 1)
        # >>> (B*N, C+R), (B*N, 1)


        ### TODO: Test above [x]


        # extract audio features for each audio chunk
        conformer_in, mask = self.extract_features(raw_audio_chunked_fac, raw_audio_len_chunked_fac)
        # >>> (B*N, (C+R)', F), (B*N, (C+R)')

        lah_size_frames = self.lookahead_size // self.feature_extraction.hop_length

        # reshape back to chunks
        conformer_in = conformer_in.view(
            batch_size, num_chunks, -1, conformer_in.size(-1)) # (B, N, (C+R)', F)
        mask = mask.view(batch_size, num_chunks, -1)  # (B, N, (C+R)')

        return conformer_in, mask, lah_size_frames


    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
        labels: torch.Tensor, labels_len: torch.Tensor
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        #torch.cuda.memory._record_memory_history()
        
        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        # chunk audio samples and extract features
        with torch.no_grad():
            conformer_in, mask, lah_size_frames = self.streaming_feature_extractor(
                raw_audio=raw_audio, raw_audio_len=raw_audio_len, chunk_size=self.chunk_size
            )

        conformer_outs = self.conformer(conformer_in, mask, lah_size_frames, k=self.num_split_blocks)

        rnnt_outs = []
        for conformer_out, out_mask, mode in conformer_outs:

            conformer_joiner_out = self.encoder_out_linear(conformer_out)
            conformer_out_lengths = torch.sum(out_mask, dim=1)  # [B, T] -> [B]

            output_logits, src_len, tgt_len = checkpoint(
                self.custom(self.joiner),
                conformer_joiner_out,
                conformer_out_lengths,
                predict_out,
                labels_len
            )

            # output_logits, src_len, tgt_len = self.joiner(
            #     source_encodings=conformer_joiner_out,
            #     source_lengths=conformer_out_lengths,
            #     target_encodings=predict_out,
            #     target_lengths=labels_len,
            # )  # output is [B, T, N, #vocab]

            if self.cfg.ctc_output_loss > 0:
                ctc_logprobs = torch.log_softmax(self.encoder_ctc(conformer_out), dim=-1)
            else:
                ctc_logprobs = None

            rnnt_outs.append((output_logits, src_len, ctc_logprobs, mode))
        
        return rnnt_outs

    def custom(self, module):
        def custom_forward(*inputs):
            output_logits, src_len, tgt_len = module(
                source_encodings=inputs[0],
                source_lengths=inputs[1],
                target_encodings=inputs[2],
                target_lengths=inputs[3],
            )

            return output_logits, src_len, tgt_len

        return custom_forward
        



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

    model_outs = model(
        raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        labels=prepended_targets, labels_len=prepended_target_lengths
    )

    num_phonemes = torch.sum(labels_len)

    for logits, audio_features_len, ctc_logprobs, encoder_mode in model_outs:
        scale = model.cfg.online_model_scale if encoder_mode == Mode.ONLINE else (1 - model.cfg.online_model_scale)
        mode_str = encoder_mode.name.lower()[:3]

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
        run_ctx.mark_as_loss(name="rnnt.%s" % mode_str, loss=rnnt_loss, inv_norm_factor=num_phonemes,
                            scale=scale)

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

    #torch.cuda.memory._dump_snapshot("my_snapshot.pickle")
