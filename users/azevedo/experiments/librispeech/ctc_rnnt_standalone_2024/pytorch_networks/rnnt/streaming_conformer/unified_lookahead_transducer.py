import torch
from torch import nn
from typing import Callable, List, Optional
from functools import cache
from enum import Enum

from .streaming_conformer_v2 import Model as StreamingConformerLAH

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor
from i6_models.primitives.specaugment import specaugment_v1_by_length

from .unified_lookahead_transducer_cfg import ModelConfig

from returnn.torch.context import get_run_ctx



class Mode(Enum):
    STREAMING = 0
    OFFLINE = 1


class Model(StreamingConformerLAH):
    def __init__(self, model_config_dict, **kwargs):

        # FIXME: do this cleaner; remove all entries that are not in parameters or fields of inherited dataclass
        super_model_config_dict = model_config_dict.copy()
        del super_model_config_dict["online_model_scale"]

        super().__init__(super_model_config_dict, **kwargs)

        self.cfg = ModelConfig.from_dict(model_config_dict)

        self.cache_funcs: List[Callable] = [self.extract_features, self.prep_streaming_input]

        self.mode: Optional[Mode] = None

    #@cache
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

    def get_chunk_size(self, **kwargs) -> int:
        chunk_size = super().get_chunk_size(**kwargs)
        return self.num_samples_to_frames(num_samples=chunk_size)

    #@cache
    def prep_streaming_input(self, conformer_in, mask):
        batch_size = conformer_in.size(0)

        chunk_size_frames = self.get_chunk_size()

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

    def clear_caches(self):
        for cache_func in self.cache_funcs:
            cache_func.cache_clear()

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

        conformer_out, out_mask = self.conformer(conformer_in, mask, lookahead_size=self.lookahead_size)

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

    for encoder_mode in [Mode.STREAMING, Mode.OFFLINE]:
        model.mode = encoder_mode
        logits, audio_features_len, ctc_logprobs = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
            labels=prepended_targets, labels_len=prepended_target_lengths
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

        scale = model.cfg.online_model_scale if encoder_mode == Mode.STREAMING else (1 - model.cfg.online_model_scale)
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

    #model.clear_caches()