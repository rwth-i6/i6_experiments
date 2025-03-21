import torch
from torch import nn
from typing import Optional
import numpy as np

from i6_experiments.users.azevedo.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.rnnt.conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import (
    Model as BaseModel,
    prior_step,
    prior_init_hook,
    prior_finish_hook
)

from i6_models.primitives.specaugment import specaugment_v1_by_length

from returnn.torch.context import get_run_ctx


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


def sample_chunk_size(ranges_weights: torch.Tensor, bin_chunk_ranges: torch.Tensor, choose_seq_len_p: float = 0.0) -> int:
    """_summary_

    Args:
        bin_widths (torch.Tensor): [S]
        bin_chunk_ranges (torch.Tensor): [S, 2]
    """
    assert ranges_weights.size(0) == bin_chunk_ranges.size(0)

    # force to use whole seq_len as chunk_size
    if np.random.rand() <= choose_seq_len_p:
        return int(bin_chunk_ranges[-1, 1])
    
    # sample index based on histogram `ranges_weights`, i.e. P(index = i) = ranges_weights[i] / (ranges_weights.sum())
    range_sample = torch.multinomial(ranges_weights, 1).item()
    # (categorical) uniform sample within the range, i.e. chunk_size_sample ~ U[range_start, range_end]
    chunk_size_sample = torch.randint(low=int(bin_chunk_ranges[range_sample, 0]), 
                                      high=int(bin_chunk_ranges[range_sample, 1]), 
                                      size=(1,)).item()

    return int(chunk_size_sample)


class Model(BaseModel):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__(model_config_dict, **kwargs)

        self.chunk_size: int = kwargs.get("chunk_size", 0)

        # greatest divisor chunking
        self.div_chunk_size: bool = kwargs.get("div_chunk_size", False)

        # dynamic chunking
        self.min_chunk_size: Optional[float] = kwargs.get("min_chunk_size", 0.15*16000)
        self.max_chunk_size: Optional[float] = kwargs.get("max_chunk_size", float("inf"))
        self.chunk_ranges_weights: Optional[list] = kwargs.get("chunk_ranges_weights", None)
        self.choose_seq_len_p: Optional[float] = kwargs.get("choose_seq_len_p", 0.0)

    def num_samples_to_frames(self, num_samples: int):
        n_fft = self.feature_extraction.n_fft
        hop_length = self.feature_extraction.hop_length

        if self.feature_extraction.center:
            return (num_samples // hop_length) + 1
        else:
            return ((num_samples - n_fft) // hop_length) + 1

    def get_chunk_size(self, **kwargs):
        if self.div_chunk_size:
            assert "conformer_in" in kwargs
            conformer_in = kwargs.get("conformer_in")

            divisor = conformer_in.size(1)
            for i in range(conformer_in.size(1)-1, 20, -1):
                if conformer_in.size(1) % i == 0:
                    divisor = i
                    break

            return divisor

        elif self.chunk_ranges_weights is not None:
            assert "raw_audio" in kwargs
            raw_audio = kwargs.get("raw_audio")
            chunk_ranges_weights = torch.tensor(self.chunk_ranges_weights, device=raw_audio.device).float()
            chunk_ranges = torch.linspace(start=self.min_chunk_size, 
                                          end=min(raw_audio.size(1), self.max_chunk_size), 
                                          steps=2*chunk_ranges_weights.size(0)).view(-1, 2)

            chunk_size = sample_chunk_size(chunk_ranges_weights, chunk_ranges, self.choose_seq_len_p)
        
        else:
            chunk_size = self.chunk_size

        return self.num_samples_to_frames(chunk_size)

    def num_samples_to_frames(self, num_samples: int):
        n_fft = self.feature_extraction.n_fft
        hop_length = self.feature_extraction.hop_length

        if self.feature_extraction.center:
            return (num_samples // hop_length) + 1
        else:
            return ((num_samples - n_fft) // hop_length) + 1

    def get_chunk_size(self, **kwargs):
        if self.div_chunk_size:
            assert "conformer_in" in kwargs
            conformer_in = kwargs.get("conformer_in")

            divisor = conformer_in.size(1)
            for i in range(conformer_in.size(1)-1, 20, -1):
                if conformer_in.size(1) % i == 0:
                    divisor = i
                    break

            return divisor

        elif self.chunk_ranges_weights is not None:
            assert "raw_audio" in kwargs
            raw_audio = kwargs.get("raw_audio")
            chunk_ranges_weights = torch.tensor(self.chunk_ranges_weights, device=raw_audio.device).float()
            chunk_ranges = torch.linspace(start=self.min_chunk_size, 
                                          end=min(raw_audio.size(1), self.max_chunk_size), 
                                          steps=2*chunk_ranges_weights.size(0)).view(-1, 2)

            chunk_size = sample_chunk_size(chunk_ranges_weights, chunk_ranges, self.choose_seq_len_p)
        
        else:
            chunk_size = self.chunk_size

        return self.num_samples_to_frames(chunk_size)


    def forward(
        self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, labels: torch.Tensor, labels_len: torch.Tensor,
            chunk_size: Optional[int] = None
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, S]
        :param labels_len: length of S as [B]
        :param chunk_size: number of audio samples in one chunk

        :return: logprobs [B, T + S, #labels + blank]
        """
        batch_size = raw_audio.size(0)

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
        # >>> [B, T', F]
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)
        
        # random chunk_size based on the weights on certain chunk_size ranges
        if self.chunk_ranges_weights.numel() > 0:
            chunk_ranges = torch.linspace(start=self.min_chunk_size, 
                                          end=min(raw_audio.size(1), self.max_chunk_size), 
                                          steps=2*self.chunk_ranges_weights.size(0)).view(-1, 2)

            chunk_size = sample_chunk_size(self.chunk_ranges_weights, chunk_ranges, self.choose_seq_len_p)

        # get chunk size in #frames        
        chunk_size_frames = self.get_chunk_size(raw_audio=raw_audio, conformer_in=conformer_in)

        chunk_size_frames = min(chunk_size_frames, conformer_in.size(1))

        # get chunk size in #frames        
        chunk_size_frames = self.get_chunk_size(raw_audio=raw_audio, conformer_in=conformer_in)

        # pad and reshape for naive streaming encoder using independent chunks
        # TODO: clip to time dim if time < chunk_size
        time_dim_pad = -conformer_in.size(1) % chunk_size_frames
        # (B, T, ...) -> (B, T+time_dim_pad, ...) = (B, T', ...)
        conformer_in = torch.nn.functional.pad(conformer_in, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = torch.nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        print(f"{chunk_size_frames = }, {conformer_in.size(-1) = }, {conformer_in.size() = }")
        conformer_in = conformer_in.view(-1, chunk_size_frames, conformer_in.size(-1))  # (B*(T'/C), C, F) = (B*N, C, F)
        mask = mask.view(-1, chunk_size_frames)  # (B*N, C)
        extended_batch_size = conformer_in.size(0)  # B*N

        print(f"{batch_size = }, {extended_batch_size = }, number of chunks {extended_batch_size // batch_size}, {chunk_size_frames = }")
        print(f"{conformer_in.shape = }")

        # get all non-empty chunks to be passed into conformer
        valid_ind = torch.any(mask == True, dim=1).nonzero().flatten().to(conformer_in.device)
        good_chunks = conformer_in[valid_ind]   # (B*(N-invalid), C, F) = (B*N', C, F)

        conformer_out, out_mask = self.conformer(good_chunks, mask[valid_ind])  # (B*N', C', F')

        # replace fully padded chunks with 0 and 'copy' conformer output of non-empty chunks
        zeros_conf_out = torch.zeros(extended_batch_size, conformer_out.size(1), conformer_out.size(2), device=conformer_out.device)
        zeros_conf_out[valid_ind] = conformer_out
        conformer_out = zeros_conf_out  # (B*N', C', F')

        zeros_mask_out = torch.full(size=(extended_batch_size, out_mask.size(-1)), fill_value=False, device=out_mask.device)
        zeros_mask_out[valid_ind] = out_mask
        out_mask = zeros_mask_out   # (B*N', C')


        # merge chunks
        conformer_out = conformer_out.view(batch_size, -1, conformer_out.size(-1))  # (B, C'*N', conformer_dim)
        out_mask = out_mask.view(batch_size, -1)  # (B, C'*N')

        conformer_joiner_out = self.encoder_out_linear(conformer_out)   # (B*N', C', F'')
        conformer_out_lengths = torch.sum(out_mask, dim=1)  # [B, C'*N'] -> [B]

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, tgt_len = self.joiner(
            source_encodings=conformer_joiner_out,
            source_lengths=conformer_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, C'*N', S, #vocab + 1]

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

    rnnt_loss = warp_rnnt.rnnt_loss(
        log_probs=logprobs,
        frames_lengths=audio_features_len.to(dtype=torch.int32),
        labels=labels,
        labels_lengths=labels_len.to(dtype=torch.int32),
        blank=model.cfg.label_target_size,
        fastemit_lambda=0.0,
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
