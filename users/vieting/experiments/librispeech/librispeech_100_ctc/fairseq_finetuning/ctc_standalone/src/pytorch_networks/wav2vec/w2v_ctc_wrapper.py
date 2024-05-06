import numpy as np
import torch
from torch import nn

import fairseq
from fairseq.dataclass import FairseqDataclass
from fairseq.models.wav2vec.wav2vec2_asr import Wav2VecCtc, Wav2Vec2CtcConfig

from .w2v_ctc_wrapper_cfg import ModelConfig

class Model(torch.nn.Module):
    def __init__(self, w2v_config_updates, **kwargs):
        """
        Args:
            w2v_config_updates: dictionary containing the configuration updates for the default FairseqConfig
        """
        super().__init__()
        self.model_config = ModelConfig.from_dict(w2v_config_updates)
        self.fairseq_config = self.model_config.build_full_config()
        self.num_update = 0
        dummy_task = self.model_config.build_dummy_task() # dummy task to build the model
        self.w2v_model = Wav2VecCtc.build_model(self.fairseq_config.model, task=dummy_task) 

    def get_num_updates(self):
        """
        Get the number of gradient updates that have been performed so far
        """
        return self.num_update
    
    def set_num_updates(self, num_update):
        """
        Set the number of gradient updates that have been performed so far
        """
        self.num_update = num_update
        self.w2v_model.w2v_encoder.set_num_updates(num_update)

    def forward(
        self, 
        source,
        padding_mask,
        corpus_key=None
    ):
        """
        Args:
            source: [B, T'] tensor (B: batch size, T': raw audio length)
            padding_mask: [B, T'] tensor
            corpus_key: str
        Returns:
            logprobs: [T, B, C] tensor (C: vocab size, T: audio feature length)
            input_lengths: [B] tensor
        """
        # get model output
        res = self.w2v_model(source=source, padding_mask=padding_mask, corpus_key=corpus_key)
        model_out = res["encoder_out"] # [T, B, C]
        padding_mask = res["padding_mask"] # [T, B]
        logprobs = self.w2v_model.get_normalized_probs(res, log_probs=True).contiguous() # [T, B, C]

        # calculate feature lengths
        if padding_mask is not None:
            non_padding_mask = ~padding_mask
            input_lengths = non_padding_mask.long().sum(-1)
        else:
            input_lengths = model_out.new_full(
                    (logprobs.size(1),), logprobs.size(0), dtype=torch.long
                )
        return logprobs, input_lengths

def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"].squeeze(2)  # [B, T']
    raw_audio_len = data["raw_audio:size1"].to("cuda")  # [B] #TODO: check if this is correct
    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B]

    try:
        padding_mask = (raw_audio_len.unsqueeze(1) - torch.arange(raw_audio.size(1), device=raw_audio.device)) <= 0
    except IndexError:
        print("raw_audio_len", raw_audio_len.size())
        print("raw_audio", raw_audio.size())
        raise

    log_probs, input_lengths = model(
        source=raw_audio,
        padding_mask=padding_mask,
    ) # [T, B, C], [B]

    ctc_loss = nn.functional.ctc_loss(
        log_probs,
        labels,
        input_lengths=input_lengths,
        target_lengths=labels_len,
        blank=model.model_config.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)
    model.set_num_updates(model.get_num_updates() + 1)

def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))