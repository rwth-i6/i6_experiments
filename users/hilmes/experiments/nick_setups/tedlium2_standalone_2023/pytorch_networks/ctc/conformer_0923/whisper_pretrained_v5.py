"""
v3: with fix to finetune layer numbers (range +1)
v4: change loading of whisper
v5: add checks for dimensions
"""

import numpy as np
import torch
from torch import nn


from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig

from returnn.torch.context import get_run_ctx

from .whisper_pretrained_v2_cfg import ModelConfig


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


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, **kwargs):
        super().__init__()
        self.cfg = ModelConfig.from_dict(model_config_dict)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.model_dict = None

        self.whisper_cfg = self.cfg.whisper_config
        run_ctx = get_run_ctx()
        if run_ctx.global_step == 0 and run_ctx.epoch == 1:
            print("Load Whisper model parameters")
            self.whisper_feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(
                f"openai/whisper-{self.whisper_cfg.name}", cache_dir="/work/asr4/hilmes/debug/whisper/transformers/")
            self.whisper: WhisperModel = WhisperModel.from_pretrained(f"openai/whisper-{self.whisper_cfg.name}",
                                                                      cache_dir="/work/asr4/hilmes/debug/whisper/transformers/")
        else:
            self.whisper_feature_extractor = WhisperFeatureExtractor()
            self.whisper = WhisperModel(WhisperConfig().from_pretrained(
                f"openai/whisper-{self.whisper_cfg.name}", cache_dir="/work/asr4/hilmes/debug/whisper/transformers/"))

        if self.whisper_cfg.finetune_layer == True:
            for param in self.whisper.parameters():
                param.requires_grad_(True)
        else:
            for param in self.whisper.parameters():
                param.requires_grad_(False)
            for layer_num in range(1, self.whisper_cfg.finetune_layer + 1):
                for name, param in self.whisper.encoder.layers[-layer_num].named_parameters():
                    param.requires_grad_(True)
        for name, param in self.whisper.encoder.named_parameters():
            if param.requires_grad:
                print(name)
        #self.upsampling_layer = torch.nn.ConvTranspose1d(
        #    in_channels=self.whisper.config.d_model, out_channels=512, kernel_size=5, stride=2, padding=1
        #)

        self.final_linear = nn.Linear(512, self.cfg.label_target_size + 1)  # + CTC blank
        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """
        assert any(param.requires_grad for param in self.whisper.encoder.parameters()) or self.whisper_cfg.finetune_layer == 0
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        if squeezed_features.shape[1] > 160 * 3000:
            squeezed_features2 = squeezed_features[:, 160 * 3000:]
            squeezed_features2 = squeezed_features2.cpu().numpy()
            features2 = self.whisper_feature_extractor(raw_speech=squeezed_features2, return_tensors="pt",
                                                      return_attention_mask=True, sampling_rate=16000)
            features2 = features2.to(device="cuda" if torch.cuda.is_available() else "cpu")
            audio_features2 = features2["input_features"]
            attention_mask2 = features2["attention_mask"]
            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                input_features2 = self.whisper._mask_input_features(audio_features2, attention_mask=attention_mask2)
            else:
                input_features2 = audio_features2
            whisper_outputs2 = self.whisper.encoder(input_features=input_features2)
            encoder_output2 = whisper_outputs2.last_hidden_state
            encoder_output2 = self.final_dropout(encoder_output2)
            logits2 = self.final_linear(encoder_output2)

        squeezed_features = squeezed_features.cpu().numpy()
        features = self.whisper_feature_extractor(raw_speech=squeezed_features, return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
        features = features.to(device="cuda" if torch.cuda.is_available() else "cpu")
        audio_features = features["input_features"]
        attention_mask = features["attention_mask"]
        #audio_features_masked_2 = torch.transpose(audio_features_masked_2, 1, 2)  # B, F, T
        # TODO: try to remove specagument for now
        # TODO: fix dev set problems
        run_ctx = get_run_ctx()
        if self.training and run_ctx.epoch >= self.specaug_start_epoch:
            input_features = self.whisper._mask_input_features(audio_features, attention_mask=attention_mask)
        else:
            input_features = audio_features
        whisper_outputs = self.whisper.encoder(input_features=input_features)
        encoder_output = whisper_outputs.last_hidden_state
        encoder_output = self.final_dropout(encoder_output)
        logits = self.final_linear(encoder_output)
        if squeezed_features.shape[1] > 160 * 3000:
            logits = torch.cat((logits, logits2), dim=1)
            attention_mask = torch.cat((attention_mask, attention_mask2), dim=1)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(attention_mask, dim=1) // 2


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


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
