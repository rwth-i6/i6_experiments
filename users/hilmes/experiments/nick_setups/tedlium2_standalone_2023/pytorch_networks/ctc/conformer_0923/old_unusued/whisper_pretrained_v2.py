"""
Based on i6modelsV1_VGG4LayerActFrontendV1_v5, modified to include whisper pretraining.
"""

import numpy as np
import torch
from torch import nn


from transformers import WhisperModel, WhisperFeatureExtractor, WhisperConfig

from returnn.torch.context import get_run_ctx

from i6_experiments.users.hilmes.experiments.nick_setups.tedlium2_standalone_2023.pytorch_networks.ctc.conformer_0923.whisper_pretrained_v2_cfg import ModelConfig


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
        self.whisper_feature_extractor = WhisperFeatureExtractor()
        self.whisper = WhisperModel(WhisperConfig().from_pretrained(
                f"openai/whisper-{self.whisper_cfg.name}", cache_dir="/work/asr4/hilmes/debug/whisper/transformers/"))
        for param in self.whisper.parameters():
            param.requires_grad_(False)
        for layer_num in range(self.whisper_cfg.finetune_layer):
            for name, param in self.whisper.encoder.layers[-layer_num].named_parameters():
                param.requires_grad_(True)
                print(name)
                print(param)
        self.upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=self.whisper.config.d_model, out_channels=512, kernel_size=5, stride=2, padding=1
        )

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
        run_ctx = get_run_ctx()
        if run_ctx.global_step == 0 and run_ctx.epoch == 1:
            self.whisper_feature_extractor: WhisperFeatureExtractor = WhisperFeatureExtractor.from_pretrained(
                f"openai/whisper-{self.whisper_cfg.name}", cache_dir="/work/asr4/hilmes/debug/whisper/transformers/")
            self.whisper: WhisperModel = WhisperModel.from_pretrained(f"openai/whisper-{self.whisper_cfg.name}",
                                                                      cache_dir="/work/asr4/hilmes/debug/whisper/transformers/")
        assert any(param.require_grad for param in self.whisper.encoder.parameters()) or self.whisper_cfg.finetune_layer == 0
        squeezed_features = torch.squeeze(raw_audio)
        squeezed_features = squeezed_features.cpu().numpy()
        features = self.whisper_feature_extractor(raw_speech=squeezed_features, return_tensors="pt", return_attention_mask=True, sampling_rate=16000)
        features = features.to(device="cuda")
        audio_features = features["input_features"]
        attention_mask = features["attention_mask"]
        # TODO: try to remove specagument for now
        run_ctx = get_run_ctx()
        if self.training and run_ctx.epoch >= self.specaug_start_epoch:
            input_features = self.whisper._mask_input_features(audio_features, attention_mask=attention_mask)
        else:
            input_features = audio_features
        whisper_outputs = self.whisper.encoder(input_features=input_features)
        encoder_output = whisper_outputs.last_hidden_state
        encoder_output = self.final_dropout(encoder_output)
        encoder_output = self.upsampling_layer(encoder_output.transpose(1, 2)).transpose(1, 2)
        encoder_output = encoder_output[:, :torch.sum(attention_mask, dim=1).max(), :]
        logits = self.final_linear(encoder_output)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(attention_mask, dim=1)


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
