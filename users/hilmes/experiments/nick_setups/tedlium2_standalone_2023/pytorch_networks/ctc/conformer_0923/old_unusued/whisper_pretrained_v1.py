"""
Based on i6modelsV1_VGG4LayerActFrontendV1_v5, modified to include whisper pretraining.
"""

import numpy as np
import torch
from torch import nn

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

import whisper
from i6_experiments.users.hilmes.experiments.nick_setups.tedlium2_standalone_2023.pytorch_networks.ctc.conformer_0923.old_unusued.whisper_modules_v1 import \
    Whisper
from whisper.audio import N_FRAMES, pad_or_trim

from returnn.torch.context import get_run_ctx

from .whisper_pretrained_v1_cfg import ModelConfig


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
        fe_config = LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
        )
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.specaug_start_epoch = self.cfg.specauc_start_epoch
        self.model_dict = None

        self.whisper_cfg = self.cfg.whisper_config
        if self.whisper_cfg.just_encoder:
            with open(f"/work/asr4/hilmes/debug/whisper/{self.whisper_cfg.name}.pt", "rb") as f:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.whisper_checkpoint = torch.load(f, map_location=device)
                self.whisper_dims = whisper.ModelDimensions(**self.whisper_checkpoint["dims"])
            self.whisper = Whisper(self.whisper_dims, self.whisper_cfg.dropout)
        else:
            raise NotImplementedError

        self.upsampling_layer = torch.nn.ConvTranspose1d(
            in_channels=self.whisper.dims.n_audio_state, out_channels=512, kernel_size=5, stride=2, padding=1
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
            self.model_dict = self.whisper.state_dict()
            print(self.model_dict.keys())
            pretrained_dict = {k: v for k, v in self.whisper_checkpoint["model_state_dict"].items() if k in self.model_dict}
            print(pretrained_dict.keys())
            self.whisper.load_state_dict(pretrained_dict)
        for param in self.whisper.parameters():
            param.requires_grad_(False)
        for layer_num in range(self.whisper_cfg.finetune_layer):
            for param in self.whisper.encoder.blocks[-layer_num].parameters():
                param.requires_grad_(True)
        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)
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
        audio_features_masked_2 = torch.transpose(audio_features_masked_2, 1, 2)
        if audio_features_masked_2.shape[-1] > N_FRAMES:
            audio_features_masked_2 = pad_or_trim(audio_features_masked_2, 2 * N_FRAMES, axis=-1)
            trans_audio_mel_features_1: torch.Tensor = audio_features_masked_2.index_select(
                dim=-1, index=torch.arange(end=N_FRAMES, device=audio_features_masked_2.device)
            )
            trans_audio_mel_features_2: torch.Tensor = audio_features_masked_2.index_select(
                dim=-1, index=torch.arange(start=N_FRAMES, end=2 * N_FRAMES, device=audio_features_masked_2.device)
            )
            x_1: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_1)
            x_1 = self.upsampling_layer(x_1.transpose(1, 2)).transpose(1, 2)
            x_2: torch.Tensor = self.whisper.encoder(trans_audio_mel_features_2)
            x_2 = self.upsampling_layer(x_2.transpose(1, 2)).transpose(1, 2)
            x = torch.cat((x_1, x_2), dim=1)
        else:
            audio_features_masked_2 = pad_or_trim(audio_features_masked_2, N_FRAMES)
            x: torch.Tensor = self.whisper.encoder(audio_features_masked_2)
            x = self.upsampling_layer(x.transpose(1, 2)).transpose(1, 2)
        # create the mask for the conformer input
        out_mask = mask_tensor(x, audio_features_len)
        conformer_out = self.final_dropout(x)
        conformer_out = conformer_out[:, :audio_features_len.max(), :]
        logits = self.final_linear(conformer_out)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


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
