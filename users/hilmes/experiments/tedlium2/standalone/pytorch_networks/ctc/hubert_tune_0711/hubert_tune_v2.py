"""
V2 with downsampling option
"""

import numpy as np
import torch
from torch import nn

from transformers import HubertModel, HubertConfig
from returnn.torch.context import get_run_ctx
from i6_models.parts.frontend.common import mask_pool

from .hubert_tune_v2_cfg import ModelConfig


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
        self.cfg: ModelConfig = ModelConfig.from_dict(model_config_dict)
        run_ctx = get_run_ctx()
        if not run_ctx.global_step and run_ctx.epoch == 1:
            print("Load Hubert model parameters")
            self.hubert: HubertModel = HubertModel.from_pretrained(f"facebook/hubert-{self.cfg.model_name}",
                                                                cache_dir="/work/asr4/hilmes/debug/whisper/transformers/")
        else:
            self.hubert: HubertModel = HubertModel(HubertConfig.from_pretrained(f"facebook/hubert-{self.cfg.model_name}",
                                                                   cache_dir="/work/asr4/hilmes/debug/whisper/transformers/"))
        if self.training:
            if self.cfg.keep_layers is not None:
                assert False, "Recheck if correct"
                if isinstance(self.cfg.keep_layers, list):
                    layer_ls = nn.ModuleList()
                    for num in self.cfg.keep_layers:
                        layer_ls.append(self.hubert.encoder.layers[num])
                    for layer, num in zip(layer_ls, self.cfg.keep_layers):
                        assert layer == self.hubert.encoder.layers[num], "Wrong layers were picked"
                elif isinstance(self.cfg.keep_layers, int):
                    self.hubert.encoder.layers = self.hubert.encoder.layers[:self.cfg.keep_layers+1]
                else:
                    raise NotImplementedError
            if self.cfg.finetune_layer is True:
                for param in self.hubert.parameters():
                    param.requires_grad_(True)
            else:
                assert False, "Recheck if correct"
                for param in self.hubert.parameters():
                    param.requires_grad_(False)
                for layer_num in range(1, self.cfg.finetune_layer + 1):
                    for name, param in self.hubert.encoder.layers[-layer_num].named_parameters():
                        param.requires_grad_(True)
            for name, param in self.hubert.encoder.named_parameters():
                if param.requires_grad:
                    print(name)
        if self.cfg.downsample_factor is not None:
            self.downsample = nn.MaxPool2d(
                kernel_size=(self.cfg.downsample_factor, 1),
                stride=(self.cfg.downsample_factor, 1),
                padding=(0, 0),
            )
        self.final_linear = nn.Linear(self.hubert.config.hidden_size, self.cfg.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)


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
        assert any(param.requires_grad for param in self.hubert.parameters()) or self.cfg.finetune_layer == 0
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        hubert_outputs = self.hubert(input_values=squeezed_features)
        encoder_output = hubert_outputs.last_hidden_state
        lens = self.hubert._get_feat_extract_output_lengths(raw_audio_len)
        if self.cfg.downsample_factor is not None:
            encoder_output = self.downsample(encoder_output)
            lens = lens // self.cfg.downsample_factor

        encoder_output = self.final_dropout(encoder_output)
        logits = self.final_linear(encoder_output)

        log_probs = torch.log_softmax(logits, dim=2)
        return log_probs, lens


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


def export(*, model: Model, f: str, **kwargs):
    from torch.onnx import export
    model.export_mode = True
    dummy_data = torch.randn(1, 30000, 1)
    dummy_data_len = torch.IntTensor([30000])
    export(
        model,
        (dummy_data, dummy_data_len),
        f=f,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        dynamic_axes={
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"},
        },
        opset_version=17,
    )

