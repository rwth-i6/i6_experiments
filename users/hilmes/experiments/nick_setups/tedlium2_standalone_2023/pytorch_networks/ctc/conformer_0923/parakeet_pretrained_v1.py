"""
Same as v1 with fix to finetune layer numbers (range +1)
with additional fix to loading
with additional option for keep layers
"""

import numpy as np
import torch
from torch import nn

from returnn.torch.context import get_run_ctx
from .parakeet_pretrained_v1_cfg import ModelConfig
from nemo.collections.asr.models.ctc_models import EncDecCTCModel


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
        self.final_dropout = nn.Dropout(p=self.cfg.final_dropout)
        self.model_dict = None

        self.parakeet_config = self.cfg.parakeet_config
        self.parakeet_model_config = EncDecCTCModel.from_pretrained(
            model_name=f"nvidia/parakeet-{self.cfg.parakeet_config.name}",
            return_config=True
        )
        del self.parakeet_model_config['train_ds']
        del self.parakeet_model_config['validation_ds']
        del self.parakeet_model_config['test_ds']
        print(self.parakeet_model_config)
        for param in self.parakeet_model_config:
            print(param, self.parakeet_model_config[param])
        run_ctx = get_run_ctx()
        if not run_ctx.global_step and run_ctx.epoch == 1:
            print("Load Parakeet model parameters")
            tmp = EncDecCTCModel.from_pretrained(
                model_name=f"nvidia/parakeet-{self.cfg.parakeet_config.name}"
            )
            self.encoder = tmp.encoder
            self.spec_augmentation = tmp.spec_augmentation
            self.preprocessor = tmp.preprocessor
            del tmp
        else:
            self.encoder = EncDecCTCModel(self.parakeet_model_config).encoder
            self.spec_augmentation = EncDecCTCModel(self.parakeet_model_config).spec_augmentation
            self.preprocessor = EncDecCTCModel(self.parakeet_model_config).preprocessor

        if self.training:
            if self.parakeet_config.keep_layers is not None:
                if isinstance(self.encoder.keep_layers, list):
                    layer_ls = nn.ModuleList()
                    for num in self.parakeet_config.keep_layers:
                        layer_ls.append(self.encoder.layers[num])
                    for layer, num in zip(layer_ls, self.parakeet_config.keep_layers):
                        assert layer == self.encoder.layers[num], "Wrong layers were picked"
                elif isinstance(self.parakeet_config.keep_layers, int):
                    self.encoder.layers = self.encoder.layers[:self.parakeet_config.keep_layers+1]
                else:
                    raise NotImplementedError
            if self.parakeet_config.finetune_layer is True:
                for param in self.encoder.parameters():
                    param.requires_grad_(True)
            else:
                for param in self.encoder.parameters():
                    param.requires_grad_(False)
                for layer_num in range(1, self.parakeet_config.finetune_layer + 1):
                    for name, param in self.encoder.layers[-layer_num].named_parameters():
                        param.requires_grad_(True)
            for name, param in self.encoder.named_parameters():
                if param.requires_grad:
                    print(name)
        self.final_linear = nn.Linear(self.encoder._feat_out, self.cfg.label_target_size + 1)  # + CTC blank
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
        assert any(param.requires_grad for param in self.parameters()) or self.parakeet_config.finetune_layer == 0
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        features, feature_len = self.preprocessor(
                input_signal=squeezed_features, length=raw_audio_len,
            )
        if self.spec_augmentation is not None and self.training:
            spec_aug_out = self.spec_augmentation(input_spec=features, length=feature_len)
        else:
            spec_aug_out = features
        encoder_out, encoder_len = self.encoder(audio_signal=spec_aug_out, length=feature_len)
        encoder_out = encoder_out.transpose(1, 2)
        encoder_output = self.final_dropout(encoder_out)
        logits = self.final_linear(encoder_output)
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, encoder_len


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

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
