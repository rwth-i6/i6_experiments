import torch
from torch import nn
import returnn.frontend as rf
from dataclasses import dataclass
from typing import Dict, Any, Union, List
from nemo.collections.asr.models.ctc_models import EncDecCTCModel
#from nemo.collections.asr.modules import ConformerEncoder
#from nemo.collections.asr.modules import ConvASRDecoder
#from nemo.collections.asr.modules import AudioToMelSpectrogramPreprocessor
from omegaconf import DictConfig
@dataclass
class ParakeetConfig:
    model_name: str
    finetune_layers: Union[List, int, bool]

    def __post_init__(self):
        self.cfg: DictConfig = EncDecCTCModel.from_pretrained(
            model_name=f"nvidia/parakeet-{self.model_name}",
            return_config=True
        )

class Model(nn.Module):

    def __init__(self, epoch, step, config_dict, **kwargs):
        super().__init__()
        self.config = ParakeetConfig(**config_dict)
        del self.config.cfg['train_ds']
        del self.config.cfg['validation_ds']
        del self.config.cfg['test_ds']
        for param in self.config.cfg:
            print(param, self.config.cfg[param])
        if step == 0 and epoch == 1 and self.training:
            # Cache Dir needs to be set in apptainer here
            print("Loading Parakeet Model Weights")
            tmp = EncDecCTCModel.from_pretrained(
                model_name=f"nvidia/parakeet-{self.config.model_name}"
            )
            self.encoder = tmp.encoder
            self.spec_augmentation = tmp.spec_augmentation
            del tmp
        else:
            self.encoder = EncDecCTCModel(self.config.cfg).encoder
            self.spec_augmentation = EncDecCTCModel(self.config.cfg).spec_augmentation

        if self.training:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            if self.config.finetune_layers:
                if isinstance(self.config.finetune_layers, list):
                    for layer_num in self.config.finetune_layers:
                        for name, param in self.encoder.layers[layer_num].named_parameters():
                            print(name)
                            param.requires_grad_(True)
                elif isinstance(self.config.finetune_layers, int) and self.config.finetune_layers is not True:
                    for layer_num in range(1, self.config.finetune_layers + 1):
                        for name, param in self.encoder.layers[-layer_num].named_parameters():
                            param.requires_grad_(True)
                elif self.config.finetune_layers is True:
                    for name, param in self.encoder.named_parameters():
                        print(name)
                        param.requires_grad_(True)
                else:
                    raise NotImplementedError
            from itertools import chain
            print("Check for trainable parameters")
            for name, param in (self.encoder.named_parameters()):
                if param.requires_grad:
                    print(name)

        self.downsample = nn.MaxPool2d(
            kernel_size=(2, 1),
            stride=(2, 1),
            padding=0,
        )

        self.up_linear = nn.Linear(self.encoder._feat_out, 256)
        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.upsample_conv2 = torch.nn.ConvTranspose1d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.upsample_conv3 = torch.nn.ConvTranspose1d(
            in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0)
        self.final_linear = nn.Linear(256, 9001)

        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"
        # else len == 0

    def forward(self, features: torch.Tensor, feature_len: torch.Tensor):
        assert any(param.requires_grad for param in self.parameters()) or self.config.finetune_layers == 0

        feature_len = feature_len.to(dtype=torch.int64)
        features = features.transpose(1, 2)  # [B, F, T]
        encoder_out, encoder_len = self.encoder(audio_signal=features, length=feature_len)  # [B, F, T]
        lin_out = self.up_linear(encoder_out.transpose(1, 2)).transpose(1, 2)  # [B, F, T]
        upsampled = self.upsample_conv(lin_out)
        upsampled = self.upsample_conv2(upsampled)
        upsampled = self.upsample_conv3(upsampled).transpose(1, 2)  # [B, T, F]
        upsampled = upsampled[:, :torch.max(feature_len)-1, :]
        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order, feature_len


def train_step(*, model: Model, extern_data, **_kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits, features_len = model(
        features=audio_features,
        feature_len=audio_features_len.to("cuda"),
    )
    features_len = features_len.to("cpu")
    #log_probs = log_probs[:, phonemes_len, :]
    #logits = logits[:, phonemes_len, :]
    assert all(phonemes_len == features_len), (phonemes_len, features_len)
    phonemes_len = features_len - 1
    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)

#
# def export2(*, model: Model, f: str):
#     model.export_mode = True
#     dummy_data = torch.randn(1, 45000, 1, device="cpu")
#     dummy_data_len = torch.IntTensor([45000])
#     scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
#     onnx_export(
#         scripted_model,
#         (dummy_data, dummy_data_len),
#         f=f,
#         verbose=True,
#         input_names=["data", "data_len"],
#         output_names=["classes"],
#         opset_version=17,
#         dynamic_axes={
#             # dict value: manually named axes
#             "data": {0: "batch", 1: "time"},
#             "data_len": {0: "batch"},
#             "classes": {0: "batch", 1: "time"},
#         },
#     )
