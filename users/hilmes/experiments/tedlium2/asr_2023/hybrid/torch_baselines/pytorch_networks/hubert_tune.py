import torch
from torch import nn
import returnn.frontend as rf
from torch.onnx import export as onnx_export
from transformers import HubertForCTC, AutoProcessor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import HubertEncoder
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1,VGG4LayerActFrontendV1Config
from dataclasses import dataclass
from typing import Union

@dataclass
class HubertCfg:
    model_name: str
    finetune_layers: Union[list, int]


def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],

class Model(nn.Module):

    def __init__(self, epoch, step, config_dict, **kwargs):
        super().__init__()
        self.cfg_cls = HubertCfg(**config_dict)
        self.model_name = self.cfg_cls.model_name
        self.config = HubertConfig.from_pretrained(f"facebook/hubert-{self.model_name}",  cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        if step == 0 and epoch == 1 and self.training:
            # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft",
            #     cache_dir='/work/asr4/hilmes/debug/whisper/hubert_processor')
            print("Loading Hubert Model Weights")
            self.hubert: HubertModel = HubertModel.from_pretrained(f"facebook/hubert-{self.model_name}",
                cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
            self.feature_projection = self.hubert.feature_projection
            self.hubert = self.hubert.encoder
            self.hubert.train()
        else:
            self.feature_projection = HubertModel(self.config).feature_projection
            self.hubert = HubertModel(self.config).encoder

        self.finetune_layers = self.cfg_cls.finetune_layers
        if self.training:
            for param in self.hubert.parameters():
                param.requires_grad_(False)
            if self.finetune_layers:
                if isinstance(self.finetune_layers, list):
                    for layer_num in self.finetune_layers:
                        print(layer_num)
                        for name, param in self.hubert.layers[layer_num].named_parameters():
                            param.requires_grad_(True)
                elif isinstance(self.finetune_layers, int) and self.finetune_layers is not True:
                    for layer_num in range(1, self.finetune_layers + 1):
                        for name, param in self.hubert.layers[-layer_num].named_parameters():
                            param.requires_grad_(True)
                elif self.finetune_layers is True:
                    for name, param in self.named_parameters():
                        print(name)
                        param.requires_grad_(True)
                else:
                    raise NotImplementedError
            else:
                pass
                assert False, "Are you sure you know what you are doing? No fine-tuning?"
            print(f"Check for grad")
            for name, param in self.named_parameters():
                if param.requires_grad:
                    print(name)

        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=5, stride=2, padding=1
        )
        self.start_linear = nn.Linear(80, 512)
        self.final_linear = nn.Linear(self.config.hidden_size, 9001)
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"
        # else len == 0

    def forward(self, audio_mel_features: torch.Tensor):
        proj_feat = self.start_linear(audio_mel_features)
        proj_feat = self.feature_projection(proj_feat)
        hubert_out = self.hubert(proj_feat).last_hidden_state
        upsampled = self.upsample_conv(hubert_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
        upsampled = upsampled[:, 0: audio_mel_features.size()[1], :]
        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, extern_data, **_kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    #audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    #audio_features = audio_features[indices, :, :]

    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits = model(
        audio_mel_features=audio_features,
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30
    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data,))
    onnx_export(
        scripted_model,
        (dummy_data,),
        f=model_filename,
        verbose=True,
        input_names=["data"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "classes": {0: "batch", 1: "time"},
        },
    )
