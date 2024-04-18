import torch
from torch import nn
import returnn.frontend as rf
from torch.onnx import export as onnx_export
from transformers import HubertConfig, HubertModel


class Model(nn.Module):

    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        self.config = HubertConfig.from_pretrained(f"facebook/hubert-{kwargs['hubert_model']}",  cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        if step == 0 and epoch == 1 and self.training:
            # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft",
            #     cache_dir='/work/asr4/hilmes/debug/whisper/hubert_processor')
            print("Loading Hubert Model Weights")
            self.hubert = HubertModel.from_pretrained(f"facebook/hubert-{kwargs['hubert_model']}",
                cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        else:
            self.hubert: HubertModel = HubertModel(
                HubertConfig.from_pretrained(f"facebook/hubert-{kwargs['hubert_model']}",
                                             cache_dir="/work/asr4/hilmes/debug/whisper/transformers/"))
        self.finetune_layers = kwargs.pop("finetune_layers", None)
        self.finetune_feature_extractor = kwargs.pop("finetune_feature_extractor", False)
        if self.training:
            for param in self.hubert.parameters():
                param.requires_grad_(False)
            if self.finetune_feature_extractor:
                for param in self.hubert.feature_extractor.parameters():
                    param.requires_grad_(True)
            if self.finetune_layers:
                if isinstance(self.finetune_layers, list):
                    for layer_num in self.finetune_layers:
                        for name, param in self.hubert.encoder.layers[layer_num].named_parameters():
                            print(name)
                            param.requires_grad_(True)
                elif isinstance(self.finetune_layers, int):
                    for layer_num in range(1, self.finetune_layers + 1):
                        for name, param in self.hubert.encoder.layers[-layer_num].named_parameters():
                            param.requires_grad_(True)
                else:
                    raise NotImplementedError
            print("Check for trainable parameters")
            for name, param in self.hubert.named_parameters():
                if param.requires_grad:
                    print(name)

        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=5, stride=2, padding=1)
        self.final_linear = nn.Linear(self.config.hidden_size, 9001)

    def forward(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        assert any(param.requires_grad for param in self.hubert.parameters()) or self.hubert_cfg.finetune_layer == 0
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        hubert_outputs = self.hubert(input_values=squeezed_features)
        audio_features_size = self.hubert._get_feat_extract_output_lengths(raw_audio_len).to(dtype=torch.int64)
        encoder_output = hubert_outputs.last_hidden_state
        upsampled = self.upsample_conv(encoder_output.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
        upsampled = upsampled[:, :torch.max(audio_features_size)*2, :]
        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order, audio_features_size*2


def train_step(*, model: Model, extern_data, **_kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits, features_len = model(
        raw_audio=audio_features,
        raw_audio_len=audio_features_len.to("cuda"),
    )
    features_len = features_len.to("cpu")
    assert all(torch.round(phonemes_len - 1.5).to(dtype=torch.int32) == features_len), (torch.round(phonemes_len - 1.5).to(dtype=torch.int32), features_len)
    phonemes_len = features_len
    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)
    targets_masked = targets_masked.long()

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export2(*, model: Model, f: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 45000, 1, device="cpu")
    dummy_data_len = torch.IntTensor([45000])
    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=f,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=17,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"},
        },
    )
