import torch
from torch import nn
import returnn.frontend as rf
from torch.onnx import export as onnx_export
from transformers import HubertForCTC, AutoProcessor, Wav2Vec2FeatureExtractor, HubertConfig, HubertModel
from transformers.models.hubert.modeling_hubert import HubertEncoder
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1,VGG4LayerActFrontendV1Config

def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],

class Model(nn.Module):

    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        self.config = HubertConfig.from_pretrained("facebook/hubert-large-ls960-ft",  cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
        if epoch == 1:
            # self.processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft",
            #     cache_dir='/work/asr4/hilmes/debug/whisper/hubert_processor')
            self.hubert = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft",
                cache_dir='/work/asr4/hilmes/debug/whisper/hubert_for_ctc')
            self.hubert = self.hubert.encoder
            self.hubert.train()
        else:
            self.hubert = HubertEncoder(self.config)

        frontend_cfg = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 1),
            pool1_kernel_size=(1, 2),
            pool1_stride=(2, 1),
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=None,
            out_features=self.config.hidden_size,
            pool2_kernel_size=(1, 2),
            pool2_stride=None,
            pool2_padding=None,
        )

        self.frontend = VGG4LayerActFrontendV1(model_cfg=frontend_cfg)
        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=self.config.hidden_size, out_channels=self.config.hidden_size, kernel_size=5, stride=2, padding=1
        )
        self.final_linear = nn.Linear(self.config.hidden_size, 9001)

    def forward(self, audio_mel_features: torch.Tensor, audio_features_len: torch.Tensor):
        mask = _lengths_to_padding_mask(audio_features_len, audio_mel_features)
        mask = torch.logical_xor(mask, torch.ones_like(mask))
        x, _ = self.frontend(tensor=audio_mel_features, sequence_mask=mask)
        hubert_out = self.hubert(x).last_hidden_state
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
    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = extern_data["classes"].raw_tensor
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    log_probs, logits = model(
        audio_mel_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
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
    dummy_data = torch.randn(1, 500, 80, device="cpu")
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