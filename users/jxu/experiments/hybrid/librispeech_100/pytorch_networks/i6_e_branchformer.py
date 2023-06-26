import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

import returnn.frontend as rf

from .i6_conformer_v2 import VGG4LayerPoolFrontendV1Config, VGG4LayerPoolFrontendV1, _lengths_to_padding_mask
from i6_models.i6_models.assemblies.e_branchformer import EbranchformerBlockV1Config, EbranchformerEncoderV1Config, EbranchformerrEncoderV1
from i6_models.i6_models.config import SubassemblyWithConfig
from i6_models.i6_models.parts.conformer import (
    ConformerMHSAV1Config as MHSAV1Config,
    ConformerPositionwiseFeedForwardV1Config as PositionwiseFeedForwardV1Config,
)
from i6_models.i6_models.parts.e_branchformer import (
    ConvolutionalGatingMLPV1Config,
    MergerV1Config,
)



class Model(torch.nn.Module):
    def __init__(self, epoch, step, **kwargs):
        super().__init__()

        conformer_size = 384
        target_size = 12001

        cfg_front_end = VGG4LayerPoolFrontendV1Config(input_size=50, conv1_channels=32,conv2_channels=64,conv3_channels=64, conv4_channels=32, conv_kernel_size=3, pool_kernel_size=(1,2),
                                                      pool_stride=(1,2), conv4_stride=(2,1), activation=nn.ReLU(), conv_padding=None, pool_padding=None)
        frontend = SubassemblyWithConfig(module_class=VGG4LayerPoolFrontendV1, cfg=cfg_front_end)

        cgmlp_cfg = ConvolutionalGatingMLPV1Config(input_dim=conformer_size, hidden_dim=6*conformer_size, kernel_size=31, dropout=0.1, activation=nn.SiLU())
        mhsa_cfg = MHSAV1Config(input_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2, dropout=0.2)
        ff_cfg = PositionwiseFeedForwardV1Config(input_dim=conformer_size, hidden_dim=1536, activation=nn.SiLU(),
                                                          dropout=0.1)
        merger_cfg =MergerV1Config(input_dim=384, kernel_size=31, dropout=0.1)

        block_cfg = EbranchformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, cgmlp_cfg=cgmlp_cfg, merger_cfg=merger_cfg)
        ebranchformer_cfg = EbranchformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)

        self.e_branchformer = EbranchformerrEncoderV1(cfg=ebranchformer_cfg)
        self.upsample_conv = torch.nn.ConvTranspose1d(in_channels=conformer_size, out_channels=conformer_size, kernel_size=5,
                                                      stride=2, padding=1)
        self.initial_linear = nn.Linear(50, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)

        self.export_mode = False

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_3 = mask_along_axis(audio_features_time_masked_2, mask_param=20, mask_value=0.0, axis=1)
            audio_features_masked = mask_along_axis(audio_features_time_masked_3, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features

        e_branchformer_in = audio_features_masked_2

        mask = None if self.export_mode else _lengths_to_padding_mask((audio_features_len + 1) // 2)

        e_branchformer_out, _ = self.e_branchformer(e_branchformer_in, mask)

        upsampled = self.upsample_conv(e_branchformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        upsampled = upsampled[:,0:audio_features.size()[1],:]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.1, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]
    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)
    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export_trace(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len = torch.ones((1,), device="cpu", dtype=torch.int32) * 30

    scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))

    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"}
        }
    )
