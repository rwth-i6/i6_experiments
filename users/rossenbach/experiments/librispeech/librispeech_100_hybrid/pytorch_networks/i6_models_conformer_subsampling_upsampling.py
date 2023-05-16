import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis
from torchaudio.models.conformer import Conformer

from i6_models.assemblies.conformer.conformer import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer import ConformerEncoderV1Config, ConformerFrontendV1Config, ConformerBlockV1Config, ConformerPositionwiseFeedForwardV1Config, ConformerConvolutionV1Config, ConformerMHSAV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1



from typing import Optional, Tuple

import torch


__all__ = ["Conformer"]


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask



class Model(torch.nn.Module):
    """
    Do convolution first, with softmax dropout

    """

    def __init__(self, epoch, step, **kwargs):
        super().__init__()

        conformer_size = 384
        target_size = 12001

        conv_cfg = ConformerConvolutionV1Config(channels=conformer_size, kernel_size=31, dropout=0.2, activation=nn.SiLU())
        mhsa_cfg = ConformerMHSAV1Config(embed_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2, dropout=0.2)
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(input_dim=conformer_size, hidden_dim=2048, activation=nn.SiLU(),
                                                          dropout=0.2)
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        front_cfg = ConformerFrontendV1Config(feature_dim=50, hidden_dim=conformer_size, dropout=0.2, conv_stride=2, conv_kernel=5,
                                              conv_padding=2, spec_aug_cfg=None)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, front_cfg=front_cfg, block_cfg=block_cfg)
        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)

        self.upsample_conv = torch.nn.ConvTranspose1d(in_channels=conformer_size, out_channels=conformer_size, kernel_size=5,
                                                      stride=2, padding=1)
        self.initial_linear = nn.Linear(50, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)

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


        conformer_in = self.initial_linear(audio_features_masked_2)

        conformer_out, _ = self.conformer(conformer_in, audio_features_len)

        upsampled = self.upsample_conv(conformer_out).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        upsampled = upsampled[:,0:input.size()[1],:]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


# scripted_model = None

def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:size1"][indices]

    #if scripted_model is None:
    #    model.eval()
    #    model.to("cpu")
    #    export_trace(model=model, model_filename="testdump.onnx")
    #    assert False

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    run_ctx.mark_as_loss(name="CE", loss=loss)

def export_trace(*, model: Model, model_filename: str):
    model.conformer.export_mode = True
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    # dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    dummy_data_len = torch.ones((1,))*30
    scripted_model = torch.jit.optimize_for_inference(torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len)))
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


