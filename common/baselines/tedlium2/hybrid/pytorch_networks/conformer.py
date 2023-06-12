import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis
from torchaudio.models.conformer import Conformer

import returnn.frontend as rf

from i6_models.assemblies.conformer.conformer import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer import ConformerEncoderV1Config, ConformerBlockV1Config, ConformerPositionwiseFeedForwardV1Config, ConformerConvolutionV1Config, ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from typing import Callable, Union
from dataclasses import dataclass
from i6_models.config import ModelConfiguration, SubassemblyWithOptions


from typing import Optional, Tuple

import torch


__all__ = ["Conformer"]

@dataclass
class ConformerVGGFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: feature dimension of input
        conv1_channels: number of channels for first conv layers
        conv2_channels: number of channels for second conv layers
        conv3_channels: number of channels for third conv layers
        conv4_channels: number of channels for fourth dconv layers
        conv_kernel_size: kernel size of conv layers
        pool1_kernel_size: kernel size of first pooling layer
        pool1_strides: strides of first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_strides: strides of second pooling layer
        activation: activation function at the end
    """

    in_features: int
    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Union[int, Tuple[int, ...]]
    pool1_kernel_size: Union[int, Tuple[int, ...]]
    pool1_strides: Optional[Union[int, Tuple[int, ...]]]
    pool2_kernel_size: Union[int, Tuple[int, ...]]
    pool2_strides: Optional[Union[int, Tuple[int, ...]]]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

    def check_valid(self):
        if isinstance(self.conv_kernel_size, int):
            assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool1_kernel_size, int):
            assert self.pool1_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"
        if isinstance(self.pool2_kernel_size, int):
            assert self.pool2_kernel_size % 2 == 1, "ConformerVGGFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class ConformerVGGFrontendV1(nn.Module):
    """
    Convolutional Front-End
    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerVGGFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = _get_padding(model_cfg.conv_kernel_size)
        pool1_padding = _get_padding(model_cfg.pool1_kernel_size)
        pool2_padding = _get_padding(model_cfg.pool2_kernel_size)

        self.conv1 = nn.Conv2d(
            in_channels=model_cfg.in_features,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=model_cfg.pool1_kernel_size,
            stride=model_cfg.pool1_strides,
            padding=pool1_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.conv2_channels,
            out_channels=model_cfg.conv3_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.conv3_channels,
            out_channels=model_cfg.conv4_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=model_cfg.pool2_kernel_size,
            stride=model_cfg.pool2_strides,
            padding=pool2_padding,
        )
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        T might be reduced to T' depending on ...
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F]
        """
        # conv 2d layers expect shape [B,F,T,C] so we have to transpose here
        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        # and add a dim
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pool1(tensor)

        # conv 2d layers expect shape [B,F,T,C] so we have to transpose here
        tensor = torch.transpose(tensor, 1, 2)  # [B,F,T]
        tensor = tensor[:, :, :, None]  # [B,F,T,C]
        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)
        tensor = torch.transpose(tensor, 1, 2)  # transpose back to [B,T,F,C]
        tensor = torch.squeeze(tensor)  # [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)

        return tensor

def _get_padding(input_size: Union[int, Tuple[int, ...]]) -> int:
    if isinstance(input_size, int):
        out = (input_size - 1) // 2
    elif isinstance(input_size, tuple):
        out = min(input_size) // 2
    else:
        raise NotImplementedError

    return out

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

        conv_cfg = ConformerConvolutionV1Config(channels=conformer_size, kernel_size=31, dropout=0.2, activation=nn.SiLU(), norm=LayerNormNC(conformer_size))
        mhsa_cfg = ConformerMHSAV1Config(input_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2, dropout=0.2)
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(input_dim=conformer_size, hidden_dim=2048, activation=nn.SiLU(),
                                                          dropout=0.2)
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = ConformerVGGFrontendV1Config(in_features=80, activation=nn.SiLU(), conv1_channels=32, conv2_channels=32, conv3_channels=64, conv4_channels=64,
            conv_kernel_size=(3, 3), pool1_kernel_size=(3, 3), pool2_kernel_size=(3, 3), pool1_strides=1, pool2_strides=1)
        frontend = SubassemblyWithOptions(module_class=ConformerVGGFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)
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
        upsampled = upsampled[:,0:audio_features.size()[1],:]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


# scripted_model = None

def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"]
    audio_features_len = audio_features.dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = extern_data["classes"][indices, :]
    phonemes_len = phonemes.dims[1].dyn_size_ext.raw_tensor

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

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)

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


