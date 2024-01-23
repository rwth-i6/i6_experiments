import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

import returnn.frontend as rf

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC, LayerNormNCConfig

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from typing import Callable, Union
from dataclasses import dataclass
from i6_models.config import ModelConfiguration, SubassemblyWithConfig


from typing import Optional, Tuple

import torch

IntTupleIntType = Union[Tuple[int, int], int]


import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Protocol, Tuple


class BaseFrontendInterface(Protocol):
    @abstractmethod
    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        raise NotImplementedError


class FrontendInterface(BaseFrontendInterface, nn.Module):
    @abstractmethod
    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        return tensor, sequence_mask


@dataclass
class VGG4LayerPoolFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layer
        pool_kernel_size: kernel size of pooling layer
        pool_padding: padding for pooling layer
        conv2_channels: number of channels for second conv layer
        conv2_stride: stride param for second conv layer
        conv3_channels: number of channels for third conv layer
        conv3_stride: stride param for third conv layer
        conv4_channels: number of channels for fourth layer
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        activation: activation function at the end
        linear_input_dim: input size of the final linear layer
        linear_output_dim: output size of the final linear layer
    """

    conv1_channels: int
    pool_kernel_size: IntTupleIntType
    pool_padding: Optional[IntTupleIntType]
    conv2_channels: int
    conv2_stride: IntTupleIntType
    conv3_channels: int
    conv3_stride: IntTupleIntType
    conv4_channels: int
    conv_kernel_size: IntTupleIntType
    conv_padding: Optional[IntTupleIntType]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    linear_input_dim: Optional[int]
    linear_output_dim: Optional[int]

    def check_valid(self):
        if isinstance(self.conv_kernel_size, int):
            assert self.conv_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"
        if isinstance(self.pool_kernel_size, int):
            assert self.pool_kernel_size % 2 == 1, "ConformerVGGFrontendV2 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


class VGG4LayerPoolFrontendV1(nn.Module):
    """
    Convolutional Front-End

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Activation
      - Pool
      - Conv
      - Activation
      - Conv
      - Activation
      - Conv

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerPoolFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (
            model_cfg.conv_padding if model_cfg.conv_padding is not None else _get_padding(model_cfg.conv_kernel_size)
        )
        pool_padding = model_cfg.pool_padding if model_cfg.pool_padding is not None else 0

        self.include_linear_layer = True if model_cfg.linear_output_dim is not None else False

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=model_cfg.conv1_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.pool = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            padding=pool_padding,
        )
        self.conv2 = nn.Conv2d(
            in_channels=model_cfg.conv1_channels,
            out_channels=model_cfg.conv2_channels,
            kernel_size=model_cfg.conv_kernel_size,
            stride=model_cfg.conv2_stride,
            padding=conv_padding,
        )
        self.conv3 = nn.Conv2d(
            in_channels=model_cfg.conv2_channels,
            out_channels=model_cfg.conv3_channels,
            kernel_size=model_cfg.conv_kernel_size,
            stride=model_cfg.conv3_stride,
            padding=conv_padding,
        )
        self.conv4 = nn.Conv2d(
            in_channels=model_cfg.conv3_channels,
            out_channels=model_cfg.conv4_channels,
            kernel_size=model_cfg.conv_kernel_size,
            padding=conv_padding,
        )
        self.activation = model_cfg.activation
        if self.include_linear_layer:
            self.linear = nn.Linear(
                in_features=model_cfg.linear_input_dim,
                out_features=model_cfg.linear_output_dim,
                bias=True,
            )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T'' depending on the stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: masking tensor of shape [B,T], contains length information of the sequences
        :return: torch.Tensor of shape [B,T',F']
        """
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        tensor = self.activation(tensor)
        tensor = self.pool(tensor)  # [B,C,T,F']

        tensor = self.conv2(tensor)  # [B,C,T',F']
        if _get_int_tuple_int(self.conv2.stride, 0) > 0:
            sequence_mask = _mask_pool(
                sequence_mask, self.conv2.kernel_size[0], self.conv2.stride[0], self.conv2.padding[0]
            )
        tensor = self.activation(tensor)

        tensor = self.conv3(tensor)  # [B,C,T",F']
        if _get_int_tuple_int(self.conv3.stride, 0) > 0:
            sequence_mask = _mask_pool(
                sequence_mask, self.conv3.kernel_size[0], self.conv3.stride[0], self.conv3.padding[0]
            )
        tensor = self.activation(tensor)

        tensor = self.conv4(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F']
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F']

        if self.include_linear_layer:
            tensor = self.linear(tensor)

        return tensor, sequence_mask


@dataclass
class VGG4LayerActFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layer
        conv2_channels: number of channels for second conv layer
        conv3_channels: number of channels for third conv layer
        conv4_channels: number of channels for fourth conv layer
        conv_kernel_size: kernel size of conv layers
        conv_padding: padding for the convolution
        pool1_kernel_size: kernel size of first pooling layer
        pool1_stride: stride of first pooling layer
        pool1_padding: padding for first pooling layer
        pool2_kernel_size: kernel size of second pooling layer
        pool2_stride: stride of second pooling layer
        pool2_padding: padding for second pooling layer
        activation: activation function at the end
        linear_input_dim: input size of the final linear layer
        linear_output_dim: output size of the final linear layer
    """

    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: IntTupleIntType
    conv_padding: Optional[IntTupleIntType]
    pool1_kernel_size: IntTupleIntType
    pool1_stride: Optional[IntTupleIntType]
    pool1_padding: Optional[IntTupleIntType]
    pool2_kernel_size: IntTupleIntType
    pool2_stride: Optional[IntTupleIntType]
    pool2_padding: Optional[IntTupleIntType]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    linear_input_dim: Optional[int]
    linear_output_dim: Optional[int]

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


class VGG4LayerActFrontendV1(nn.Module):
    """
    Convolutional Front-End

    The frond-end utilizes convolutional and pooling layers, as well as activation functions
    to transform a feature vector, typically Log-Mel or Gammatone for audio, into an intermediate
    representation.

    Structure of the front-end:
      - Conv
      - Conv
      - Activation
      - Pool
      - Conv
      - Conv
      - Activation
      - Pool

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: VGG4LayerActFrontendV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()

        model_cfg.check_valid()

        conv_padding = (
            model_cfg.conv_padding if model_cfg.conv_padding is not None else _get_padding(model_cfg.conv_kernel_size)
        )
        pool1_padding = model_cfg.pool1_padding if model_cfg.pool1_padding is not None else 0
        pool2_padding = model_cfg.pool2_padding if model_cfg.pool2_padding is not None else 0

        self.include_linear_layer = True if model_cfg.linear_output_dim is not None else False

        self.conv1 = nn.Conv2d(
            in_channels=1,
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
            stride=model_cfg.pool1_stride,
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
            stride=model_cfg.pool2_stride,
            padding=pool2_padding,
        )
        self.activation = model_cfg.activation
        if self.include_linear_layer:
            self.linear = nn.Linear(
                in_features=model_cfg.linear_input_dim,
                out_features=model_cfg.linear_output_dim,
                bias=True,
            )

    def forward(self, tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' or T'' depending on stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :param sequence_mask: the sequence mask for the tensor
        :return: torch.Tensor of shape [B,T",F'] and the shape of the sequence mask
        """
        # and add a dim
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        tensor = self.conv2(tensor)

        tensor = self.activation(tensor)
        tensor = self.pool1(tensor)  # [B,C,T',F']
        if sequence_mask is not None:
            sequence_mask = _mask_pool(
                sequence_mask,
                _get_int_tuple_int(self.pool1.kernel_size, 0),
                _get_int_tuple_int(self.pool1.stride, 0),
                _get_int_tuple_int(self.pool1.padding, 0),
            )

        tensor = self.conv3(tensor)
        tensor = self.conv4(tensor)

        tensor = self.activation(tensor)
        tensor = self.pool2(tensor)  # [B,C,T",F"]
        if sequence_mask is not None:
            sequence_mask = _mask_pool(
                sequence_mask,
                _get_int_tuple_int(self.pool2.kernel_size, 0),
                _get_int_tuple_int(self.pool2.stride, 0),
                _get_int_tuple_int(self.pool2.padding, 0),
            )

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T",C,F"]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T",C*F"]

        if self.include_linear_layer:
            tensor = self.linear(tensor)

        return tensor, sequence_mask


def _mask_pool(seq_mask: torch.Tensor, kernel_size: int, stride: int, padding: int) -> Optional[torch.Tensor]:
    """
    :param seq_mask: [B,T]
    :param kernel_size:
    :param stride:
    :param padding:
    :return: [B,T'] using maxpool
    """
    seq_mask = seq_mask.float()
    seq_mask = torch.unsqueeze(seq_mask, 1)  # [B,1,T]
    seq_mask = nn.functional.max_pool1d(seq_mask, kernel_size, stride, padding)  # [B,1,T']
    seq_mask = torch.squeeze(seq_mask, 1)  # [B,T']
    seq_mask = seq_mask.bool()
    return seq_mask


def _lengths_to_padding_mask(lengths: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return padding_mask.float()


def _get_int_tuple_int(variable: IntTupleIntType, index: int) -> int:
    return variable[index] if isinstance(variable, Tuple) else variable


def _get_padding(input_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    """
    get padding in order to not reduce the time dimension
    :param input_size:
    :return:
    """
    if isinstance(input_size, int):
        return (input_size - 1) // 2
    elif isinstance(input_size, tuple):
        return tuple((s - 1) // 2 for s in input_size)
    else:
        raise TypeError(f"unexpected size type {type(input_size)}")


class Model(torch.nn.Module):
    """
    Do convolution first, with softmax dropout

    """

    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        conformer_size = 384
        target_size = 9001

        conv_cfg = ConformerConvolutionV1Config(
            channels=conformer_size,
            kernel_size=31, # TODO change to 9 or 17
            dropout=0.2,
            activation=nn.SiLU(),
            norm=SubassemblyWithConfig(LayerNormNC, LayerNormNCConfig(channels=conformer_size)),
        )
        mhsa_cfg = ConformerMHSAV1Config(
            input_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2, dropout=0.2 # TODO heads: 6
        )
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=conformer_size, hidden_dim=2048, activation=nn.SiLU(), dropout=0.2 # TODO hidden Dim to 4x conformer size
        )
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = VGG4LayerActFrontendV1Config(
            linear_input_dim=1248,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=3,
            pool1_kernel_size=(1, 2),
            pool1_stride=(2, 1),
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=None,
            linear_output_dim=conformer_size,
            pool2_kernel_size=(1, 2),
            pool2_stride=None,
            pool2_padding=None,
        )

        frontend = SubassemblyWithConfig(module_class=VGG4LayerActFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)
        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)

        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=conformer_size, out_channels=conformer_size, kernel_size=5, stride=2, padding=1
        )
        # self.initial_linear = nn.Linear(80, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)
        self.export_mode = False

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_time_masked = mask_along_axis(audio_features, mask_param=20, mask_value=0.0, axis=1)
            audio_features_time_masked_2 = mask_along_axis(
                audio_features_time_masked, mask_param=20, mask_value=0.0, axis=1
            )
            audio_features_time_masked_3 = mask_along_axis(
                audio_features_time_masked_2, mask_param=20, mask_value=0.0, axis=1
            )
            audio_features_masked = mask_along_axis(audio_features_time_masked_3, mask_param=10, mask_value=0.0, axis=2)
            audio_features_masked_2 = mask_along_axis(audio_features_masked, mask_param=10, mask_value=0.0, axis=2)
        else:
            audio_features_masked_2 = audio_features

        # conformer_in = self.initial_linear(audio_features_masked_2)

        if not self.export_mode:
            mask = _lengths_to_padding_mask(audio_features_len)
        else:
            mask = None
        conformer_out, _ = self.conformer(audio_features_masked_2, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        upsampled = upsampled[:, 0 : audio_features.size()[1], :]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


# scripted_model = None


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]
    # from returnn.frontend import Tensor
    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]
    # if scripted_model is None:
    #     scripted_model = torch.jit.script(model)

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    # dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    dummy_data_len = torch.ones((1,)) * 30
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
            "classes": {0: "batch", 1: "time"},
        },
    )


def forward_step(*, model: Model, extern_data, **kwargs):
    """
    Function used in inference.
    """
    data = extern_data["data"]
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]
    # from returnn.frontend import Tensor
    log_probs, logits = model(audio_features, audio_features_len.to("cuda"))
    rf.get_run_ctx().mark_as_default_output(tensor=log_probs)
