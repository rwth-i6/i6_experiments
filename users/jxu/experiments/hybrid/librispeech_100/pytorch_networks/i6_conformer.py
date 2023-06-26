import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

import returnn.frontend as rf
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Union

from i6_models.i6_models.assemblies.conformer import ConformerBlockV1Config, ConformerEncoderV1Config, ConformerBlockV1, ConformerEncoderV1
from i6_models.i6_models.parts.conformer import ConformerMHSAV1Config, ConformerConvolutionV1Config, ConformerPositionwiseFeedForwardV1Config
from i6_models.i6_models.parts.conformer import ConformerMHSAV1, ConformerConvolutionV1, ConformerPositionwiseFeedForwardV1
from i6_models.i6_models.parts.conformer import LayerNormNC
from i6_models.i6_models.config import ModelConfiguration, SubassemblyWithConfig


@dataclass
class VGG4LayerPoolFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        conv1_channels: number of channels for first conv layers
        conv2_channels: number of channels for second conv layers
        conv3_channels: number of channels for third conv layers
        conv4_channels: number of channels for fourth conv layers
        conv_padding: padding for the convolution
        conv_kernel_size: kernel size of conv layers
        pool_kernel_size: kernel size of pooling layer
        pool_stride: stride of pooling layer
        pool_padding: padding for pooling layer
        activation: activation function at the end
    """

    conv1_channels: int
    conv2_channels: int
    conv3_channels: int
    conv4_channels: int
    conv_kernel_size: Union[int, Tuple[int, ...]]
    conv_padding: Optional[Union[int, Tuple[int, ...]]]
    conv4_stride: Optional[Union[int, Tuple[int, ...]]]
    pool_kernel_size: Union[int, Tuple[int, ...]]
    pool_stride: Optional[Union[int, Tuple[int, ...]]]
    pool_padding: Optional[Union[int, Tuple[int, ...]]]
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]

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
        self.pool = nn.MaxPool2d(
            kernel_size=model_cfg.pool_kernel_size,
            stride=model_cfg.pool_stride,
            padding=pool_padding,
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
            stride=model_cfg.conv4_stride,
            padding=conv_padding,
        )
        self.activation = model_cfg.activation
        self.initial_linear = nn.Linear(50, 384)
        self.downsample_conv = torch.nn.Conv1d(in_channels=384, out_channels=384, kernel_size=5, stride=2,
                                               padding=2)

    def forward(self, tensor: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' depending on the stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F']
        """
        # tensor = tensor[:, None, :, :]  # [B,C=1,T,F]
        #
        # tensor = self.conv1(tensor)
        # tensor = self.activation(tensor)
        #
        # # tensor = self.pool(tensor)
        #
        # tensor = self.conv2(tensor)
        # tensor = self.activation(tensor)
        #
        # tensor = self.conv3(tensor)
        # tensor = self.activation(tensor)
        #
        # tensor = self.conv4(tensor)
        #
        # tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T',C,F]
        # tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T',C*F]
        tensor = self.initial_linear(tensor)
        tensor = torch.transpose(tensor, 1, 2)
        tensor = self.downsample_conv(tensor)
        tensor = torch.transpose(tensor, 1, 2)

        return tensor, mask

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
    ) >= lengths.unsqueeze(1)
    return padding_mask


class Model(torch.nn.Module):
    def __init__(self, epoch, step, **kwargs):
        super().__init__()

        conformer_size = 384
        target_size = 12001

        cfg_front_end = VGG4LayerPoolFrontendV1Config(conv1_channels=32,conv2_channels=64,conv3_channels=64, conv4_channels=32, conv_kernel_size=3, pool_kernel_size=(1,2),
                                                      pool_stride=(1,2), conv4_stride=(2,1), activation=nn.SiLU(), conv_padding=None, pool_padding=None)
        frontend = SubassemblyWithConfig(module_class=VGG4LayerPoolFrontendV1, cfg=cfg_front_end)

        conv_cfg = ConformerConvolutionV1Config(channels=conformer_size, kernel_size=31, dropout=0.2, activation=nn.SiLU(),
                                                norm=LayerNormNC(conformer_size))
        mhsa_cfg = ConformerMHSAV1Config(input_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2, dropout=0.2)
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(input_dim=conformer_size, hidden_dim=2048, activation=nn.SiLU(),
                                                          dropout=0.1)

        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)

        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)
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

        # conformer_in = self.initial_linear(audio_features_masked_2)
        conformer_in = audio_features_masked_2

        # also downsample the mask for training, in ONNX export we currently ignore the mask
        mask = None if self.export_mode else _lengths_to_padding_mask((audio_features_len+1)//2)
        # mask = _lengths_to_padding_mask(audio_features_len)

        conformer_out, _ = self.conformer(conformer_in, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]

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
    #from returnn.frontend import Tensor
    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]

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