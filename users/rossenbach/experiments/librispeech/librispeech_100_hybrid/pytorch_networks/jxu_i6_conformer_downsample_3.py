import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Union

from i6_models.assemblies.conformer import ConformerBlockV1Config, ConformerEncoderV1Config
from i6_models.parts.conformer import ConformerMHSAV1Config, ConformerConvolutionV1Config, \
    ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer import ConformerMHSAV1, ConformerPositionwiseFeedForwardV1
from i6_models.parts.conformer import LayerNormNC
from i6_models.config import ModelConfiguration, ModuleFactoryV1


class ConformerConvolutionV1(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.pointwise_conv1 = nn.Linear(in_features=model_cfg.channels, out_features=2 * model_cfg.channels)
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
        )
        self.pointwise_conv2 = nn.Linear(in_features=model_cfg.channels, out_features=model_cfg.channels)
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = nn.LayerNorm(model_cfg.channels)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.depthwise_conv(tensor)

        tensor = self.norm(tensor.transpose(1, -1)).transpose(1, -1)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pointwise_conv2(tensor)

        return self.dropout(tensor)


class ConformerBlockV1(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff_1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV1(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff_2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, tensor: torch.Tensor, key_padding_mask: Optional[torch.Tensor]):
        """
        :param tensor: input tensor of shape [B, T, F]
        :param Optional[torch.Tensor] key_padding_mask: could be a binary or float mask of shape (B, T)
        which will be applied/added to dot product, used to mask padded key positions out
        :return: torch.Tensor of shape [B, T, F]
        """
        assert tensor is not None
        residual = tensor  # [B, T, F]
        x = self.ff_1(residual)  # [B, T, F]
        residual = 0.5 * x + residual  # [B, T, F]
        x = self.conv(residual)  # [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.mhsa(residual, sequence_mask=key_padding_mask)  # [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.ff_2(residual)  # [B, T, F]
        x = 0.5 * x + residual  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x


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

    input_size: int
    output_size: int
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

        self.frontend_linear = nn.Linear(
            model_cfg.input_size // model_cfg.pool_kernel_size[1] * model_cfg.conv4_channels, model_cfg.output_size)

    def forward(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        T might be reduced to T' depending on the stride of the layers

        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T',F']
        """
        tensor = tensor[:, None, :, :]  # [B,C=1,T,F]

        tensor = self.conv1(tensor)
        tensor = self.activation(tensor)

        tensor = self.pool(tensor)

        tensor = self.conv2(tensor)
        tensor = self.activation(tensor)

        tensor = self.conv3(tensor)
        tensor = self.activation(tensor)

        tensor = self.conv4(tensor)

        tensor = torch.transpose(tensor, 1, 2)  # transpose to [B,T',C,F]
        tensor = torch.flatten(tensor, start_dim=2, end_dim=-1)  # [B,T',C*F]

        tensor = self.frontend_linear(tensor)

        return tensor


class ConformerEncoderV1(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()

        self.module_list = torch.nn.ModuleList([ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]
        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x = self.frontend(data_tensor)  # [B, T, F']

        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask


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
    Convert lengths to a sequence mask

    :param lengths: [B]
    :return: B x T, where 1 means within sequence and 0 means outside sequence
    """
    batch_size = lengths.shape[0]
    max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) < lengths.unsqueeze(1)
    return padding_mask


def apply_spec_aug(input, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
    """
    :param Tensor input: the input audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        input = mask_along_axis(input, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        input = mask_along_axis(input, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return input


class Model(torch.nn.Module):
    def __init__(self, **net_kwargs):
        super().__init__()
        
        self.net_kwargs = {
            "model_size": 256,
            "num_layers": 10,
            "kernel_size": 9,
            "num_repeat_time": 15,
            "max_dim_time": 20,
            "num_repeat_feat": 5,
            "max_dim_feat": 10,
        }

        
        conformer_size = self.net_kwargs["model_size"]
        num_layers = self.net_kwargs["num_layers"]
        kernel_size = self.net_kwargs["kernel_size"]
        target_size = 12001

        cfg_front_end = VGG4LayerPoolFrontendV1Config(input_size=50, output_size=conformer_size, conv1_channels=32,
                                                      conv2_channels=64, conv3_channels=64, conv4_channels=32,
                                                      conv_kernel_size=3, pool_kernel_size=(1, 2),
                                                      pool_stride=(1, 2), conv4_stride=(3, 1), activation=nn.ReLU(),
                                                      conv_padding=None, pool_padding=None)
        frontend = ModuleFactoryV1(module_class=VGG4LayerPoolFrontendV1, cfg=cfg_front_end)

        conv_cfg = ConformerConvolutionV1Config(channels=conformer_size, kernel_size=kernel_size, dropout=0.2,
                                                activation=nn.SiLU(),
                                                norm=LayerNormNC(conformer_size))
        mhsa_cfg = ConformerMHSAV1Config(input_dim=conformer_size, num_att_heads=4, att_weights_dropout=0.2,
                                         dropout=0.2)
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(input_dim=conformer_size, hidden_dim=conformer_size * 4,
                                                          activation=nn.SiLU(),
                                                          dropout=0.2)

        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=num_layers, frontend=frontend, block_cfg=block_cfg)

        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)
        self.upsample_conv = torch.nn.ConvTranspose1d(in_channels=conformer_size, out_channels=conformer_size,
                                                      kernel_size=3,
                                                      stride=3, padding=0)
        self.initial_linear = nn.Linear(50, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)

        self.export_mode = False

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_masked_2 = apply_spec_aug(audio_features, self.net_kwargs["num_repeat_time"],
                                                     self.net_kwargs["max_dim_time"],
                                                     self.net_kwargs["num_repeat_feat"],
                                                     self.net_kwargs["max_dim_feat"])
        else:
            audio_features_masked_2 = audio_features

        # conformer_in = self.initial_linear(audio_features_masked_2)
        conformer_in = audio_features_masked_2

        # also downsample the mask for training, in ONNX export we currently ignore the mask
        # mask = None if self.export_mode else _lengths_to_padding_mask((audio_features_len+1)//2)
        # also downsample the mask for training, in ONNX export we currently ignore the mask
        mask = None if self.export_mode else _lengths_to_padding_mask((audio_features_len + 2) // 3)

        conformer_out, _ = self.conformer(conformer_in, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        upsampled = upsampled[:, 0:audio_features.size()[1], :]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.1, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:size1"][indices]

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len.to("cpu"), batch_first=True,
                                                       enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    run_ctx.mark_as_loss(name="CE", loss=loss)


def export_trace(*, model: Model, args, f: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30

    #scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
    scripted_model = torch.jit.optimize_for_inference(torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len)))

    onnx_export(
        scripted_model,
        (dummy_data, dummy_data_len),
        f=f,
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
