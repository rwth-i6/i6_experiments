import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from i6_models.assemblies.conformer import ConformerBlockV1Config
from i6_models.parts.conformer import ConformerMHSAV1Config, ConformerConvolutionV1Config, ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer import ConformerMHSAV1, ConformerConvolutionV1, ConformerPositionwiseFeedForwardV1


from typing import Optional, Tuple

import torch


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
        x = self.mhsa(residual, key_padding_mask=key_padding_mask)  # [B, T, F]
        residual = x + residual  # [B, T, F]
        x = self.ff_2(residual)  # [B, T, F]
        x = 0.5 * x + residual  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x

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


class Conformer(torch.nn.Module):
    r"""Conformer architecture introduced in
    *Conformer: Convolution-augmented Transformer for Speech Recognition*
    :cite:`gulati2020conformer`.

    Extended version with very simple downsampling and upsampling

    Args:
        input_dim (int): input dimension.
        num_heads (int): number of attention heads in each Conformer layer.
        ffn_dim (int): hidden layer dimension of feedforward networks.
        num_layers (int): number of Conformer layers to instantiate.
        depthwise_conv_kernel_size (int): kernel size of each Conformer layer's depthwise convolution layer.
        dropout (float, optional): dropout probability. (Default: 0.0)
        use_group_norm (bool, optional): use ``GroupNorm`` rather than ``BatchNorm1d``
            in the convolution module. (Default: ``False``)
        convolution_first (bool, optional): apply the convolution module ahead of
            the attention module. (Default: ``False``)

    Examples:
        >>> conformer = Conformer(
        >>>     input_dim=80,
        >>>     num_heads=4,
        >>>     ffn_dim=128,
        >>>     num_layers=4,
        >>>     depthwise_conv_kernel_size=31,
        >>> )
        >>> lengths = torch.randint(1, 400, (10,))  # (batch,)
        >>> input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
        >>> output = conformer(input, lengths)
    """

    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        depthwise_conv_kernel_size: int,
        dropout: float = 0.0,
        use_group_norm: bool = False,
        convolution_first: bool = False,
    ):
        super().__init__()

        self.downsample_conv = torch.nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, stride=2, padding=2)

        block_config = ConformerBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                input_dim=input_dim,
                hidden_dim=ffn_dim,
                dropout=dropout
            ),
            conv_cfg=ConformerConvolutionV1Config(
                channels=input_dim,
                kernel_size=depthwise_conv_kernel_size,
                dropout=dropout,
                activation=nn.functional.silu,
                norm=nn.BatchNorm1d(num_features=input_dim),
            ),
            mhsa_cfg=ConformerMHSAV1Config(
                input_dim=input_dim,
                num_att_heads=num_heads,
                att_weights_dropout=dropout,
                dropout=dropout,
            )
        )
        self.conformer_layers = torch.nn.ModuleList([
            ConformerBlockV1(block_config) for _ in range(num_layers)
        ])
        self.upsample_conv = torch.nn.ConvTranspose1d(in_channels=input_dim, out_channels=input_dim, kernel_size=5, stride=2, padding=1)
        self.export_mode = False

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            input (torch.Tensor): with shape `(B, T, input_dim)`.
            lengths (torch.Tensor): with shape `(B,)` and i-th element representing
                number of valid frames for i-th batch element in ``input``.

        Returns:
            (torch.Tensor, torch.Tensor)
                torch.Tensor
                    output frames, with shape `(B, T, input_dim)`
                torch.Tensor
                    output lengths, with shape `(B,)` and i-th element representing
                    number of valid frames for i-th batch element in output frames.
        """

        # downsampling is done as [B, F, T]
        input_downsampled = self.downsample_conv.forward(input.transpose(1,2)).transpose(1, 2)

        # also downsample the mask for training, in ONNX export we currently ignore the mask
        encoder_padding_mask = None if self.export_mode else _lengths_to_padding_mask((lengths+1)//2)

        # Conformer is applied as [B, T, F]
        x = input_downsampled
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)  # [B, T, F]

        conf_output = torch.permute(x, (0, 2 ,1))  # [B, F, T] for upsampling
        upsampled = self.upsample_conv(conf_output).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        out_upsampled = upsampled[:,0:input.size()[1],:]

        return out_upsampled, lengths


class Model(torch.nn.Module):
    """
    Do convolution first, with softmax dropout

    """

    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        conformer_size = 384
        target_size=12001
        self.initial_linear = nn.Linear(50, conformer_size)
        self.conformer = Conformer(
            input_dim=conformer_size,
            num_heads=4,
            ffn_dim=2048,
            num_layers=12,
            depthwise_conv_kernel_size=31,
            dropout=0.2,
            convolution_first=True,
        )
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

        conformer_out_dropped = nn.functional.dropout(conformer_out, p=0.2, training=self.training)
        logits = self.final_linear(conformer_out_dropped)  # [B, T, F]
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


# def export(*, model: Model, model_filename: str):
#     scripted_model = torch.jit.optimize_for_inference(torch.jit.script(model.eval()))
#     dummy_data = torch.randn(1, 30, 50, device="cpu")
#     dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
#     onnx_export(
#         scripted_model,
#         (dummy_data, dummy_data_len),
#         f=model_filename,
#         verbose=True,
#         input_names=["data", "data_len"],
#         output_names=["classes"],
#         dynamic_axes={
#             # dict value: manually named axes
#             "data": {0: "batch", 1: "time"},
#             "data_len": {0: "batch"},
#             "classes": {0: "batch", 1: "time"}
#         }
#     )
#

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


