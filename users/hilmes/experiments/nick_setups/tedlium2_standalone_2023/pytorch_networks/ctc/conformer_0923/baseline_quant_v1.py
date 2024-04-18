"""
Like v2, but with i6_models specaugment (v3)
and now controllable start time for when specaugment is applied (v4)
and with the proper feature extraction from i6-models (v5)
"""
import math

import numpy as np
import torch
from torch import nn
import copy
from typing import Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

from returnn.torch.context import get_run_ctx

from .baseline_quant_v1_cfg import QuantModelTrainConfigV1, QuantModelConfigV1, ConformerPositionwiseFeedForwardQuantV1Config, QuantizedMultiheadAttentionV1Config, ConformerConvolutionQuantV1Config, ConformerBlockQuantV1Config, ConformerEncoderQuantV1Config
from .baseline_quant_v1_modules import LinearQuant, QuantizedMultiheadAttention

class ConformerPositionwiseFeedForwardQuant(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardQuantV1Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff = LinearQuant(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            bit_prec=cfg.bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            activation_quant_dtype=cfg.activation_quant_dtype,
            activation_quant_method=cfg.activation_quant_method,
            moving_average=cfg.moving_average,
            bias=True
        )
        self.activation = cfg.activation
        self.linear_out = LinearQuant(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            bit_prec=cfg.bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            activation_quant_dtype=cfg.activation_quant_dtype,
            activation_quant_method=cfg.activation_quant_method,
            moving_average=cfg.moving_average,
            bias=True
        )
        self.dropout = cfg.dropout

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        return tensor

class ConformerMHSAQuant(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: QuantizedMultiheadAttentionV1Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = QuantizedMultiheadAttention(cfg=cfg)
        self.dropout = cfg.dropout

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted
        which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, mask=inv_sequence_mask
        )  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor


class ConformerConvolutionQuant(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionQuantV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.pointwise_conv1 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            bit_prec=model_cfg.bit_prec_point,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            activation_quant_dtype=model_cfg.activation_quant_dtype,
            activation_quant_method=model_cfg.activation_quant_method,
            moving_average=model_cfg.moving_average,
            bias=True
        )
        # TODO: Quantize this
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
        )
        self.pointwise_conv2 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            bit_prec=model_cfg.bit_prec_point,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            activation_quant_dtype=model_cfg.activation_quant_dtype,
            activation_quant_method=model_cfg.activation_quant_method,
            moving_average=model_cfg.moving_average,
            bias=True
        )
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = copy.deepcopy(model_cfg.norm)
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

        tensor = self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pointwise_conv2(tensor)

        return self.dropout(tensor)


class ConformerBlockQuant(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockQuantV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAQuant(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionQuant(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  # [B, T, F]
        x = self.conv(x) + x  # [B, T, F]
        x = 0.5 * self.ff2(x) + x  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x


class ConformerEncoderQuant(nn.Module):
    """
    TODO
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderQuantV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockQuant(cfg.block_cfg) for _ in range(cfg.num_layers)])

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
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class Model(torch.nn.Module):
    def __init__(self, model_config_dict, quant_config_dict=None, **kwargs):
        epoch = kwargs.pop("epoch")
        step = kwargs.pop("step")
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"

        super().__init__()
        self.train_config = QuantModelTrainConfigV1.from_dict(model_config_dict)
        fe_config = LogMelFeatureExtractionV1Config(
            sample_rate=16000,
            win_size=0.025,
            hop_size=0.01,
            f_min=60,
            f_max=7600,
            min_amp=1e-10,
            num_filters=80,
            center=False,
        )
        frontend_config = self.train_config.frontend_config
        conformer_size = self.train_config.conformer_size
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        if quant_config_dict:
            print("Using Quantizable Model")
            self.cfg = QuantModelConfigV1.from_dict(quant_config_dict)
            conformer_config = ConformerEncoderQuantV1Config(
                num_layers=self.train_config.num_layers,
                frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
                block_cfg=ConformerBlockQuantV1Config(
                    ff_cfg=ConformerPositionwiseFeedForwardQuantV1Config(
                        input_dim=conformer_size,
                        hidden_dim=self.train_config.ff_dim,
                        dropout=self.train_config.ff_dropout,
                        activation=nn.functional.silu,
                        weight_quant_dtype=self.cfg.weight_quant_dtype,
                        weight_quant_method=self.cfg.weight_quant_method,
                        activation_quant_dtype=self.cfg.activation_quant_dtype,
                        activation_quant_method=self.cfg.activation_quant_method,
                        moving_average=self.cfg.moving_average,
                        bit_prec=self.cfg.bit_prec,
                    ),
                    mhsa_cfg=QuantizedMultiheadAttentionV1Config(
                        input_dim=conformer_size,
                        num_att_heads=self.train_config.num_heads,
                        att_weights_dropout=self.train_config.att_weights_dropout,
                        dropout=self.train_config.mhsa_dropout,
                        weight_quant_dtype=self.cfg.weight_quant_dtype,
                        weight_quant_method=self.cfg.weight_quant_method,
                        activation_quant_dtype=self.cfg.activation_quant_dtype,
                        activation_quant_method=self.cfg.activation_quant_method,
                        dot_quant_dtype=self.cfg.dot_quant_dtype,
                        dot_quant_method=self.cfg.dot_quant_method,
                        Av_quant_dtype=self.cfg.Av_quant_dtype,
                        Av_quant_method=self.cfg.Av_quant_method,
                        bit_prec_W_q=self.cfg.bit_prec,
                        bit_prec_W_k=self.cfg.bit_prec,
                        bit_prec_W_v=self.cfg.bit_prec,
                        bit_prec_dot=self.cfg.bit_prec,
                        bit_prec_A_v=self.cfg.bit_prec,
                        bit_prec_W_o=self.cfg.bit_prec,
                        moving_average=self.cfg.moving_average
                    ),
                    conv_cfg=ConformerConvolutionQuantV1Config(
                        channels=conformer_size,
                        kernel_size=self.train_config.conv_kernel_size,
                        dropout=self.train_config.conv_dropout,
                        activation=nn.functional.silu,
                        norm=LayerNormNC(conformer_size),
                        weight_quant_dtype=self.cfg.weight_quant_dtype,
                        weight_quant_method=self.cfg.weight_quant_method,
                        activation_quant_dtype=self.cfg.activation_quant_dtype,
                        activation_quant_method=self.cfg.activation_quant_method,
                        moving_average=self.cfg.moving_average,
                        bit_prec_point=self.cfg.bit_prec,
                    ),
                ),
            )
            self.conformer = ConformerEncoderQuant(cfg=conformer_config)
        else:
            conformer_config = ConformerEncoderV1Config(
                num_layers=self.train_config.num_layers,
                frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
                block_cfg=ConformerBlockV1Config(
                    ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                        input_dim=conformer_size,
                        hidden_dim=self.train_config.ff_dim,
                        dropout=self.train_config.ff_dropout,
                        activation=nn.functional.silu,
                    ),
                    mhsa_cfg=ConformerMHSAV1Config(
                        input_dim=conformer_size,
                        num_att_heads=self.train_config.num_heads,
                        att_weights_dropout=self.train_config.att_weights_dropout,
                        dropout=self.train_config.mhsa_dropout,
                    ),
                    conv_cfg=ConformerConvolutionV1Config(
                        channels=conformer_size,
                        kernel_size=self.train_config.conv_kernel_size,
                        dropout=self.train_config.conv_dropout,
                        activation=nn.functional.silu,
                        norm=LayerNormNC(conformer_size),
                    ),
                ),
            )
            self.conformer = ConformerEncoderV1(cfg=conformer_config)


        self.final_linear = nn.Linear(conformer_size, self.train_config.label_target_size + 1)  # + CTC blank TODO: do we quant here too?
        self.final_dropout = nn.Dropout(p=self.train_config.final_dropout)
        self.specaug_start_epoch = self.train_config.specauc_start_epoch

        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        squeezed_features = torch.squeeze(raw_audio)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.train_config.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.train_config.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.train_config.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.train_config.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.train_config.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))


def static_quant_init_hook(run_ctx, **kwargs):
    # These flags are not required for quant, only later for forward
    run_ctx.iterative_quant = False
    run_ctx.apply_quant = False
    run_ctx.tag_file = open("seq_tags.txt", "wt")


def static_quant_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]
    for tag in data["seq_tag"]:
        run_ctx.tag_file.write(tag + "\n")
    print(data['seq_tag'])
    _, _ = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

def static_quant_finish_hook(run_ctx, **kwargs):
    run_ctx.tag_file.close()
    torch.save({"model": run_ctx.engine._model.state_dict(), "epoch": 250, "step": run_ctx.engine._train_step}, "model.pt")