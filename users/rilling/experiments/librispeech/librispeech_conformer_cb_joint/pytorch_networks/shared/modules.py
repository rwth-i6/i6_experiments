# Author: Jaehyeon Kim; copied from: https://github.com/jaywalnut310/glow-tts

import torch
from torch import nn
from torch.nn import functional as F
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1, ConformerBlockV1
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config

from .commons import fused_add_tanh_sigmoid_multiply, fused_add_tanh_sigmoid_multiply_no_jit

from . import commons
from . import attentions
from .mask import mask_tensor


class LinearBlock(torch.nn.Module):
    def __init__(self, in_size, out_size, p_dropout):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param p_dropout: dropout probability
        """
        super().__init__()
        self.lin = nn.Linear(in_size, out_size)
        self.norm = nn.LayerNorm(out_size)
        # self.norm = nn.BatchNorm1d(num_features=out_size)
        self.p_dropout = p_dropout

    def forward(self, x):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x = self.lin(x)
        # x = x.transpose(1,2)
        x = self.norm(x)
        # x = x.transpose(1,2)
        x = nn.functional.relu(x)
        x = nn.functional.dropout(x, p=self.p_dropout, training=self.training)
        return x


class Conv1DBlock(torch.nn.Module):
    """
    A 1D-Convolution with ReLU, batch-norm and non-broadcasted p_dropout
    Will pad to the same output length
    """

    def __init__(self, in_size, out_size, filter_size, p_dropout, norm="layer"):
        """
        :param in_size: input feature size
        :param out_size: output feature size
        :param filter_size: filter size
        :param p_dropout: dropout probability
        """
        super().__init__()
        assert filter_size % 2 == 1, "Only odd filter sizes allowed"
        self.conv = nn.Conv1d(in_size, out_size, filter_size, padding=filter_size // 2)
        self.ln = LayerNorm(channels=out_size)
        self.p_dropout = p_dropout

    def forward(self, x_with_mask):
        """
        :param x: [B, F_in, T]
        :return: [B, F_out, T]
        """
        x, x_mask = x_with_mask
        x = self.conv(x * x_mask)
        x = nn.functional.relu(x)
        x = self.ln(x)  # Layer normalization
        x = nn.functional.dropout(x, p=self.p_dropout, training=self.training)
        return (x, x_mask)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)

        x = (x - mean) * torch.rsqrt(variance + self.eps)

        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        assert n_layers > 1, "Number of layers should be larger than 0."

        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for i in range(self.n_layers):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

class WN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert kernel_size % 2 == 1
        assert hidden_channels % 2 == 0
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = torch.nn.ModuleList()
        self.res_skip_layers = torch.nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")

        for i in range(n_layers):
            dilation = dilation_rate**i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = torch.nn.Conv1d(
                hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation, padding=padding
            )
            in_layer = torch.nn.utils.weight_norm(in_layer, name="weight")
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = torch.nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = torch.nn.utils.weight_norm(res_skip_layer, name="weight")
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask=None, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            x_in = self.drop(x_in)

            if g is not None:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset : cond_offset + 2 * self.hidden_channels, :]
            else:
                g_l = torch.zeros_like(x_in)

            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)

            res_skip_acts = self.res_skip_layers[i](acts)

            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, : self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels :, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            torch.nn.utils.remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            torch.nn.utils.remove_weight_norm(l)
        for l in self.res_skip_layers:
            torch.nn.utils.remove_weight_norm(l)

class ConformerCouplingNet(nn.Module):
    def __init__(
            self, in_channels, hidden_channels, kernel_size, n_layers, n_heads=2, gin_channels = 0, p_dropout = 0
    ):
        super(ConformerCouplingNet, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout
        
        if gin_channels != 0:
            cond_layer = torch.nn.Conv1d(gin_channels, in_channels * n_layers, 1)
            self.cond_layer = torch.nn.utils.weight_norm(cond_layer, name="weight")
        
        conformer_block_config = ConformerBlockV1Config(
            ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                input_dim=self.in_channels,
                hidden_dim=self.hidden_channels,
                dropout=self.p_dropout,
                activation=nn.functional.silu
            ),
            mhsa_cfg=ConformerMHSAV1Config(
                input_dim=self.in_channels,
                num_att_heads=self.n_heads,
                att_weights_dropout=self.p_dropout,
                dropout=self.p_dropout
            ),
            conv_cfg=ConformerConvolutionV1Config(
                channels=self.in_channels,
                kernel_size=self.kernel_size,
                dropout=self.p_dropout,
                activation=nn.functional.silu,
                norm=LayerNormNC(self.in_channels)
            )
        )

        self.conformer = nn.ModuleList([ConformerBlockV1(conformer_block_config) for _ in range(self.n_layers)])

    def forward(self, x, x_mask, g=None, **kwargs):
        n_channels_tensor = torch.IntTensor([self.in_channels // 2])

        if g is not None:
            g = self.cond_layer(g)

        for i in range(self.n_layers):
            x_in = self.conformer[i](x.transpose(1,2), x_mask.squeeze(1)).transpose(1,2)

            if g is not None:
                cond_offset = i * self.in_channels
                g_l = g[:, cond_offset : cond_offset + self.in_channels, :]
            else:
                g_l = torch.zeros_like(x_in)
            # x = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            x = x_in + g_l

        return x * x_mask



class TDNN(nn.Module):
    def __init__(
        self, input_dim=23, output_dim=512, context_size=5, stride=1, dilation=1, batch_norm=False, dropout_p=0.2
    ):
        """
        TDNN as defined by https://www.danielpovey.com/files/2015_interspeech_multisplice.pdf

        Affine transformation not applied globally to all frames but smaller windows with local context

        batch_norm: True to include batch normalisation after the non linearity

        Context size and dilation determine the frames selected
        (although context size is not really defined in the traditional sense)
        For example:
            context size 5 and dilation 1 is equivalent to [-2,-1,0,1,2]
            context size 3 and dilation 2 is equivalent to [-2, 0, 2]
            context size 1 and dilation 1 is equivalent to [0]
        """
        super(TDNN, self).__init__()
        self.context_size = context_size
        self.stride = stride
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dilation = dilation
        self.dropout_p = dropout_p
        self.batch_norm = batch_norm

        self.kernel = nn.Conv1d(input_dim, output_dim, context_size, dilation=dilation, stride=stride)
        self.nonlinearity = nn.ReLU()
        if self.batch_norm:
            self.bn = nn.BatchNorm1d(output_dim)
        if self.dropout_p:
            self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        """
        input: size (batch, seq_len, input_features)
        outpu: size (batch, new_seq_len, output_features)
        """

        _, d, _ = x.shape
        assert d == self.input_dim, "Input dimension was wrong. Expected ({}), got ({})".format(self.input_dim, d)
        # x = x.unsqueeze(1)

        # N, output_dim*context_size, new_t = x.shape
        # breakpoint()
        x = self.kernel(x.float())
        x = self.nonlinearity(x)

        if self.batch_norm:
            # x = x.transpose(1,2)
            x = self.bn(x)
            # x = x.transpose(1,2)

        if self.dropout_p:
            x = self.drop(x)

        return x


class ActNorm(nn.Module):
    def __init__(self, channels, ddi=False, **kwargs):
        super().__init__()
        self.channels = channels
        self.initialized = not ddi

        self.logs = nn.Parameter(torch.zeros(1, channels, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        if x_mask is None:
            x_mask = torch.ones(x.size(0), 1, x.size(2)).to(device=x.device, dtype=x.dtype)
        x_len = torch.sum(x_mask, [1, 2])
        if not self.initialized:
            self.initialize(x, x_mask)
            self.initialized = True

        if reverse:
            z = (x - self.bias) * torch.exp(-self.logs) * x_mask
            logdet = None
        else:
            z = (self.bias + torch.exp(self.logs) * x) * x_mask
            logdet = torch.sum(self.logs) * x_len  # [b]

        return z, logdet

    def store_inverse(self):
        pass

    def set_ddi(self, ddi):
        self.initialized = not ddi

    def initialize(self, x, x_mask):
        with torch.no_grad():
            denom = torch.sum(x_mask, [0, 2])
            m = torch.sum(x * x_mask, [0, 2]) / denom
            m_sq = torch.sum(x * x * x_mask, [0, 2]) / denom
            v = m_sq - (m**2)
            logs = 0.5 * torch.log(torch.clamp_min(v, 1e-6))

            bias_init = (-m * torch.exp(-logs)).view(*self.bias.shape).to(dtype=self.bias.dtype)
            logs_init = (-logs).view(*self.logs.shape).to(dtype=self.logs.dtype)

            self.bias.data.copy_(bias_init)
            self.logs.data.copy_(logs_init)


class InvConvNear(nn.Module):
    def __init__(self, channels, n_split=4, no_jacobian=False, **kwargs):
        super().__init__()
        assert n_split % 2 == 0
        self.channels = channels
        self.n_split = n_split
        self.no_jacobian = no_jacobian

        w_init = torch.linalg.qr(torch.FloatTensor(self.n_split, self.n_split).normal_())[0]
        if torch.det(w_init) < 0:
            w_init[:, 0] = -1 * w_init[:, 0]
        self.weight = nn.Parameter(w_init)

    def forward(self, x, x_mask=None, reverse=False, **kwargs):
        b, c, t = x.size()
        assert c % self.n_split == 0
        if x_mask is None:
            x_mask = 1
            x_len = torch.ones((b,), dtype=x.dtype, device=x.device) * t
        else:
            x_len = torch.sum(x_mask, [1, 2])

        # group the channels into two halves for the splitting in the coupling layer and into (channels // n_split) groups (160 channels // 4 splits => 40 groups)
        x = x.view(b, 2, c // self.n_split, self.n_split // 2, t)
        # Mix the groups between the two halves of the vector to make sure that each half consists of parts from the original top and the bottom
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(b, self.n_split, c // self.n_split, t)

        if reverse:
            if hasattr(self, "weight_inv"):
                weight = self.weight_inv
            else:
                weight = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)
            logdet = None
        else:
            weight = self.weight
            if self.no_jacobian:
                logdet = 0
            else:
                logdet = torch.logdet(self.weight) * (c / self.n_split) * x_len  # [b]

        # n_split x n_split kernels are shared between the groups
        # bring the n_split x n_split kernel in the necessary shape for it to be used for 1x1 convolutions:
        weight = weight.view(self.n_split, self.n_split, 1, 1)
        z = F.conv2d(x, weight)

        # undo the permutation and grouping:
        z = z.view(b, 2, self.n_split // 2, c // self.n_split, t)
        z = z.permute(0, 1, 3, 2, 4).contiguous().view(b, c, t) * x_mask
        return z, logdet

    def store_inverse(self):
        self.weight_inv = torch.inverse(self.weight.float()).to(dtype=self.weight.dtype)


class FinalLinear(nn.Module):
    def __init__(self, in_size, hidden_size, target_size, n_layers=1, p_dropout=0.01):
        super().__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()

        self.layers.append(LinearBlock(self.in_size, self.hidden_size, self.p_dropout))

        for i in range(1, n_layers - 1):
            self.layers.append(LinearBlock(self.hidden_size, self.hidden_size, self.p_dropout))

        self.layers.append(LinearBlock(self.hidden_size, self.target_size, self.p_dropout))

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class FinalConvolutional(nn.Module):
    def __init__(
        self, in_channels, hidden_channels, target_channels, kernel_size=3, padding=1, n_layers=1, p_dropout=0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.target_channels = target_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.n_layers = n_layers
        self.p_dropout = p_dropout

        self.layers = nn.ModuleList()

        self.layers.append(Conv1DBlock(self.in_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout))

        for _ in range(1, n_layers - 1):
            self.layers.append(
                Conv1DBlock(self.hidden_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout)
            )

        self.layers.append(Conv1DBlock(self.in_channels, self.hidden_channels, self.kernel_size, p_dropout=p_dropout))

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Flow(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_blocks,
        n_layers,
        p_dropout=0.0,
        n_split=4,
        n_sqz=2,
        sigmoid_scale=False,
        gin_channels=0,
    ):
        """Flow-based decoder model

        Args:
            in_channels (int): Number of incoming channels
            hidden_channels (int): Number of hidden channels
            kernel_size (int): Kernel Size for convolutions in coupling blocks
            dilation_rate (float): Dilation Rate to define dilation in convolutions of coupling block
            n_blocks (int): Number of coupling blocks
            n_layers (int): Number of layers in CNN of the coupling blocks
            p_dropout (float, optional): Dropout probability for CNN in coupling blocks. Defaults to 0..
            n_split (int, optional): Number of splits for the 1x1 convolution for flows in the decoder. Defaults to 4.
            n_sqz (int, optional): Squeeze. Defaults to 1.
            sigmoid_scale (bool, optional): Boolean to define if log probs in coupling layers should be rescaled using sigmoid. Defaults to False.
            gin_channels (int, optional): Number of speaker embedding channels. Defaults to 0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.p_dropout = p_dropout
        self.n_split = n_split
        self.n_sqz = n_sqz
        self.sigmoid_scale = sigmoid_scale
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()

        for b in range(n_blocks):
            self.flows.append(ActNorm(channels=in_channels * n_sqz))
            self.flows.append(InvConvNear(channels=in_channels * n_sqz, n_split=n_split))
            self.flows.append(
                attentions.CouplingBlock(
                    in_channels * n_sqz,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation_rate=dilation_rate,
                    n_layers=n_layers,
                    gin_channels=gin_channels,
                    p_dropout=p_dropout,
                    sigmoid_scale=sigmoid_scale,
                )
            )

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            flows = self.flows
            logdet_tot = 0
        else:
            flows = reversed(self.flows)
            logdet_tot = None

        if self.n_sqz > 1:
            x, x_mask = commons.channel_squeeze(x, x_mask, self.n_sqz)
        for f in flows:
            if not reverse:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
                logdet_tot += logdet
            else:
                x, logdet = f(x, x_mask, g=g, reverse=reverse)
        if self.n_sqz > 1:
            x, x_mask = commons.channel_unsqueeze(x, x_mask, self.n_sqz)
        return x, logdet_tot

    def store_inverse(self):
        for f in self.flows:
            f.store_inverse()
