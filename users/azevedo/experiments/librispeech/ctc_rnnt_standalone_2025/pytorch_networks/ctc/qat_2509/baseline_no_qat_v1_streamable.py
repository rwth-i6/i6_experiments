"""
no QAT baseline
streamable version which allows for streaming training
"""

import math

import numpy as np
import torch
from torch import nn
import copy
from typing import Tuple, Optional, List

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .baseline_no_qat_v1_streamable_cfg import (
    ModelTrainNoQuantConfigV1,
    ConformerPositionwiseFeedForwardNoQuantV1Config,
    MultiheadAttentionNoQuantV1Config,
    ConformerConvolutionNoQuantV1Config,
    ConformerBlockNoQuantV1Config,
    ConformerEncoderNoQuantV1Config,
    StreamableFeatureExtractorV1Config
)
from .baseline_no_qat_v1_modules_streamable import MultiheadAttentionNoQuantStreamable

from ...streamable_module import StreamableModule
from ...encoders.components.frontend.streamable_vgg_act import StreamableVGG4LayerActFrontendV1
from ...encoders.components.feature_extractor.streamable_feature_extractor_v1 import StreamableFeatureExtractorV1
from ...common import Mode, create_chunk_mask, add_lookahead

from .._base_streamable_ctc import StreamableCTC as Model
from ...trainers import train_handler
from ..train_step_mode import CTCTrainStepMode



class ConformerPositionwiseFeedForwardNoQuant(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardNoQuantV1Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff = nn.Linear(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            bias=True,
        )
        self.activation = cfg.activation
        self.linear_out = nn.Linear(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            bias=True,
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


###############################################################################################################
# NOTE: now streamable
class ConformerMHSANoQuantStreamable(StreamableModule):
    """
    Conformer multi-headed self-attention module with streamable mhsa

    Note: Needs to be a StreamableModule so that the mode of MultiHeadAttentionNoQuantStreamable is set appropriately.
    """

    def __init__(self, cfg: MultiheadAttentionNoQuantV1Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = MultiheadAttentionNoQuantStreamable(cfg=cfg)
        self.dropout = cfg.dropout

    def forward_offline(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F) or (B, N, C, F) if we come from self.forward_streaming
        :param sequence_mask: bool mask of shape (B, T) or (B, N, C), True signals within sequence, False outside, will be inverted
        which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        inv_attn_mask = None if attn_mask is None else compat.logical_not(attn_mask)

        output_tensor = self.layernorm(input_tensor)  # [B,T,F] or [B,N,C,F] (but we only do layernorm across last dim so its fine)

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, sequence_mask=inv_sequence_mask, attn_mask=inv_attn_mask
        )  # [B,T,F] or [B,N,C,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor
    
    def forward_streaming(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param input_tensor: (B, N, C, F)
        :param sequence_mask: (B, N, C)
        :return: (B, N, C, F)
        """
        return self.forward_offline(input_tensor, sequence_mask, attn_mask)
    
    def infer(
            self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:
        """
        :param x: chunk, carryover and future frames, with shape [t, F]
        :param seq_mask:
        :param ext_chunk_sz: number of chunk frames and future frames = C+R
        :return: chunk with shape [C+R, F] where each feature attended to the carryover and future context
        """
        # x.shape: [t, F]
        attn_mask = torch.ones(x.size(0), x.size(0), device=x.device, dtype=torch.bool)
        y = self.forward_offline(
            input_tensor=x.unsqueeze(0), sequence_mask=seq_mask.unsqueeze(0), attn_mask=attn_mask)
        
        return y[0, -ext_chunk_sz:]  # [C+R, F]


###############################################################################################################
# NOTE: now streamable
class ConformerConvolutionNoQuantStreamable(StreamableModule):
    """
    Conformer convolution module with support for streaming training and inference.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionNoQuantV1Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.model_cfg = model_cfg
        self.pointwise_conv1 = nn.Linear(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            bias=True,
        )
        self.depthwise_conv = nn.Conv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
            bias=True,
            stride=1,
            dilation=1,
        )
        self.pointwise_conv2 = nn.Linear(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            bias=True,
        )
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = copy.deepcopy(model_cfg.norm)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward_offline(self, tensor: torch.Tensor) -> torch.Tensor:
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
    
        
    def forward_streaming(self, tensor:torch.Tensor, lookahead_size: int, carry_over_size: int):
        """
        Transform chunks into "carryover + chunk" and call self.forward_offline to compute causal convolution.

        :param tensor: [B, N, C+R, F] = [batch_size, number_of_chunks, chunk_size+future_context_size, conformer_dim]
        :param lookahead_size: the future acoustic context size, i.e. number of future frames per chunk = R
        :param carry_over_size: number of past chunks we may convolve over
        :return: [B, N, C+R, F]

        B: batch size
        N: number of chunks
        C+R: chunk size (+ future acoustic context size), i.e. the "extended" chunk size
        F: feature dimension
        """
        assert tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = tensor.shape
        kernel_radius = self.depthwise_conv.kernel_size[0] // 2  # = KRN//2

        # tensor to be filled and passed to forward_offline
        conv_in = torch.zeros(
            bsz, num_chunks, kernel_radius + chunk_sz, tensor.size(-1),
            device=tensor.device
        )

        # we remove future-acoustic-context (fac) as conv convolves over multiple past chunks w/o their fac
        # FIXME: why didnt i just do:
        # tensor = tensor[:, :, :-lookahead_size].contiguous()
        tensor = tensor.flatten(1, 2)  # [B, N*(C+R), F]
        chunks_no_fac = tensor.unfold(
            1, chunk_sz - lookahead_size, chunk_sz
        ).swapaxes(-2, -1)  # [B, N, C, F]

        for i in range(num_chunks):
            if i > 0:
                # calc how many past chunks needed for conv
                conv_carry = math.ceil(kernel_radius / (chunk_sz - lookahead_size))
                # don't go over predefined carryover
                conv_carry = min(carry_over_size, conv_carry)
                carry_no_fac = chunks_no_fac[:, max(0, i - conv_carry): i].flatten(1, 2)
                carry_no_fac = carry_no_fac[:, :kernel_radius]

                conv_in[:, i, -chunk_sz - carry_no_fac.size(1):-chunk_sz] = carry_no_fac

            t_step = i * chunk_sz
            # add chunk itself
            conv_in[:, i, -chunk_sz:] = tensor[:, t_step: t_step + chunk_sz]

        conv_in = conv_in.flatten(0, 1)  # [B*N, KRN//2 + C+R, F]  (KRN is the kernel_size)

        out = self.forward_offline(conv_in)
        out = out[:, -chunk_sz:]  # remove kernel_radius and get [B*N, C+R, F]
        out = out.view(bsz, num_chunks, chunk_sz, -1)  # [B, N, C+R, F]

        return out

    def infer(
            self, x: torch.Tensor, states: Optional[List[torch.Tensor]], chunk_sz: int, lookahead_sz: int
    ) -> torch.Tensor:
        """
        :param x: the current chunk
        :param states: cached previous chunks of current layer (carryover)
        :param chunk_sz:
        :param lookahead_sz:
        """
        if states is not None:
            states_no_fac = [layer_out[:-lookahead_sz] for layer_out in states]  # remove future-acoustic-context and build carryover
            x = torch.cat((*states_no_fac, x), dim=0).unsqueeze(0)  # combine carryover and chunk like in self.forward_streaming
            x = self.forward_offline(x)[:, -chunk_sz:]  # [1, C+R, F]
        else:
            x = x.unsqueeze(0)
            x = self.forward_offline(x)  # [1, C+R, F]

        return x.squeeze(0)


###############################################################################################################
# NOTE: now streamable
class ConformerBlockNoQuantStreamable(StreamableModule):
    """
    Streamable Conformer block module
    """

    def __init__(self, cfg: ConformerBlockNoQuantV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardNoQuant(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSANoQuantStreamable(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionNoQuantStreamable(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardNoQuant(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward_offline(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
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
    
    def forward_streaming(
        self, x: torch.Tensor, /, sequence_mask: torch.Tensor, attn_mask: torch.Tensor, 
        lookahead_size: int, carry_over_size: int
    ):
        """
        :param x: [B, N, C+R, F']
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside (e.g. padding), shape: [B, N, C+R]
        :param attn_mask: expecting a causal mask that prevents chunks from attending to future chunks with shape [N*(C+R), N*(C+R)]
        :param lookahead_size: number of future frames per chunk (R)
        :param carry_over_size: number of past chunks we may depend on per block (i.e. attend to or convolve over)
        :return: [B, N, C+R, F']
        """
        assert x.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = x.shape

        x = 0.5 * self.ff1(x) + x  # [B, N, C+R, F']
        x = self.mhsa(x, sequence_mask, attn_mask) + x  # [B, N, C+R, F']

        x = x.masked_fill((~sequence_mask.unsqueeze(-1)), 0.0)  # convolution of previous layer might have overwritten 0-padding
        x = self.conv(x, lookahead_size, carry_over_size) + x

        x = 0.5 * self.ff2(x) + x  # [B, N, C+R, F']
        x = self.final_layer_norm(x)  # [B, N, C+R, F']

        # FIXME: unnecessary reshape
        x = x.reshape(bsz, num_chunks, chunk_sz, x.size(-1))  # [B, N, C+R, F']

        return x

    def infer(
            self,
            input: torch.Tensor,
            sequence_mask: torch.Tensor,
            states: Optional[List[torch.Tensor]],
            curr_layer: Optional[torch.Tensor],
            lookahead_size: int,
    ) -> torch.Tensor:
        """
        Compute encoder block outputs based on previously cached chunks (states) and current chunk of subsampled features.

        :param input: chunk outputs with shape [C+R, F'], where C, R are the chunk size and the lookahead size in #frames respectively
        :param sequence_mask:
        :param states: encoder block outputs of the previous chunk at the previous layer (the carryover for mhsa)
        :param curr_layer: encoder block outputs of the previous chunk at the current layer (needed for convolution)
        :param lookahead_size: R
        :return: encoder block outputs of the current chunk [C+R, F']
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            # combine carryover and chunk: [C+R, F'] -> [(K+1)*(C+R), F'] = [t, F'] where K is the carryover size
            all_curr_chunks = torch.cat((*states, input), dim=0) 

            # build sequence_mask for mhsa
            seq_mask = torch.ones(all_curr_chunks.size(0), device=input.device, dtype=bool).view(-1, ext_chunk_sz)  # (K+1, C+R)
            if lookahead_size > 0:
                seq_mask[:-1, -lookahead_size:] = False  # we want to ignore all fac except the one of current chunk
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.flatten()  # [t]
        else:
            all_curr_chunks = input  # [C+R, F']
            seq_mask = sequence_mask.flatten()  # [C+R]

        #
        # block forward computation
        #
        x = 0.5 * self.ff1(all_curr_chunks) + all_curr_chunks  # [t, F']

        x = self.mhsa.infer(x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz) + x[-ext_chunk_sz:]  # [C+R, F']

        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)
        x = self.conv.infer(x, states=curr_layer, chunk_sz=ext_chunk_sz, lookahead_sz=lookahead_size) + x  # [C+R, F']

        x = 0.5 * self.ff2(x) + x  # [C+R, F']
        x = self.final_layer_norm(x)  # [C+R, F']

        return x    


###############################################################################################################
# NOTE: now streamable
class ConformerEncoderNoQuantStreamable(StreamableModule):
    """
    Implementation of a streamable Conformer. 
    Derived from the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks and allows for streaming training and decoding.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderNoQuantV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend: StreamableVGG4LayerActFrontendV1 = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockNoQuantStreamable(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward_offline(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def forward_streaming(
            self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor,
            chunk_size: int, lookahead_size: int, carry_over_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, N, C', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside (e.g. padding), shape: [B, N, C']
        :param chunk_size:
        :param lookahead_size: number of future frames per chunk (R)
        :param carry_over_size: number of past chunks we may depend on per block (i.e. attend to or convolve over)
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, C' * N, F'],
            out_seq_mask is a torch.Tensor of shape [B, C' * N]

        
        F: input feature dim, F': internal and output feature dim
        C': data chunk size, C: down-sampled chunk size (internal chunk size)
        N: number of chunks per sequence
        R: number of future subsampled frames each chunk gets appended
        """
        # data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len, chunk_size)

        batch_sz, num_chunks, _, _ = data_tensor.shape

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, N, C', F] -> [B, N, C, F']

        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        # [B, N, C, F'] -> [B, N, C+R, F']
        x, sequence_mask = add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

        attn_mask = create_chunk_mask(
            seq_len=(x.size(1) * x.size(2)),
            chunk_size=x.size(2) - lookahead_size,
            lookahead_size=lookahead_size,
            carry_over_size=carry_over_size,
            device=x.device
        )

        for module in self.module_list:
            x = module(
                x, sequence_mask, attn_mask=attn_mask,
                lookahead_size=lookahead_size, carry_over_size=carry_over_size
            )  # [B, N, C+R, F']

        # remove lookahead frames from every chunk
        if lookahead_size > 0:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        x = x.flatten(1, 2)  # [B, C*N, F']
        sequence_mask = sequence_mask.flatten(1, 2)  # [B, C*N]

        return x, sequence_mask

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: [1, P*C', F], where P is the number of future chunks we need for the future frames of current chunk
        :param lengths: the number of non-padding frames [1,]
        :param states: list of encoder block outputs of previous chunks (each output having shape [C, F'])
        :param chunk_size: C'
        :param lookahead_size: R
        :return: encoder outputs of the current chunk, number of (non-padding) encoder outputs, intermediate encoder block outputs
        """
        assert self._mode == Mode.STREAMING, "Expected encoder to be in streaming mode for streaming inference."

        # chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
        # audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)
        chunk_size_frames = chunk_size
        audio_features, audio_features_len = input, lengths

        # [1, P*C', F] -> [P, C, F']
        x, sequence_mask = self.frontend.infer(audio_features, audio_features_len, chunk_size_frames)

        # add future acoustic context to the current chunk
        if lookahead_size > 0:
            chunk = x[0]  # the current chunk whose encoder outputs we want to compute with shape [C, F']
            chunk_seq_mask = sequence_mask[0]  # [C]

            future_ac_ctx = x[1:]  # the future acoustic context [P-1, C, F']
            fut_seq_mask = sequence_mask[1:]  # [P-1, C]

            # combine all future chunks and extract the `lookahead_size` future frames
            future_ac_ctx = future_ac_ctx.view(-1, x.size(-1))  # [(P-1)*C, F'] =: [t, F']
            fut_seq_mask = fut_seq_mask.view(-1)  # [t,]
            future_ac_ctx = future_ac_ctx[:lookahead_size]  # [R, F']
            fut_seq_mask = fut_seq_mask[:lookahead_size]  # [R,]

            # combine current chunk and its future frames
            x = torch.cat((chunk, future_ac_ctx), dim=0)  # [C+R, F']
            sequence_mask = torch.cat((chunk_seq_mask, fut_seq_mask), dim=0).unsqueeze(0)  # [1, C+R]
        else:
            x = x[0]

        # save layer outs for next chunk (state for next chunk)
        layer_outs = [x]
        prev_layer = curr_layer = None

        for i, module in enumerate(self.encoder_blocks):
            if states is not None:
                # first chunk is not provided with any previous states
                prev_layer = [prev_chunk[-1][i] for prev_chunk in states]
                curr_layer = [prev_chunk[-1][i + 1] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=prev_layer, curr_layer=curr_layer, lookahead_size=lookahead_size)
            layer_outs.append(x)

        # remove fac if any
        if lookahead_size > 0:
            x = x[:-lookahead_size]  # [C, F']
            sequence_mask = sequence_mask[:, :-lookahead_size]  # [1, C]

        x = x.unsqueeze(0)  # [1, C, F']

        return x, torch.sum(sequence_mask, dim=1), layer_outs


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


###############################################################################################################
# NOTE: now streamable
class Model(StreamableModule):
    def __init__(self, model_config_dict, **kwargs):
        epoch = kwargs.pop("epoch")
        step = kwargs.pop("step")
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"

        super().__init__()
        self.train_config = ModelTrainNoQuantConfigV1.from_dict(model_config_dict)
        # fe_config = self.train_config.feature_extraction_config
        fe_config = StreamableFeatureExtractorV1Config(
            logmel_cfg=self.train_config.feature_extraction_config, 
            specaug_cfg=self.train_config.specaug_config, specaug_start_epoch=self.train_config.specauc_start_epoch
        )
        frontend_config = self.train_config.frontend_config
        
        conformer_size = self.train_config.conformer_size
        # self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)

        self.feature_extraction = StreamableFeatureExtractorV1(cfg=fe_config)
        conformer_config = ConformerEncoderNoQuantV1Config(
            num_layers=self.train_config.num_layers,
            frontend=ModuleFactoryV1(module_class=StreamableVGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockNoQuantV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardNoQuantV1Config(
                    input_dim=conformer_size,
                    hidden_dim=self.train_config.ff_dim,
                    dropout=self.train_config.ff_dropout,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=MultiheadAttentionNoQuantV1Config(
                    input_dim=conformer_size,
                    num_att_heads=self.train_config.num_heads,
                    att_weights_dropout=self.train_config.att_weights_dropout,
                    dropout=self.train_config.mhsa_dropout,
                ),
                conv_cfg=ConformerConvolutionNoQuantV1Config(
                    channels=conformer_size,
                    kernel_size=self.train_config.conv_kernel_size,
                    dropout=self.train_config.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                ),
            ),
        )
        self.conformer = ConformerEncoderNoQuantStreamable(cfg=conformer_config)

        self.final_linear = nn.Linear(conformer_size, self.train_config.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.train_config.final_dropout)
        self.specaug_start_epoch = self.train_config.specauc_start_epoch

        # streaming relevant params
        self.chunk_size = self.train_config.chunk_size
        self.lookahead_size = self.train_config.lookahead_size
        self.carry_over_size = self.train_config.carry_over_size

        self.cfg = self.train_config  # FIXME: need this for train_step, specifically CTCTrainStepMode

    def forward_offline(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len)
        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)  # FIXME: ctc_refactored was better w/o final_dropout
        logits = self.final_linear(conformer_out)

        # log_probs = torch.log_softmax(logits, dim=2)
        # return log_probs, torch.sum(out_mask, dim=1)

        return logits, torch.sum(out_mask, dim=1)

    def forward_streaming(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T', #labels + blank]
        """
        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len, self.chunk_size)  # [B, N, C', F]
        conformer_out, out_mask = self.conformer(
            conformer_in, mask, chunk_size=self.chunk_size, lookahead_size=self.lookahead_size, carry_over_size=self.carry_over_size,
        )  # [B, C*N, F'] = [B, T', F']
        conformer_out = self.final_dropout(conformer_out)
        logits = self.final_linear(conformer_out)
        
        # log_probs = torch.log_softmax(logits, dim=2)
        # return log_probs, torch.sum(out_mask, dim=1) 

        return logits, torch.sum(out_mask, dim=1)

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio samples as [B=1, T, 1] where T includes future frames
        :param lengths: length of T as [B=1]
        :param states:
        :param chunk_size:
        :param lookahead_size:
        """
        assert chunk_size is not None and lookahead_size is not None
        assert input.dim() == 3 and input.size(0) == 1, "Streaming inference expects input with shape [B=1, T, 1]."

        with torch.no_grad():
            chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
            conformer_in, mask = self.feature_extraction.infer(input, lengths, chunk_size_frames)
            encoder_out, encoder_out_lengths, state = self.conformer.infer(
                conformer_in, mask, states, chunk_size=chunk_size, lookahead_size=lookahead_size
            )
            encoder_out = self.final_linear(encoder_out)
        
        # return encoder_out[:, :encoder_out_lengths[0]], encoder_out_lengths, [state]
        return encoder_out, encoder_out_lengths, [state]



def train_step(*, model: Model, data, run_ctx, **kwargs):
    train_strat: train_handler.TrainingStrategy = None
    train_step_mode = CTCTrainStepMode()
    match model.cfg.train_mode:
        case train_handler.TrainMode.UNIFIED:
            train_strat = train_handler.TrainUnified(model, train_step_mode, streaming_scale=model.cfg.streaming_scale)
        case train_handler.TrainMode.SWITCHING:
            train_strat = train_handler.TrainSwitching(model, train_step_mode, run_ctx=run_ctx)
        case train_handler.TrainMode.STREAMING:
            train_strat = train_handler.TrainStreaming(model, train_step_mode)
        case train_handler.TrainMode.OFFLINE:
            train_strat = train_handler.TrainOffline(model, train_step_mode)
        case _:
            raise NotImplementedError("Training Strategy not available.")

    loss_dict, num_phonemes = train_strat.step(data)

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key,
            loss=loss_dict[loss_key]["loss"],
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
        )


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

    if model.cfg.train_mode == train_handler.TrainMode.OFFLINE:
        model.set_mode_cascaded(Mode.OFFLINE)
    elif model.cfg.train_mode == train_handler.TrainMode.STREAMING:
        model.set_mode_cascaded(Mode.STREAMING)
    elif model.cfg.train_mode == train_handler.TrainMode.SWITCHING:
        model.set_mode_cascaded(Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE)
    else:
        raise NotImplementedError

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    model.unset_mode_cascaded()

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))


