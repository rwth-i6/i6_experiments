import torch
from torch import nn
from typing import Optional, List, Tuple, Union
import math
from copy import deepcopy
from dataclasses import dataclass

from i6_models.util import compat

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1, LogMelFeatureExtractionV1Config, filters

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.parts.conformer import ConformerMHSAV1Config, ConformerPositionwiseFeedForwardV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerBlockV1Config
from i6_models.parts.conformer import (
    ConformerConvolutionV1Config,
    ConformerPositionwiseFeedForwardV1,
)
from i6_models.primitives.specaugment import specaugment_v1_by_length

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor, Joiner
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import SpecaugConfig

from ..auxil.functional import add_lookahead_v2, create_chunk_mask, Mode

from returnn.torch.context import get_run_ctx


class StreamableModule(nn.Module):
    """
    Abstract class for modules that operate differently in offline- and streaming inference mode
    """
    def __init__(self):
        super().__init__()
        self._mode = None

    def set_mode(self, mode: Mode) -> None:
        assert mode is not None, ""

        self._mode = mode

    def set_mode_cascaded(self, mode: Mode) -> None:
        assert mode is not None, ""
        
        if self._mode == mode:
            return

        self._mode = mode

        for m in self.modules():
            if isinstance(m, StreamableModule):
                m.set_mode(mode)

    def forward(self, *args, **kwargs):
        assert self._mode is not None, ""

        if self._mode == Mode.STREAMING:
            return self.forward_streaming(*args, **kwargs)
        else:
            return self.forward_offline(*args, **kwargs)

    def forward_offline(self, *args, **kwargs):
        raise NotImplementedError("Implement offline forward pass")

    def forward_streaming(self, *args, **kwargs):
        raise NotImplementedError("Implement streaming forward pass")
    
    def infer(self, *args, **kwargs):
        raise NotImplementedError("Implement infer")
    

class StreamableFeatureExtractorV1(StreamableModule):
    def __init__(
            self, cfg: LogMelFeatureExtractionV1Config, specaug_cfg: SpecaugConfig, specaug_start_epoch: int
    ):
        super().__init__()

        self.logmel = LogMelFeatureExtractionV1(cfg)
        self.specaug_config = specaug_cfg
        self.specaug_start_epoch = specaug_start_epoch

    def num_samples_to_frames(self, num_samples: int):
        if self.logmel.center:
            return (num_samples // self.logmel.hop_length) + 1
        else:
            return ((num_samples - self.logmel.n_fft) // self.logmel.hop_length) + 1

    def prep_streaming_input(self, features: torch.Tensor, mask: torch.Tensor, chunk_sz: int):
        bsz = features.size(0)

        chunk_size_frames = self.num_samples_to_frames(num_samples=int(chunk_sz))
        # pad conformer time-dim to be able to chunk (by reshaping) below
        time_dim_pad = -features.size(1) % chunk_size_frames
        # [B, T, *] -> [B, T+time_dim_pad, *] = [B, T', *]
        features = nn.functional.pad(features, (0, 0, 0, time_dim_pad),
                                               "constant", 0)
        mask = nn.functional.pad(mask, (0, time_dim_pad), "constant", False)

        # separate chunks to signal the conformer that we are chunking input
        features = features.view(bsz, -1, chunk_size_frames,
                                         features.size(-1))  # [B, (T'/C), C, F] = [B, N, C, F]
        mask = mask.view(bsz, -1, chunk_size_frames)  # [B, N, C]

        return features, mask

    def forward_offline(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param raw_audio: [B, T', <?>] 
        :param raw_audio_len: <= T' (in samples)

        :return: [B, T, F] features
        """
        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.logmel(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        out = audio_features_masked_2
        mask = mask_tensor(out, audio_features_len)

        return out, mask
    
    def forward_streaming(self, raw_audio: torch.Tensor, length: torch.Tensor, chunk_sz: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        :return: [B, N, C, F], [B, N, C]
        """
        features, mask = self.forward_offline(raw_audio=raw_audio, raw_audio_len=length)
        out, mask = self.prep_streaming_input(features, mask, chunk_sz)

        return out, mask

    def infer(self, input, lengths, chunk_sz_frames):
        audio_features, audio_features_lengths = self.forward_offline(input, lengths)

        time_dim_pad = -audio_features.size(1) % chunk_sz_frames
        audio_features = torch.nn.functional.pad(audio_features, (0, 0, 0, time_dim_pad), "constant", 0)

        return audio_features, audio_features_lengths.sum(dim=-1)
    

class StreamableLayerNormV1(StreamableModule):
    def __init__(self, input_dim: torch.Tensor, dual_mode: bool = True):
        super().__init__()
        self.layernorm_off = nn.LayerNorm(input_dim)
        self.layernorm_on = nn.LayerNorm(input_dim) if dual_mode else self.layernorm_off

    def forward_offline(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_off(x)
    
    def forward_streaming(self, x: torch.Tensor) -> torch.Tensor:
        return self.layernorm_on(x)


class StreamableJoinerV1(StreamableModule):
    r"""Streamable RNN-T joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(
            self, input_dim: int, output_dim: int, activation: str = "relu", dropout: float = 0.0, dual_mode: bool = True
    ) -> None:
        super().__init__()
        self.joiner_off = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        )
        self.joiner_on = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout
        ) if dual_mode else self.joiner_off

    def forward_offline(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_off(source_encodings, source_lengths, target_encodings, target_lengths)
    
    def forward_streaming(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_on(source_encodings, source_lengths, target_encodings, target_lengths)


class StreamableConformerMHSAV1(StreamableModule):
    def __init__(self, cfg: ConformerMHSAV1Config, dual_mode: bool):
        super().__init__()

        # TODO: diff layernorm depending on context (add flag to config?)
        self.layernorm = StreamableLayerNormV1(cfg.input_dim, dual_mode=dual_mode)
        self.mhsa = nn.MultiheadAttention(
            cfg.input_dim, cfg.num_att_heads, dropout=cfg.att_weights_dropout, batch_first=True
        )
        self.dropout = cfg.dropout

    def forward_offline(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param input_tensor: [B, T, F]
        :param sequence_mask: [B, T]
        :param attn_mask: [T, T]

        :return: [B, T, F]
        """

        inv_sequence_mask = compat.logical_not(sequence_mask)
        inv_attn_mask = None
        if attn_mask is not None:
            inv_attn_mask = compat.logical_not(attn_mask)

        output_tensor = self.layernorm(input_tensor)  # [B, T, F]
        
        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, 
            key_padding_mask=inv_sequence_mask, attn_mask=inv_attn_mask, 
            need_weights=False
        )  # [B, T, F]

        output_tensor = nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B, T, F]

        return output_tensor

    def forward_streaming(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param input_tensor: [B, N, C, F]
        :param sequence_mask: [B, N, C]
        :param attn_mask: [N*C, N*C]

        :return: [B, N, C, F]
        """
        assert input_tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = input_tensor.shape

        # [B, N, C, F] -> [B, N*C, F]
        input_tensor = input_tensor.flatten(1, 2)
        sequence_mask = sequence_mask.flatten(1, 2)

        out = self.forward_offline(input_tensor, sequence_mask, attn_mask)

        out = out.view(bsz, num_chunks, chunk_sz, input_tensor.size(-1))

        return out

    def infer(
            self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:
        
        y = self.layernorm(x)
        q = y[-ext_chunk_sz:]  # [C+R, F']

        inv_seq_mask = ~seq_mask
        output_tensor, _ = self.mhsa(
            q, y, y, key_padding_mask=inv_seq_mask, need_weights=False
        )  # [C+R, F]
        x = output_tensor + x[-ext_chunk_sz:]  # [C+R, F]

        return x


class StreamableConformerConvolutionV1(StreamableModule):
    def __init__(self, model_cfg: ConformerConvolutionV1Config, dual_mode: bool):
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
        self.layer_norm = StreamableLayerNormV1(model_cfg.channels, dual_mode=dual_mode)
        self.norm = deepcopy(model_cfg.norm)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation

    def forward_offline(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B, T, F]

        :return: torch.Tensor of shape [B, T, F]
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

    def forward_streaming(self, tensor: torch.Tensor, lookahead_sz: int, carry_over_size: int) -> torch.Tensor:
        """
        :param tensor: [B, N, C, F]
        :param lookahead_sz: number of future frames in chunk

        :return: [B, N, C, F]
        """
        assert tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = tensor.shape
        kernel_radius = self.depthwise_conv.kernel_size[0] // 2

        conv_in = torch.zeros(
            bsz, num_chunks, kernel_radius + chunk_sz, tensor.size(-1),
            device=tensor.device
        )

        # conv convolves over multiple past chunks w/o their fac
        tensor = tensor.flatten(1, 2)  # [B, N*C, F]
        chunks_no_fac = tensor.unfold(
            1, chunk_sz-lookahead_sz, chunk_sz
        ).swapaxes(-2, -1)  # [B, N, C-R, F]

        for i in range(num_chunks):
            if i > 0:
                # how many past chunks needed for conv
                conv_carry = math.ceil(kernel_radius / (chunk_sz - lookahead_sz))
                # don't go over predefined carryover
                conv_carry = min(carry_over_size, conv_carry)
                carry_no_fac = chunks_no_fac[:, max(0, i-conv_carry): i].flatten(1, 2)
                carry_no_fac = carry_no_fac[:, :kernel_radius]

                conv_in[:, i, -chunk_sz-carry_no_fac.size(1):-chunk_sz] = carry_no_fac
                    
            t_step = i * chunk_sz
            # add chunk itself
            conv_in[:, i, -chunk_sz:] = tensor[:, t_step: t_step+chunk_sz]

        conv_in = conv_in.flatten(0, 1)  # [B*N, KRN//2 + C, F]

        out = self.forward_offline(conv_in)
        out = out[:, -chunk_sz:]  # [B*N, C, F]
        out = out.view(bsz, num_chunks, chunk_sz, -1)

        return out

    def infer(self, x: torch.Tensor, states: Optional[List[torch.Tensor]], chunk_sz: int, lookahead_sz: int) -> torch.Tensor:
        if states is not None:
            states_no_fac = [layer_out[:-lookahead_sz] for layer_out in states]
            x = torch.cat((*states_no_fac, x), dim=0).unsqueeze(0)
            x = self.forward_offline(x)[:, -chunk_sz:]  # [1, C+R, F]
        else:
            x = x.unsqueeze(0)
            x = self.forward_offline(x)  # [1, C+R, F]

        return x.squeeze(0)


class StreamableConformerBlockV1(StreamableModule):
    def __init__(self, cfg: ConformerBlockV1Config, dual_mode: bool):
        super().__init__()

        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = StreamableConformerMHSAV1(cfg=cfg.mhsa_cfg, dual_mode=dual_mode)
        self.conv = StreamableConformerConvolutionV1(model_cfg=cfg.conv_cfg, dual_mode=dual_mode)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = StreamableLayerNormV1(cfg.ff_cfg.input_dim, dual_mode=dual_mode)

    def forward_offline(self, x: torch.Tensor, /, sequence_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        T = N*C

        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """

        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  # [B, T, F]
        x = self.conv(x) + x  # [B, T, F]
        x = 0.5 * self.ff2(x) + x  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]

        return x

    def forward_streaming(
            self, x: torch.Tensor, /, sequence_mask: torch.Tensor, 
            attn_mask: torch.Tensor, lookahead_size: int, carry_over_size: int,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, N, C, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, N, C]
        :param attn_mask: attention mask [N*C, N*C]
        :param lookahead_size: number of future frames in chunk

        :return: torch.Tensor of shape [B, N, C, F]
        """
        assert x.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = x.shape

        x = 0.5 * self.ff1(x) + x  # [B, N, C, F]
        x = self.mhsa(x, sequence_mask, attn_mask) + x  # [B, N, C, F]

        x = x.masked_fill((~sequence_mask.unsqueeze(-1)), 0.0)
        x = self.conv(x, lookahead_size, carry_over_size) + x

        x = 0.5 * self.ff2(x) + x  # [B, T, F] or [B*N, C, F]
        x = self.final_layer_norm(x)  # [B, T, F] or [B*N, C, F]

        x = x.reshape(bsz, num_chunks, chunk_sz, x.size(-1))  # [B, N, C, F]

        return x

    def infer(
            self,
            input: torch.Tensor,
            sequence_mask: torch.Tensor,
            states: Optional[List[torch.Tensor]],
            curr_layer: Optional[torch.Tensor],
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed [C+R, F']
            where C = chunk_size, R = lookahead_size (, K = carry_over_size)
        sequence_mask: [1, C+R]
        states: List[Tensor[C+R, F']] corresponding to previous chunk output of lower layer
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            all_curr_chunks = torch.cat((*states, input), dim=0)  # [(K+1)*(C+R), F'] = [t, F']

            seq_mask = torch.ones(
                all_curr_chunks.size(0), device=input.device, dtype=bool
            ).view(-1, ext_chunk_sz)  # (K+1, C+R)

            if lookahead_size > 0:
                seq_mask[:-1, -lookahead_size:] = False
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.flatten()  # [t]

        else:
            all_curr_chunks = input  # [C+R, F']
            seq_mask = sequence_mask.flatten()  # [C+R]

        #
        # block forward computation
        #
        x = 0.5 * self.ff1(all_curr_chunks) + all_curr_chunks  # [t, F']

        x = self.mhsa.infer(x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz)

        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)  # [C+R, F']
        x = self.conv.infer(
            x, states=curr_layer, chunk_sz=ext_chunk_sz, lookahead_sz=lookahead_size
        ) + x

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


@dataclass
class StreamableConformerEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int
    dual_mode: bool

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockV1Config


class StreamableConformerEncoderV1(StreamableModule):
    def __init__(self, cfg: StreamableConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = nn.ModuleList(
            [StreamableConformerBlockV1(cfg.block_cfg, dual_mode=cfg.dual_mode) for _ in range(cfg.num_layers)]
        )

    def forward_offline(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
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
    
    def forward_streaming(self, 
            data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int, carry_over_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        B: batch size, N: number of chunks = T/C, C: chunk size, F: feature dim, F': internal and output feature dim,
        T': data time dim, T: down-sampled time dim (internal time dim)

        :param data_tensor: input tensor, shape: [B, N, C, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 padding, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead frames chunk is able to attend to

        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]
        """
        assert data_tensor.dim() == 4, ""

        batch_sz, num_chunks, _, _ = data_tensor.shape
        attn_mask = None

        data_tensor = data_tensor.flatten(0, 1)  # [B*N, C, F]
        sequence_mask = sequence_mask.flatten(0, 1)
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)

        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        # [B, N, C', F'] -> [B, N, C'+R, F']
        x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

        attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                        chunk_size=x.size(2) - lookahead_size,
                                        lookahead_size=lookahead_size,
                                        carry_over_size=carry_over_size,
                                        device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask=attn_mask, 
                       lookahead_size=lookahead_size, carry_over_size=carry_over_size)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if lookahead_size > 0:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        return x, sequence_mask  # [B, N, C', F']

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: Optional[int] = None,
            lookahead_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        P = num. of future chunks

        :param input: audio frames [P, C, F], assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: [1,] true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param chunk_size: ...
        :param lookahead_size: number of lookahead frames chunk is able to attend to
        """
        if self._mode != Mode.STREAMING:
            self.set_mode_cascaded(Mode.STREAMING)

        print(f"{input.shape = }, {lengths.shape = }")
        # [P, C] where first P is current chunk and rest is for future ac ctx.
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, P*C)

        input = input.view(-1, chunk_size, input.size(-1))  # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size)  # (P, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (P, C', F')

        if lookahead_size > 0:
            chunk = x[0]  # [C', F']
            chunk_seq_mask = sequence_mask[0]  # [C',]

            future_ac_ctx = x[1:]  # [P-1, C', F']
            fut_seq_mask = sequence_mask[1:]  # [P-1, C']

            future_ac_ctx = future_ac_ctx.view(-1, x.size(-1))  # [t, F']
            fut_seq_mask = fut_seq_mask.view(-1)  # [t,]

            future_ac_ctx = future_ac_ctx[:lookahead_size]  # [R, F']
            fut_seq_mask = fut_seq_mask[:lookahead_size]  # [R,]

            x = torch.cat((chunk, future_ac_ctx), dim=0)  # [C'+R, F']
            sequence_mask = torch.cat((chunk_seq_mask, fut_seq_mask), dim=0).unsqueeze(0)  # [1, C+R]
        else:
            x = x[0]

        # save layer outs for next chunk (state)
        layer_outs = [x]
        prev_layer = curr_layer = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                prev_layer = [prev_chunk[-1][i] for prev_chunk in states]
                curr_layer = [prev_chunk[-1][i+1] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=prev_layer, curr_layer=curr_layer, lookahead_size=lookahead_size)
            layer_outs.append(x)

        # remove fac if any
        if lookahead_size > 0:
            x = x[:-lookahead_size]  # [C', F']
            sequence_mask = sequence_mask[:, :-lookahead_size]  # [1, C']

        x = x.unsqueeze(0)  # [1, C', F']

        return x, sequence_mask, layer_outs
