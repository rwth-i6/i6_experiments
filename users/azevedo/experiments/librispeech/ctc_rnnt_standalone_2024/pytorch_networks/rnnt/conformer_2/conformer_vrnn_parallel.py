import torch
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import math

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1Config

from i6_models.util import compat
from i6_models.config import ModelConfiguration

from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerPositionwiseFeedForwardV1,

    ConformerMHSAV1Config,
)
from .conformer_v2 import mask_tensor
from .conformer_v3 import create_chunk_mask

from torch.utils.checkpoint import checkpoint_sequential

from .mhav3 import MultiheadUnifiedAttention


class Mode(Enum):
    ONLINE = 0
    OFFLINE = 1


@dataclass
class VRNNV1Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    dropout: float
    activation: Callable[[torch.Tensor], torch.Tensor] = torch.nn.functional.relu
    

@dataclass
class ConformerEncoderV3Config(ConformerEncoderV1Config):
    vrnn_config: VRNNV1Config


class ConformerMHSAV3(torch.nn.Module):
    def __init__(self, cfg: ConformerMHSAV1Config):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = MultiheadUnifiedAttention(
            cfg.input_dim, cfg.num_att_heads, dropout=cfg.att_weights_dropout, batch_first=True
        )
        self.dropout = cfg.dropout

    def forward(
        self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        
        inv_sequence_mask = compat.logical_not(sequence_mask)
        inv_attn_mask = None
        if attn_mask is not None:
            inv_attn_mask = compat.logical_not(attn_mask)

        output_tensor = self.layernorm(input_tensor)  # [B,T,F]
        
        if self.train():
            (output_tensor, output_tensor_offline), _ = self.mhsa(
                output_tensor, output_tensor, output_tensor, 
                key_padding_mask=inv_sequence_mask, attn_mask=inv_attn_mask, 
                need_weights=True, need_keypad_out=True
            )  # [B,T,F]
        else:
            output_tensor, _ = self.mhsa(
                output_tensor, output_tensor, output_tensor, 
                key_padding_mask=inv_sequence_mask, attn_mask=inv_attn_mask, 
                need_weights=False
            )  # [B,T,F]
        
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]
        if not self.train():
            return output_tensor

        output_tensor_offline = torch.nn.functional.dropout(output_tensor_offline, p=self.dropout, training=self.training)  # [B,T,F]

        output_tensor = torch.cat(
            (output_tensor, output_tensor_offline), dim=0
        ).view(2, *output_tensor.shape)  # [2, B, T, F]

        return output_tensor
    

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[List[torch.Tensor]],
        lookahead_size: int = 0,
        mode: Mode = Mode.ONLINE,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        pass


class ConformerBlockHeadV1(torch.nn.Module):
    def __init__(self, ff_cfg, mhsa_cfg):
        super().__init__()

        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=ff_cfg)
        self.mhsa = ConformerMHSAV3(cfg=mhsa_cfg)

    def forward(
        self,
        x: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, N, C+R, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T] or [B, N, C]
        :param attn_mask: attention mask
        :return: (torch.Tensor, torch.Tensor) with sizes [B, T, F]
        """

        # TODO only works in train mode (dont use forward for inference => complete self.infer)
        assert (x.dim() == sequence_mask.dim() + 1 == 4)

        batch_size: int = x.size(0)

        # reshape [B, N, C+R, F] to [B, (C+R)*N, F] to use attn_mask for chunks
        x = x.view(batch_size, -1, x.size(-1))
        sequence_mask = sequence_mask.view(sequence_mask.size(0), -1)   # [B, (C+R)*N]

        x = 0.5 * self.ff1(x) + x  # [B, T, F], T = (C+R)*N

        on_out, off_out = self.mhsa(x, sequence_mask, attn_mask)  # ([B, T, F], [B, T, F])

        return on_out + x, off_out + x
    

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[torch.Tensor],
        lookahead_size: int = 0,
        mode: Mode = Mode.ONLINE,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (C+R, F')
            where C is the chunk_size and R the lookahead_size
        sequence_mask: (1, C+R)
        states: List[Tensor(C+R, F')] corresponding to previous chunk output of lower layer
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            all_curr_chunks = torch.cat((*states, input), dim=0)  # ((i+1)*(C+R), F') = (t, F')
            all_curr_chunks = all_curr_chunks.reshape(-1, all_curr_chunks.size(-1))  # (t, F')
            
            seq_mask = torch.zeros(ext_chunk_sz, sequence_mask.size(-1), device=input.device, dtype=bool)
            seq_mask[:-1, -lookahead_size:] = False  # dont attend to lookahead
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.view(-1)  # (t)
            
        else:
            all_curr_chunks = input  # (C+R, F')
            seq_mask = sequence_mask  # (1, C+R)


        x = 0.5 * self.ff1(input) + input  # [C+R, F']

        # multihead attention
        y = self.mhsa.layernorm(x)

        inv_sequence_mask = compat.logical_not(seq_mask)

        if mode == Mode.ONLINE:
            output_tensor, _ = self.mhsa.mhsa(
                y, all_curr_chunks, all_curr_chunks,
                key_padding_mask=inv_sequence_mask, need_weights=False
            )  # [C+R, F]

            return output_tensor + x  # [C+R, F]

        (on_out, off_out), _ = self.mhsa.mhsa(
            y, all_curr_chunks, all_curr_chunks,  
            key_padding_mask=inv_sequence_mask, need_weights=True, need_keypad_out=True
        )  # [T,F]


        return on_out + x, off_out + x
    

class ConformerBlockTailV1(torch.nn.Module):
    def __init__(self, conv_cfg, ff_cfg):
        super().__init__()

        self.conv = ConformerConvolutionV1(model_cfg=conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, sequence_mask: torch.Tensor, chunk_sz: Optional[int] = None) -> torch.Tensor:
        batch_size = x.size(0)
        chunked = chunk_sz is not None

        # convolution
        if chunked:
            # make convolution chunk-independent by reshaping [B, C(+R)*N, F] to [B*N, C(+R), F]
            x = x.reshape(-1, chunk_sz, x.size(-1))
            #sequence_mask = sequence_mask.reshape(-1, chunk_sz)

        sequence_mask = sequence_mask.view(*x.shape[:-1])
        # prevent convolution to write padding when using multiple layers
        x = x.masked_fill((~sequence_mask[:, :, None]), 0.0)
        x = self.conv(x) + x  # [B, T, F]

        x = 0.5 * self.ff2(x) + x  # [B, T, F] or [B*N, C(+R), F]
        x = self.final_layer_norm(x)  # [B, T, F] or [B*N, C(+R), F]

        if chunked:
            x = x.reshape(batch_size, -1, chunk_sz, x.size(-1))   # [B, N, C(+R), F]

        return x

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        ext_chunk_sz: int = 0,
    ):
        x = input

        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)
        x = self.conv(input) + input  # [C+R, F]

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


class VRNNV1(torch.nn.Module):
    def __init__(self, cfg: VRNNV1Config):
        super().__init__()

        # TODO just use rnn
        self.linear = torch.nn.Linear(in_features=cfg.input_dim, out_features=cfg.input_dim, bias=True)
        self.activation = cfg.activation
        self.dropout = cfg.dropout
        self.layer_norm = torch.nn.LayerNorm(cfg.input_dim)

    def forward(self, tensor: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :param state: shape [B,T,F]
        :return: shape [B,T,F], F=input_dim
        """
        x = self.linear(tensor + state)  # [B,T,F]
        x = self.activation(x)  # [B,T,F]
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)  # [B,T,F]
        x = self.layer_norm(x + tensor)
        return x
    

class ConformerVRNNEncoderV1(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV3Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.vrnn = VRNNV1(cfg.vrnn_config)

        self.block_heads = torch.nn.ModuleList([ConformerBlockHeadV1(
            cfg.block_cfg.ff_cfg,
            cfg.block_cfg.mhsa_cfg
        ) for _ in range(cfg.num_layers)])

        self.block_tails = torch.nn.ModuleList([ConformerBlockTailV1(
            conv_cfg=cfg.block_cfg.conv_cfg,
            ff_cfg=cfg.block_cfg.ff_cfg
        ) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, N, C+R, F] where C is the chunk size,
            N = T'/C the number of chunks, R the future acoustic context size (in #frames)
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead-frames chunk is able to attend to
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        batch_size: int = data_tensor.size(0)
        attn_mask: Optional[torch.Tensor] = None
        subs_lookahead_size = math.ceil(lookahead_size/6)  # div by 6 as we have sub_factor of 6

        #
        # frontend
        #
        # reshaping [B, N, C+R, F] to [B*N, C+R, F] for frontend (pooling + conv2d)
        data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
        sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B*N, (C+R)', F']
        ext_chunk_sz = x.size(1)

        #
        # conformer blocks
        #
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, (C+R)', F']
        sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, (C+R)']

        # create chunk-causal attn_mask
        # TODO: hyperparam carry_over (so far: carry_over = chunk_size)
        attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                      chunk_size=x.size(2)-subs_lookahead_size, 
                                      lookahead_size=subs_lookahead_size, device=x.device)

        #
        # joint computation
        #
        # initial v-rnn state
        y_state = x.clone().view(batch_size, -1, x.size(-1))  # [B, T, F']
        for i in range(len(self.block_heads)):
            x, y = self.block_heads[i](x, sequence_mask, attn_mask)

            x = self.block_tails[i](x, sequence_mask, chunk_sz=ext_chunk_sz)
            y = self.block_tails[i](y, sequence_mask)

            y_state = self.vrnn(tensor=y, state=y_state)  # [B, T, F']

        #
        # prepare output
        #
        def _merge_drop_fac(tensor_in):
            tensor_in = tensor_in.view(-1, ext_chunk_sz, tensor_in.size(-1))  # [B*N, C'+R', ...]
            tensor_in = tensor_in[:, :-subs_lookahead_size].reshape(batch_size, -1, tensor_in.size(-1))  # [B, C'*N, ...]
            return tensor_in
        
        seq_mask = _merge_drop_fac(sequence_mask.unsqueeze(-1)).squeeze(-1)  # [B, T']

        y = _merge_drop_fac(y_state)  # [B, T', F']
        offline_out = (y, seq_mask, Mode.OFFLINE)

        x = _merge_drop_fac(x)  # [B, T', F']
        online_out = (x, seq_mask, Mode.ONLINE)

        return online_out, offline_out

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        lookahead_size: Optional[int] = None,
        mode: Mode = Mode.ONLINE,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C+R, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
        :param lookahead_size: R = number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend,
        """
        # replace first frames of each chunk with future acoustic context
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, C+R)
        x, sequence_mask = self.frontend(input, sequence_mask)
        x = x.squeeze(0)  # ((C+R)', F')

        subs_lookahead_size = math.ceil(lookahead_size/6)
        ext_chunk_sz = x.size(0)

        # save layer outs for next chunk (state)
        layer_outs: List[torch.Tensor] = [x]
        state: Optional[torch.Tensor] = None

        y_state = None
        if mode == Mode.OFFLINE:
            y_state = x.clone().view(1, -1, x.size(-1))

        for i in range(len(self.block_heads)):
            if states is not None:
                state = states[-1][i]

            x, y = self.block_heads[i].infer(
                input=x,
                sequence_mask=sequence_mask,
                states=state,
                lookahead_size=subs_lookahead_size,
                mode=mode
            )

            x = self.block_tails[i].infer(
                input=x,
                sequence_mask=sequence_mask,
                ext_chunk_sz=ext_chunk_sz
            )

            if mode == Mode.OFFLINE:
                y = self.block_tails[i].infer(
                    input=y,
                    sequence_mask=sequence_mask,
                    ext_chunk_sz=0
                )
                y_state = self.vrnn(tensor=y, state=y_state)
            else:
                layer_outs.append(x)

        if mode == Mode.OFFLINE:
            x = y_state

        x = x.unsqueeze(0)  # (1, C', F')
        x = x[:, :-subs_lookahead_size].contiguous()

        return x, sequence_mask, layer_outs
