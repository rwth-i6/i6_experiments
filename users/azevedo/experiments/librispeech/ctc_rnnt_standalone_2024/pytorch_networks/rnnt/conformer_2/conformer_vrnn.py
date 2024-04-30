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
        self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
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
    
    def infer():
        pass



class ConformerBlockV5(torch.nn.Module):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV3(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionV1(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)


    def forward(
        self,
        x: torch.Tensor,
        /,
        sequence_mask: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F] or [B, N, C, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T] or [B, N, C]
        :param attn_mask: attention mask
        :return: torch.Tensor of shape [B, T, F] or [B, N, C, F]
        """

        # TODO only works in train mode (dont use forward for inference => complete self.infer)
        assert (x.dim() == sequence_mask.dim() + 1 == 4)

        chunk_size: int = x.size(-2)
        batch_size: int = x.size(0)


        # reshape [B, N, C+R, F] to [B, (C+R)*N, F] to use attn_mask for chunks
        x = x.view(batch_size, -1, x.size(-1))
        sequence_mask = sequence_mask.view(sequence_mask.size(0), -1)   # [B, (C+R)*N]

        x = 0.5 * self.ff1(x) + x  # [B, T, F], T = (C+R)*N


        #
        # multi-head self attention
        #
        uni = self.mhsa(x, sequence_mask, attn_mask)  # [2, B, T, F]
        x = uni + x.unsqueeze(0)

        # TODO: split paths here to make use of parallelization (create a new module and use ModuleList)
        # Alternatively: replace ConformerBlock with 2 modules:
        #   - ConformerBlockHead: does computation up to here and returns `uni`
        #   - ConformerBlockTail: does rest of computation
        #
        # convolution
        #
        x = x.flatten(0, 1)  # [2*B, T, F]
        sequence_mask = sequence_mask.repeat(2, 1)  # [2*B, T]

        # prevent convolution to overwrite padding when using multiple layers
        x = x.masked_fill((~sequence_mask[:, :, None]), 0)

        # chunk-indpendent conv
        chunk_conv_in = x[:batch_size].view(-1, chunk_size, x.size(-1))  # [B*N, C+R, F]
        y = self.conv(chunk_conv_in)
        y = y.view(1, batch_size, -1, chunk_size, y.size(-1))  # [1, B, N, C+R, F]

        # full convolution
        z = self.conv(x)  # [2*B, T, F]
        z = z.view(2, batch_size, -1, chunk_size, z.size(-1))  # [2, B, N, C+R, F]
        #print(f"{x.shape = }, {z.shape = }, {y.shape = }")

        # watch out that online chunks dont look over their future boundary
        kernel_sz = self.conv.depthwise_conv.kernel_size[0]
        z[0, :, :, :-(kernel_sz - 1)//2] = y[0, :, :, :-(kernel_sz - 1)//2]

        # remove overlap
        z = 2*y - z

        z = z.view_as(x)  # [2*B, T, F]
        x = z + x


        #
        # rest
        #
        x = 0.5 * self.ff2(x) + x  # [2, B, T, F]
        x = self.final_layer_norm(x)  # [2, B, T, F]

        x = x.view(2, batch_size, -1, chunk_size, x.size(-1))   # [2, B, N, C+R, F]
        
        return x


    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[torch.Tensor],
        lookahead_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (C+R, F')
            where C is the chunk_size and R the lookahead_size
        input should be parseable to forward without attn_mask
        states: (C+R, F') corresponding to previous chunk output of lower layer
        """
        if states is not None:
            block_in = torch.cat((states, input), dim=0)  # (2*(C+R), F')
            block_in = block_in.reshape(2, -1, block_in.size(-1)).unsqueeze(0)  # (1, 2, C+R, F') for block.foward()

            seq_mask = torch.cat(
                (torch.full_like(sequence_mask, True, device=input.device), sequence_mask),
                dim=0
            ).unsqueeze(0)  # (1, 2, C+R)
            # we dont condition on future frames of past chunk (like in training)
            seq_mask[:, 0, -lookahead_size:] = False
        else:
            block_in = input[None, None]  # (1, 1, C+R, F')
            seq_mask = sequence_mask.unsqueeze(0)  # (1, 1, C+R)

        # TODO: unnecessary computation of mha weights for previous chunk
        #print(f"{block_in.shape = }, {seq_mask.shape = }")
        block_out = self(block_in, seq_mask)  # (1, 1|2, C+R, F')
        current_chunk_out = block_out[0, -1]  # (C+R, F')

        return current_chunk_out
    


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
    



class ConformerEncoderV5(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV3Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.vrnn = VRNNV1(cfg.vrnn_config)
        self.module_list = torch.nn.ModuleList([ConformerBlockV5(cfg.block_cfg) for _ in range(cfg.num_layers)])

    @staticmethod
    def run_fn(module, sequence_mask):
        def block_forward(in_tensor):
            out_tensor = module(in_tensor, sequence_mask)
            return out_tensor
        
        return block_forward


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
        y = x.clone().view(batch_size, -1, x.size(-1))  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [2, B, N, (C+R)', F']

            y = self.vrnn(
                tensor=x[1].view(batch_size, -1, x.size(-1)),  # [B, T, F']
                state=y
            )  # [B, T, F']
            x = x[0]

        #
        # prepare output
        #
        def _merge_drop_fac(tensor_in):
            tensor_in = tensor_in.view(-1, ext_chunk_sz, tensor_in.size(-1))  # [B*N, C'+R', ...]
            tensor_in = tensor_in[:, :-subs_lookahead_size].reshape(batch_size, -1, tensor_in.size(-1))  # [B, C'*N, ...]
            return tensor_in
        
        seq_mask = _merge_drop_fac(sequence_mask.unsqueeze(-1)).squeeze(-1)  # [B, T']

        y = _merge_drop_fac(y)  # [B, T', F']
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C+R, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
        :param lookahead_size = R: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend,
        """
        # replace first frames of each chunk with future acoustic context
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, C+R)
        x, sequence_mask = self.frontend(input, sequence_mask)
        x = x.squeeze(0)  # ((C+R)', F')

        # save layer outs for next chunk (state)
        layer_outs: List[torch.Tensor] = [x]
        state: Optional[torch.Tensor] = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                state = states[-1][i]

            x = module.infer(x, sequence_mask, states=state, lookahead_size=lookahead_size)
            layer_outs.append(x)

        x = x.unsqueeze(0)  # (1, C', F')

        return x, sequence_mask, layer_outs
