import torch
from typing import Optional, List, Tuple
import math

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerEncoderV1Config
from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerPositionwiseFeedForwardV1,
)
from ..conformer_2.conformer_v2 import ConformerMHSAV2

from ..auxil.functional import create_chunk_mask, mask_tensor, Mode


class ConformerBlockCOPreLAHV1(torch.nn.Module):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardV1(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAV2(cfg=cfg.mhsa_cfg)
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
        use_chunks: bool = (x.dim() == sequence_mask.dim() + 1 == 4)
        chunk_size: int = x.size(-2)
        batch_size: int = x.size(0)

        if use_chunks:
            # reshape [B, N, C(+R), F] to [B, C(+R)*N, F] to use attn_mask for chunks
            x = x.view(batch_size, -1, x.size(-1))
            sequence_mask = sequence_mask.view(sequence_mask.size(0), -1)  # [B, C(+R)*N]

        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask, attn_mask) + x  # [B, T, F]

        if use_chunks:
            # make convolution causal for chunks by reshaping [B, C(+R)*N, F] to [B*N, C(+R), F]
            x = x.reshape(-1, chunk_size, x.size(-1))
            sequence_mask = sequence_mask.reshape(-1, chunk_size)

        # prevent convolution to write padding when using multiple layers
        x = x.masked_fill((~sequence_mask[:, :, None]), 0.0)
        x = self.conv(x) + x  # [B, T, F]

        x = 0.5 * self.ff2(x) + x  # [B, T, F] or [B*N, C(+R), F]
        x = self.final_layer_norm(x)  # [B, T, F] or [B*N, C(+R), F]

        if use_chunks:
            x = x.reshape(batch_size, -1, chunk_size, x.size(-1))  # [B, N, C(+R), F]

        return x

    def infer_old(
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
        # print(f"{block_in.shape = }, {seq_mask.shape = }")
        block_out = self(block_in, seq_mask)  # (1, 1|2, C+R, F')
        current_chunk_out = block_out[0, -1]  # (C+R, F')

        return current_chunk_out

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[torch.Tensor],
        lookahead_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (C+R, F')
            where C = chunk_size, R = lookahead_size (, K = carry_over_size)
        sequence_mask: (1, C+R)
        states: List[Tensor(C+R, F')] corresponding to previous chunk output of lower layer
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            all_curr_chunks = torch.cat((*states, input), dim=0)  # [(K+1)*(C+R), F'] = [t, F']

            seq_mask = torch.ones(all_curr_chunks.size(0), device=input.device, dtype=bool).view(-1, ext_chunk_sz)
            seq_mask[:-1, -lookahead_size:] = False
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.flatten()  # [t]

        else:
            all_curr_chunks = input  # [C+R, F']
            seq_mask = sequence_mask.flatten()  # [C+R]

        #
        # Block forward computation
        #
        x = 0.5 * self.ff1(all_curr_chunks) + all_curr_chunks  # [t, F']

        # multihead attention
        y = self.mhsa.layernorm(x)
        q = y[-ext_chunk_sz:]  # [C+R, F']

        inv_seq_mask = ~seq_mask
        output_tensor, _ = self.mhsa.mhsa(
            q, y, y, key_padding_mask=inv_seq_mask, need_weights=False
        )  # [C+R, F]
        x = output_tensor + x[-ext_chunk_sz:]  # [C+R, F]

        # chunk-independent convolution
        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)

        x = x.unsqueeze(0)
        x = self.conv(x) + x  # [1, C+R, F]
        x = x.squeeze(0)

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


class ConformerEncoderCOPreLAHV1(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList(
            [ConformerBlockCOPreLAHV1(cfg.block_cfg) for _ in range(cfg.num_layers)]
        )

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, 
        lookahead_size: Optional[float], carry_over_size: Optional[int] = None, mode: Optional[Mode] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, N, C+R, F] where C is the chunk size,
            N = T'/C the number of chunks, R the future acoustic context size (in #frames)
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead-frames chunk is able to attend to
        :param carry_over_size:
        :param mode:
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """            
        assert None not in [lookahead_size, carry_over_size, mode]

        batch_size = data_tensor.size(0)
        #
        # frontend
        #
        # reshaping [B, N, C+R, F] to [B*N, C+R, F] for frontend (pooling + conv2d)
        if mode == Mode.STREAMING:
            data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
            sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B*N, (C+R)', F']

        if mode == Mode.STREAMING:
            subs_lookahead_size = math.ceil(lookahead_size / 6)  # div by 6 as we have sub_factor of 6
            ext_chunk_sz = x.size(1)

        #
        # conformer blocks
        #

        # create chunk-causal attn_mask
        attn_mask = None
        if mode == mode.STREAMING:
            # we are chunking => reshape to [B, N, (C+R)', F']
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, (C+R)']
            # FIXME: this is so unnecessary (could use easier mask function)
            attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2) - subs_lookahead_size,
                                          carry_over_size=carry_over_size,
                                          lookahead_size=subs_lookahead_size, device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)

        #
        # prepping results
        #
        def _merge_drop_fac(tensor_in):
            tensor_in = tensor_in.view(-1, ext_chunk_sz, tensor_in.size(-1))  # [B*N, C'+R', ...]
            tensor_in = tensor_in[:, :-subs_lookahead_size].reshape(batch_size, -1, tensor_in.size(-1))
            return tensor_in

        if mode == Mode.STREAMING:
            mask_merged = _merge_drop_fac(sequence_mask.unsqueeze(-1)).squeeze(-1)  # [B, T]
            out_merged = _merge_drop_fac(x)

            return out_merged, mask_merged  # [B, T', F']

        return x, sequence_mask

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
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param lookahead_size = R: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        """
        # replace first frames of each chunk with future acoustic context
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, C+R)
        x, sequence_mask = self.frontend(input, sequence_mask)
        x = x.squeeze(0)  # ((C+R)', F')
        subs_lookahead_size = math.ceil(lookahead_size / 6)

        # save layer outs for next chunk (state)
        layer_outs = [x]
        state = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                state = [prev_chunk[-1][i] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=state, lookahead_size=subs_lookahead_size)
            layer_outs.append(x)

        x = x.unsqueeze(0)  # (1, (C+R)', F')
        x = x[:, :-subs_lookahead_size].contiguous()
        # TODO: test: why didnt i include this before
        sequence_mask = sequence_mask[:, :-subs_lookahead_size]   # (1, C')

        return x, sequence_mask, layer_outs
