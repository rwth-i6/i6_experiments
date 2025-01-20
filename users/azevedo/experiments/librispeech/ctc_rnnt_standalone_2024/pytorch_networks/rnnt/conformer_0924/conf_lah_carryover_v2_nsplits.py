import torch
from torch import nn
from typing import Optional, List, Tuple, Union
import math

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from ..conformer_2.conformer_v3 import (
    mask_tensor,
    ConformerBlockV3,
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config


from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor

from returnn.torch.context import get_run_ctx

from ..auxil.functional import add_lookahead_v2, create_chunk_mask, pad_chunk_frames, Mode


class ConformerBlockV3COV1(ConformerBlockV3):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__(cfg)

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[List[torch.Tensor]],
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
            if lookahead_size > 0:
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
        # TODO: test
        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)

        x = x.unsqueeze(0)
        x = self.conv(x) + x  # [1, C+R, F]
        x = x.squeeze(0)

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


class ConformerEncoderCOV2NSplits(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV3COV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, 
        lookahead_size: int = 0, carry_over_size: int = 1, k: int = None, prior: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F] or [B, N, C, F] where C is the chunk size
            and N = T'/C the number of chunks
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead frames chunk is able to attend to
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        assert k is not None, "Need min. number of splits"

        # if not self.training and data_tensor.dim() == 3 and not prior:
        #     outs, seq_mask = self._infer_offline(input=data_tensor, sequence_mask=sequence_mask,
        #                                          lookahead_size=lookahead_size, carry_over_size=carry_over_size, k=k)
        #     return [(outs, seq_mask, Mode.OFFLINE)]

        batch_size: int = data_tensor.size(0)
        attn_mask: Optional[torch.Tensor] = None

        # chunking by reshaping [B, N, C, F] to [B*N, C, F] for frontend (pooling + conv2d)
        data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
        sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']

        # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
        x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
        sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

        # [B, N, C', F'] -> [B, N, C'+R, F']
        x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)
        ext_chunk_sz = x.size(2)

        # create chunk-causal attn_mask
        attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                        chunk_size=x.size(2)-lookahead_size, 
                                        lookahead_size=lookahead_size, 
                                        carry_over_size=carry_over_size,
                                        device=x.device)

        # streaming computation for both offline and online mode
        for module in self.module_list[:-k]:
            x = module(x, sequence_mask, attn_mask)  # [B, N, (C+R)', F']

        # split last k blocks for offline and streaming mode
        if lookahead_size > 0:
            y = x[:, :, :-lookahead_size].clone()
            sequence_mask_off = sequence_mask[:, :, :-lookahead_size]
        else:
            y = x.clone()
            sequence_mask_off = sequence_mask
        
        y = y.view(batch_size, -1, x.size(-1))  # used for offline path
        sequence_mask_off = sequence_mask_off.reshape(batch_size, -1)

        for module in self.module_list[-k:]:
            x = module(x, sequence_mask, attn_mask)  # [B, N, (C+R)', F']
            y = module(y, sequence_mask_off)  # [B, T, F']

        off_out = (y, sequence_mask_off, Mode.OFFLINE)  # [B, T, F']

        def _merge_drop_fac(tensor_in):
            if lookahead_size > 0:
                tensor_in = tensor_in.view(-1, ext_chunk_sz, tensor_in.size(-1))  # [B*N, C'+R', ...]
                tensor_in = tensor_in[:, :-lookahead_size]
            
            tensor_in = tensor_in.reshape(batch_size, -1, tensor_in.size(-1))

            return tensor_in
        
        on_mask_merged = _merge_drop_fac(sequence_mask.unsqueeze(-1)).squeeze(-1)  # [B, T]
        on_merged = _merge_drop_fac(x)
        on_out = (on_merged, on_mask_merged, Mode.STREAMING)  # [B, T', F']

        return on_out, off_out

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        chunk_size: Optional[int] = None,
        lookahead_size: Optional[int] = None,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param chunk_size: ...
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        """
        # (P, C) where first P is current chunk and rest is for future ac ctx.
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, P*C)

        input = input.view(-1, chunk_size, input.size(-1))    # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size)   # (P, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (P, C', F')

        # TODO: deal with fac size 0 (basically copy conf_lah_carryover_v2 infer)
        chunk = x[0]  # (C', F')
        chunk_seq_mask = sequence_mask[0]  # (C',)

        future_ac_ctx = x[1:]  # (P-1, C', F')
        fut_seq_mask = sequence_mask[1:]  # (P-1, C')

        future_ac_ctx = future_ac_ctx.view(-1, x.size(-1))  # (t, F')
        fut_seq_mask = fut_seq_mask.view(-1)  # (t,)

        future_ac_ctx = future_ac_ctx[:lookahead_size]  # (R, F')
        fut_seq_mask = fut_seq_mask[:lookahead_size]  # (R,)

        x = torch.cat((chunk, future_ac_ctx), dim=0)  # (C'+R, F')
        sequence_mask = torch.cat((chunk_seq_mask, fut_seq_mask), dim=0).unsqueeze(0)  # (1, C+R)

        # save layer outs for next chunk (state)
        layer_outs = [x]
        state = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                # state = states[-1][i]
                state = [prev_chunk[-1][i] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=state, lookahead_size=lookahead_size)
            layer_outs.append(x)

        if lookahead_size > 0:
            x = x[:-lookahead_size].unsqueeze(0)  # (1, C', F')
            sequence_mask = sequence_mask[:, :-lookahead_size]   # (1, C')

        return x, sequence_mask, layer_outs

    # def _infer_offline(
    #     self,
    #     input: torch.Tensor,
    #     sequence_mask: torch.Tensor,
    #     chunk_size: Optional[int] = None,
    #     carry_over_size: int = None,
    #     lookahead_size: Optional[int] = None,
    #     k: Optional[int] = None
    # ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        
    #     batch_size: int = input.size(0)
    #     attn_mask: Optional[torch.Tensor] = None

    #     input, sequence_mask = pad_chunk_frames(input, sequence_mask, self.chunk_size)

    #     x, sequence_mask = self.frontend(input, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']
    #     ext_chunk_sz = x.size(1)

    #     # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
    #     x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
    #     sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

    #     # [B, N, C', F'] -> [B, N, C'+R, F']
    #     x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

    #     # create chunk-causal attn_mask
    #     attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
    #                                     chunk_size=x.size(2)-lookahead_size, 
    #                                     lookahead_size=lookahead_size, 
    #                                     carry_over_size=carry_over_size,
    #                                     device=x.device)

    #     # streaming computation for both offline and online mode
    #     for module in self.module_list[:-k]:
    #         x = module(x, sequence_mask, attn_mask)  # [B, N, (C+R)', F']

    #     # split last k blocks for offline and streaming mode
    #     if lookahead_size > 0:
    #         y = x[:, :, :-lookahead_size].clone()
    #         sequence_mask_off = sequence_mask[:, :, :-lookahead_size]
    #     else:
    #         y = x.clone()
    #         sequence_mask_off = sequence_mask
        
    #     y = y.view(batch_size, -1, x.size(-1))  # used for offline path
    #     sequence_mask_off = sequence_mask_off.reshape(batch_size, -1)

    #     for module in self.module_list[-k:]:
    #         y = module(y, sequence_mask_off)  # [B, T, F']

    #     return y, sequence_mask_off  # [B, T, F']