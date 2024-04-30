import torch
from torch import nn
from typing import Optional, List, Tuple, Union

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from ..conformer_2.conformer_v3 import (
    mask_tensor,
    ConformerBlockV3,
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config


from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor

from returnn.torch.context import get_run_ctx

from ..auxil.functional import add_lookahead_v2, create_chunk_mask


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
            
            seq_mask = torch.ones(all_curr_chunks.size(0), device=input.device, dtype=bool).view(-1, ext_chunk_sz)  # (K+1, C+R)

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
        # y = self.mhsa.layernorm(x)
        # q = y[-ext_chunk_sz:]  # [C+R, F']

        # inv_seq_mask = ~seq_mask
        # output_tensor, _ = self.mhsa.mhsa(
        #     q, y, y, key_padding_mask=inv_seq_mask, need_weights=False
        # )  # [C+R, F]
        # x = output_tensor + x[-ext_chunk_sz:]  # [C+R, F]
        x = self.mhsa.infer(x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz)

        # chunk-independent convolution
        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)

        x = x.unsqueeze(0)
        x = self.conv(x) + x  # [1, C+R, F]
        x = x.squeeze(0)

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


class ConformerEncoderCOV2(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV3COV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int = 0, carry_over_size: int = 1,
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
        batch_size: int = data_tensor.size(0)
        use_chunks: bool = (data_tensor.dim() == sequence_mask.dim() + 1 == 4)
        attn_mask: Optional[torch.Tensor] = None

        if use_chunks:
            # chunking by reshaping [B, N, C, F] to [B*N, C, F] for frontend (pooling + conv2d)
            data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
            sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']

        if use_chunks:
            # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

            # [B, N, C', F'] -> [B, N, C'+R, F']
            x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

            # create chunk-causal attn_mask
            attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2)-lookahead_size, 
                                          lookahead_size=lookahead_size, 
                                          carry_over_size=carry_over_size,
                                          device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if use_chunks and lookahead_size > 0:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        return x, sequence_mask     # [B, N, C', F']

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        chunk_size: Optional[int] = None,
        lookahead_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (P, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param chunk_size: ...
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of future chunks
        """
        # (P, C) where first P is current chunk and rest is for future ac ctx.
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, P*C)

        input = input.view(-1, chunk_size, input.size(-1))    # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size)   # (P, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (P, C', F')

        if lookahead_size > 0:
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
        else:
            x = x[0]

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
            x = x[:-lookahead_size]  # (C', F')
            sequence_mask = sequence_mask[:, :-lookahead_size]   # (1, C')
        
        x = x.unsqueeze(0)  # (1, C', F')

        return x, sequence_mask, layer_outs
