from __future__ import annotations

__all__ = [
    "ConformerRoPEEncoderV1COV1",
]

import torch
from typing import List, Optional, Tuple

from i6_models.assemblies.conformer import ConformerEncoderV1Config
from ..conformer_0924.conf_lah_carryover_v2 import (
    ConformerEncoderCOV2
)

from ..auxil.functional import rotary_pos_encoding, create_chunk_mask, add_lookahead_v2, mask_tensor


class ConformerRoPEEncoderV1COV1(ConformerEncoderCOV2):
    """
    Modifications compared to ConformerEncoderV2:
    - supports Shaw's relative positional encoding using learnable position embeddings
      and Transformer-XL style relative PE using fixed sinusoidal or learnable position embeddings
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__(cfg)

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

        # add rotary positional encoding
        x = rotary_pos_encoding(x)

        if use_chunks:
            # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

            # [B, N, C', F'] -> [B, N, C'+R, F']
            x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

            # create chunk-causal attn_mask
            attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2) - lookahead_size,
                                          lookahead_size=lookahead_size,
                                          carry_over_size=carry_over_size,
                                          device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if use_chunks and lookahead_size > 0:
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
        time_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (P, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param chunk_size: ...
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of future chunks
        :param time_step: index of first frame in chunk wrt to whole audio
        """
        # (P, C) where first P is current chunk and rest is for future ac ctx.
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, P*C)

        input = input.view(-1, chunk_size, input.size(-1))  # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size)  # (P, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (P, C', F')

        x = rotary_pos_encoding(x, start_time=time_step)

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
            sequence_mask = sequence_mask[:, :-lookahead_size]  # (1, C')

        x = x.unsqueeze(0)  # (1, C', F')

        return x, sequence_mask, layer_outs
