import torch
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from .conformer_v3 import (
    mask_tensor,
    ConformerBlockV3,
    ConformerEncoderV3
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config



class ConformerBlockV3p1(ConformerBlockV3):
    def __init__(self, cfg: ConformerBlockV1Config):
        super().__init__(cfg)

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
            # padding we added to state (to make line above possible) should not be attended to
            seq_mask[:, 0, -lookahead_size:] = False
        else:
            block_in = input[None, None]  # (1, 1, C+R, F')
            seq_mask = sequence_mask.unsqueeze(0)  # (1, 1, C+R)

        # TODO: unnecessary computation of mha weights for previous chunk
        #print(f"{block_in.shape = }, {seq_mask.shape = }")
        block_out = self(block_in, seq_mask)  # (1, 1|2, C+R, F')
        current_chunk_out = block_out[0, -1]  # (C+R, F')

        return current_chunk_out


class ConformerEncoderCarryoverV1(ConformerEncoderV3):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__(cfg)


    def create_chunk_mask(self, seq_len: int, chunk_size: int, lookahead_size: int = 0, 
                          carry_over: Optional[int] = None, device: Union[torch.device, str] = "cpu"
    ) -> torch.Tensor:
        """
        chunk_size := num. subsampled frames in one chunk
        seq_len = N * (chunk_size + lookahead_size) with N = #chunks
        output of some embed may see every embed in the past and in the current chunk
        """
        attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

        chunk_ext_size = chunk_size + lookahead_size

        if carry_over is None:
            carry_over = chunk_ext_size
        
        for i in range(0, seq_len, chunk_ext_size):
            # attend to past chunk(s)
            attn_mask[i:i + chunk_ext_size, max(0, i - carry_over):i] = True

            # attend to current chunk and its lookahead
            attn_mask[i:i + chunk_ext_size, i:i + chunk_ext_size] = True

        # remove redundant lookahead
        attn_mask = attn_mask.view(attn_mask.size(0), -1, chunk_ext_size)  # [seq_len, N, C+R]
        attn_mask[:, :, :-lookahead_size] = False
        attn_mask = attn_mask.view(seq_len, seq_len)

        return attn_mask

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        lookahead_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        """
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, 2*C)

        input = input.view(2, -1, input.size(2))    # (2, C, F)
        sequence_mask = sequence_mask.view(2, -1)   # (2, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (2, C', F')

        x = x.unsqueeze(0)  # (1, 2, C', F')
        sequence_mask = sequence_mask.unsqueeze(0)  # (1, 2, C')

        x, sequence_mask = self.add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)  # (1, 2, C'+R)
        x = x[0, 0]  # (C' + R, F')
        sequence_mask = sequence_mask[0, 0].unsqueeze(0)  # (1, C'+R)

        # save layer outs for next chunk (state)
        layer_outs: List[torch.Tensor] = [x]
        state: Optional[torch.Tensor] = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                state = states[-1][i]

            x = module.infer(x, sequence_mask, states=state, lookahead_size=lookahead_size)
            layer_outs.append(x)

        x = x[:-lookahead_size].unsqueeze(0)  # (1, C', F')
        sequence_mask = sequence_mask[:, :-lookahead_size]   # (1, C')

        return x, sequence_mask, layer_outs
