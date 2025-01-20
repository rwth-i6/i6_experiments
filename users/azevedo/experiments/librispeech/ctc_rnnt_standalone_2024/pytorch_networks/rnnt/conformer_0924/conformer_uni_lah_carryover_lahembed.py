import torch
from typing import Tuple, Optional, List
from .conformer_uni_lah_carryover import ConformerEncoderCOV1, ConformerEncoderV1Config

from ..auxil.functional import add_lookahead, create_chunk_mask, mask_tensor


class ConformerEncoderCOLAHEmbedV1(ConformerEncoderCOV1):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__(cfg)

        self.lah_encoding = torch.nn.Parameter(torch.randn(1, cfg.block_cfg.ff_cfg.input_dim))  # [1, F']

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
            x, sequence_mask = add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

            # NOTE: only difference to ConformerEncoderCOV1
            x[:, :, -lookahead_size:] = x[:, :, -lookahead_size:] + self.lah_encoding
            # >>> [R, F'] + [1, F'] = [R, F']

            # create chunk-causal attn_mask
            attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2) - lookahead_size,
                                          lookahead_size=lookahead_size,
                                          carry_over_size=carry_over_size,
                                          device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if use_chunks:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        return x, sequence_mask  # [B, N, C', F']

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

        x, sequence_mask = add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)  # (1, 2, C'+R)
        x = x[0, 0]  # (C' + R, F')
        sequence_mask = sequence_mask[0, 0].unsqueeze(0)  # (1, C'+R)

        x[-lookahead_size:] = x[-lookahead_size:] + self.lah_encoding

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

        x = x[:-lookahead_size].unsqueeze(0)  # (1, C', F')
        sequence_mask = sequence_mask[:, :-lookahead_size]   # (1, C')

        return x, sequence_mask, layer_outs