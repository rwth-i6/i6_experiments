import torch
from typing import Optional, List, Tuple, Union

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from .conformer_v2 import (
    mask_tensor,
    ConformerMHSAV2
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config


from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerPositionwiseFeedForwardV1,
)



def create_chunk_mask(seq_len: int, chunk_size: int, lookahead_size: int = 0,
                      carry_over: Optional[int] = None, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """
    seq_len = chunk_size * (N + lookahead_size) with N = #chunks
    output of some embed may see every embed in the past and in the current chunk
    """
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)

    chunk_ext_size = chunk_size + lookahead_size
    for i in range(0, seq_len, chunk_ext_size):
        # NOTE: isolated chunks would be equivalent to setting attention mask to
        #       attn_mask[i:i+chunk_size, i:i+chunk_size] = True

        # attend to past chunk but not its lookahead
        attn_mask[i:i + chunk_ext_size, max(0, i - chunk_ext_size):max(0, i - lookahead_size)] = True
        # attend to current chunk and its lookahead
        attn_mask[i:i + chunk_ext_size, i:i + chunk_ext_size] = True

    return attn_mask


class ConformerBlockV3(torch.nn.Module):
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
            # reshape [B, N, C, F] to [B, C*N, F] to use attn_mask for chunks
            x = x.view(batch_size, -1, x.size(-1))
            sequence_mask = sequence_mask.view(sequence_mask.size(0), -1)   # [B, C*N]

        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask, attn_mask) + x  # [B, T, F]

        if use_chunks:
            # make convolution causal for chunks by reshaping [B, C*N, F] to [B*N, C, F]
            x = x.reshape(-1, chunk_size, x.size(-1))
            sequence_mask = sequence_mask.reshape(-1, chunk_size)

        # FIXME: a test (fix convolution overflow)
        x = x.masked_fill((~sequence_mask[:, :, None]), 0.0)

        x = self.conv(x) + x  # [B, T, F] or [B*N, C, F]

        x = 0.5 * self.ff2(x) + x  # [B, T, F] or [B*N, C, F]
        x = self.final_layer_norm(x)  # [B, T, F] or [B*N, C, F]

        if use_chunks:
            x = x.reshape(batch_size, -1, chunk_size, x.size(-1))   # [B, N, C, F]
        
        return x

    # TODO rework this and make use of create_chunk_mask
    def infer_DEPR(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (2, C, F')
        input should be parseable to forward without attn_mask
        states: (C, F') corresponding to previous chunk output of lower layer
        """
        if states is not None:
            #print(f"{states.shape = }")
            block_in = torch.cat((states.unsqueeze(0), input), dim=0)  # (3, C, F')
            
            num_chunks = 3
            block_in = block_in.reshape(num_chunks, -1, block_in.size(-1)).unsqueeze(0)  # (1, 3, C, F') for block.foward()

            #print(f"{states.shape = }, {sequence_mask.shape = }")
            seq_mask = torch.cat(
                (torch.full((1, sequence_mask.size(1)), True, device=input.device), sequence_mask),
                dim=0
            ).unsqueeze(0)  # (1, 3, C)

            #print(f"{states.shape = }, {seq_mask.shape = }")
        else:
            block_in = input.unsqueeze(0)  # (1, 2, C, F')
            seq_mask = sequence_mask.unsqueeze(0)  # (1, 2, C)

            #print(f"{block_in.shape = }, {seq_mask.shape = }")
        
        # TODO: unnecessary computation of mha weights for previous chunk
        block_out = self(block_in, seq_mask)  # (1, 2|3, C, F')

        # FIXME: 
        # - should be determined dynamically (e.g. based on lookahead size, better: create_chunk_mask)
        # - a lot of unnecessary computations
        current_chunk_out = block_out[0, -2:]  # (2, C, F')

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


class ConformerEncoderV3(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV3(cfg.block_cfg) for _ in range(cfg.num_layers)])


    def create_chunk_mask(self, seq_len: int, chunk_size: int, lookahead_size: int = 0,
                          carry_over: Optional[int] = None, device: Union[torch.device, str] = "cpu"
    ) -> torch.Tensor:
        
        return create_chunk_mask(
            seq_len=seq_len, chunk_size=chunk_size, lookahead_size=lookahead_size, 
            carry_over=carry_over, device=device)


    @staticmethod
    def add_lookahead(x: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int):
        if lookahead_size <= 0:
            return x, sequence_mask

        def roll_cat(left, last_val):
            # for each chunk i we want to concatenate it with lookahead frames of chunk i+1
            # i    ||  i+1[:lookahead_size]
            # i+1  ||  i+2[:lookahead_size]
            right = torch.roll(left[:, :, :lookahead_size], shifts=-1, dims=1)
            right[:, -1] = last_val  # last chunk has no lookahead
            return torch.cat((left, right), dim=2)

        # lookahead (assumes lookahead_size < chunk_size)
        x = roll_cat(x, 0)  # (B, N, C'+R, F')
        # adjust sequence mask
        sequence_mask = roll_cat(sequence_mask, False)  # (B, N, C'+R)

        return x, sequence_mask

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int = 0,
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
        #print(f"{x.shape = }, {sequence_mask.shape = }, {lookahead_size = }")

        if use_chunks:
            # we are chunking, thus reshape to [B, N, C'+R, F'] where R = lookahead_size
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))  # [B, N, C', F']
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']

            # [B, N, C', F'] -> [B, N, C'+R, F']
            x, sequence_mask = self.add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

            # create chunk-causal attn_mask
            attn_mask = self.create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                          chunk_size=x.size(2)-lookahead_size, 
                                          lookahead_size=lookahead_size, device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if use_chunks:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        return x, sequence_mask     # [B, N, C', F']

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
        lookahead_size: Optional[int] = None,
        chunk_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
        :param lookahead_size: number of lookahead frames chunk is able to attend to
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        :param chunk_size
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
