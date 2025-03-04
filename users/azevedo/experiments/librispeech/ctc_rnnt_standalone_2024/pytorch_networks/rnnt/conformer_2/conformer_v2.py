import torch
from torch import nn
from typing import Optional, List, Tuple, Union

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config

from i6_models.util import compat

from i6_models.parts.conformer import (
    ConformerConvolutionV1,
    ConformerMHSAV1Config,
    ConformerPositionwiseFeedForwardV1,
)

# FIXME: just a quickfix for the error described below
from i6_experiments.users.azevedo.experiments.librispeech.ctc_rnnt_standalone_2024.pytorch_networks.rnnt.conformer_2.mha_hack import (
    MultiheadAttention as MultiHacktention
)

# https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580 and
# NOTE: monkeypatch of F.multi_head_attention_forward
from .mhav2 import _scaled_dot_product_attention
torch.nn.functional.scaled_dot_product_attention = _scaled_dot_product_attention


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


def create_chunk_mask(seq_len: int, chunk_size: int, device: Union[torch.device, str] = "cpu") -> torch.Tensor:
    """
    output of some embed may see every embed in the past and in the current chunk
    """
    attn_mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
    
    for i in range(0, seq_len, chunk_size):
        attn_mask[i:i+chunk_size, max(0, i-chunk_size):i+chunk_size] = True

    return attn_mask


class ConformerMHSAV2(torch.nn.Module):
    def __init__(self, cfg: ConformerMHSAV1Config):
        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)

        # FIXME (quickfix by https://github.com/pytorch/pytorch/issues/41508#issuecomment-1723119580)
        # self.mhsa = MultiHacktention(
        #     cfg.input_dim, cfg.num_att_heads, dropout=cfg.att_weights_dropout, batch_first=True
        # )
        self.mhsa = torch.nn.MultiheadAttention(
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
        
        # TODO: maybe in inference mode only give last chunk as query (or key) and val to save redundant weights computation for previous chunk
        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, 
            key_padding_mask=inv_sequence_mask, attn_mask=inv_attn_mask, 
            need_weights=False
        )  # [B,T,F]

        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor
    
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


class ConformerBlockV2(torch.nn.Module):
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

    def infer(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        states: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed (C, F')
        input should be parseable to forward without attn_mask
        states: (C, F') corresponding to previous chunk output of lower layer
        """
        if states is not None:
            block_in = torch.cat((states, input), dim=0)  # (2*C, F')
            block_in = block_in.reshape(2, -1, block_in.size(-1)).unsqueeze(0)  # (1, 2, C, F') for block.foward()

            seq_mask = torch.cat(
                (torch.full_like(sequence_mask, True, device=input.device), sequence_mask),
                dim=0
            ).unsqueeze(0)  # (1, 2, C)
        else:
            block_in = input[None, None]  # (1, 1, C, F')
            seq_mask = sequence_mask.unsqueeze(0)  # (1, 1, C)
        
        # TODO: unnecessary computation of mha weights for previous chunk
        block_out = self(block_in, seq_mask)  # (1, 1|2, C, F')
        current_chunk_out = block_out[0, -1]  # (C, F')

        return current_chunk_out


class ConformerEncoderV2(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV2(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F] or [B, N, C, F] where C is the chunk size
            and N = T'/C the number of chunks
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T'] or [B, N, C]
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        batch_size: int = data_tensor.size(0)
        use_chunks: bool = (data_tensor.dim() == sequence_mask.dim() + 1 == 4)

        if use_chunks:
            # chunking by reshaping [B, N, C, F] to [B*N, C, F] for frontend (pooling + conv2d)
            data_tensor = data_tensor.view(-1, data_tensor.size(-2), data_tensor.size(-1))
            sequence_mask = sequence_mask.view(-1, sequence_mask.size(-1))

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']
        print(f"{x.shape = }, {sequence_mask.shape = }")

        if use_chunks:
            # we are chunking, thus reshape to [B, N, C', F']
            x = x.view(batch_size, -1, x.size(-2), x.size(-1))
            sequence_mask = sequence_mask.view(batch_size, -1, sequence_mask.size(-1))  # [B, N, C']
            # create chunk-causal attn_mask
            attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),  # N * C' = T (crspnds. to whole sequence)
                                          chunk_size=x.size(2), device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C', F']

        return x, sequence_mask     # [B, N, C', F']

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio frames (1, C, F), assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: (1,) true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F'] where
            P = num. of previous chunks (we only save previous chunk => P = 1)
            L = num. of layers = num. of encoder blocks (L = 0 corresponds to frames of prev. chunk after frontend)
            F' = conformer dim after frontend
        """
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, C)
        x, sequence_mask = self.frontend(input, sequence_mask)
        x = x.squeeze(0)  # (C', F')
        # save layer outs for next chunk (state)
        layer_outs: List[torch.Tensor] = [x]
        state: Optional[torch.Tensor] = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                state = states[-1][i]

            x = module.infer(x, sequence_mask, states=state)
            layer_outs.append(x)

        x = x.unsqueeze(0)  # (1, C', F')

        return x, sequence_mask, layer_outs
