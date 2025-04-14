import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union, Literal
import math
from copy import deepcopy
from dataclasses import dataclass

from i6_models.util import compat

from i6_models.config import ModelConfiguration, ModuleFactoryV1

from i6_models.parts.conformer import (
    ConformerConvolutionV2,
    ConformerMHSARelPosV1Config,
    ConformerPositionwiseFeedForwardV2,
    ConformerPositionwiseFeedForwardV2Config,
)
from ..conformer_1124.conf_relpos_streaming_v1 import ConformerRelPosBlockV1COV1Config

from i6_models.parts.conformer import (
    ConformerConvolutionV1Config,
    ConformerPositionwiseFeedForwardV1,
)

from i6_models.parts.dropout import BroadcastDropout

from i6_models.primitives.specaugment import specaugment_v1_by_length

from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1 import Predictor, Joiner

from ..conformer_0225.conf_lah_carryover_v4 import (
    StreamableModule,
    StreamableLayerNormV1,
    StreamableFeatureExtractorV1,
    StreamableConformerConvolutionV1
)
from ..auxil.functional import add_lookahead_v2, create_chunk_mask, Mode, mask_tensor

from returnn.torch.context import get_run_ctx



class StreamableJoinerV1(StreamableModule):
    r"""Streamable RNN-T joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")

    Taken directly from torchaudio
    """

    def __init__(
            self, input_dim: int, output_dim: int, activation: str = "relu", 
            dropout: float = 0.0, dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None, 
            dual_mode: bool = True
    ) -> None:
        super().__init__()
        self.joiner_off = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            dropout_broadcast_axes=dropout_broadcast_axes
        )
        self.joiner_on = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            dropout_broadcast_axes=dropout_broadcast_axes
        ) if dual_mode else self.joiner_off

    def forward_offline(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_off(source_encodings, source_lengths, target_encodings, target_lengths)
    
    def forward_streaming(
            self,
            source_encodings: torch.Tensor,
            source_lengths: torch.Tensor,
            target_encodings: torch.Tensor,
            target_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.joiner_on(source_encodings, source_lengths, target_encodings, target_lengths)


class StreamableConformerMHSARelPosV2(StreamableModule):
    def __init__(self, cfg: ConformerMHSARelPosV1Config, dual_mode: bool):
        super().__init__()

        self.layernorm = StreamableLayerNormV1(cfg.input_dim, dual_mode=dual_mode)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.learnable_pos_emb = cfg.learnable_pos_emb
        self.rel_pos_clip = cfg.rel_pos_clip
        self.separate_pos_emb_per_head = cfg.separate_pos_emb_per_head
        self.with_pos_bias = cfg.with_pos_bias
        self.pos_emb_dropout = nn.Dropout(cfg.pos_emb_dropout)

        assert not self.learnable_pos_emb or self.rel_pos_clip

        self.att_weights_dropout = nn.Dropout(cfg.att_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.qkv_proj = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=cfg.with_bias)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.with_bias)

        self.register_parameter("rel_pos_embeddings", None)
        self.register_parameter("pos_bias_u", None)
        self.register_parameter("pos_bias_v", None)

        self.pos_emb_dim = (
            self.embed_dim if cfg.with_linear_pos or cfg.separate_pos_emb_per_head else self.embed_dim_per_head
        )
        if self.learnable_pos_emb:
            self.rel_pos_embeddings = nn.parameter.Parameter(torch.empty(self.rel_pos_clip * 2 + 1, self.pos_emb_dim))
        if cfg.with_linear_pos:
            self.linear_pos = nn.Linear(
                self.pos_emb_dim,
                self.embed_dim if cfg.separate_pos_emb_per_head else self.embed_dim_per_head,
                bias=False,
            )
        else:
            self.linear_pos = nn.Identity()

        if self.with_pos_bias:
            self.pos_bias_u = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))
            self.pos_bias_v = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))

        self.dropout = BroadcastDropout(cfg.dropout, dropout_broadcast_axes=cfg.dropout_broadcast_axes)

        self._reset_parameters()

    def _reset_parameters(self):
        if self.learnable_pos_emb:
            nn.init.xavier_normal_(self.rel_pos_embeddings)
        if self.with_pos_bias:
            # init taken from espnet default
            nn.init.xavier_uniform_(self.pos_bias_u)
            nn.init.xavier_uniform_(self.pos_bias_v)

    def forward_offline(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside
        :param attn_mask: bool mask of shape (T, T)
        """
        output_tensor = self.layernorm(input_tensor)  # [B, T, F]

        time_dim_size = output_tensor.shape[1]
        batch_dim_size = output_tensor.shape[0]

        # attention mask
        # T: query seq. length, T' key/value seg length; T = T' if same input tensor

        inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T']
        inv_sequence_mask = inv_sequence_mask.unsqueeze(1)  # [B, 1, T']
        if attn_mask is not None:
            inv_attn_mask = compat.logical_not(attn_mask)
            inv_attn_mask = inv_attn_mask.unsqueeze(0)  # [1, T', T']

        total_mask = inv_sequence_mask
        if attn_mask is not None:
            total_mask = total_mask + inv_attn_mask
        total_mask = total_mask.unsqueeze(1)  # [B, 1, T', T']

        # query, key and value sequences
        query_seq, key_seq, value_seq = self.qkv_proj(output_tensor).chunk(3, dim=-1)  # [B, T, #heads * F']
        q = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']
        k = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']

        if self.learnable_pos_emb:
            pos_seq_q = torch.arange(time_dim_size, device=input_tensor.device)
            pos_seq_k = torch.arange(time_dim_size, device=input_tensor.device)

            distance_mat = pos_seq_k[None, :] - pos_seq_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.rel_pos_clip, self.rel_pos_clip)

            final_mat = distance_mat_clipped + self.rel_pos_clip

            rel_pos_embeddings = self.rel_pos_embeddings[final_mat]  # [T, T', pos_emb_dim]
        else:
            rel_pos_embeddings = self._sinusoidal_pe(
                torch.arange(time_dim_size - 1, -time_dim_size, -1, device=input_tensor.device, dtype=torch.float32),
                self.pos_emb_dim,
            ).view(
                1, 2 * time_dim_size - 1, self.pos_emb_dim
            )  # [1, T+T'-1, pos_emb_dim]

        # dropout relative positional embeddings
        rel_pos_embeddings = self.pos_emb_dropout(
            rel_pos_embeddings
        )  # [T, T', pos_emb_dim] or [1, T+T'-1, pos_emb_dim]
        rel_pos_embeddings = rel_pos_embeddings.unsqueeze(2)  # [T, T', 1, pos_emb_dim] or [1, T+T'-1, 1, pos_emb_dim]

        # linear transformation or identity
        rel_pos_embeddings = self.linear_pos(rel_pos_embeddings)  # [T, T', 1, F'|F] or [1, T+T'-1, 1, F'|F]

        if self.separate_pos_emb_per_head:
            rel_pos_embeddings = rel_pos_embeddings.squeeze(2).reshape(
                *rel_pos_embeddings.shape[:2], -1, self.embed_dim_per_head
            )  # [T, T', #heads, F'] or [1, T+T'-1, #heads, F']

        q_with_bias_u = q + self.pos_bias_u if self.with_pos_bias else q  # [B, T, #heads, F']
        q_with_bias_v = q + self.pos_bias_v if self.with_pos_bias else q

        # attention matrix a and c
        attn_ac = torch.einsum("bihf, bjhf -> bhij", q_with_bias_u, k)  # [B, #heads, T, T']

        # attention matrix b and d
        attn_bd = torch.einsum(
            "bihf, ijhf -> bhij", q_with_bias_v, rel_pos_embeddings
        )  # [B, #heads, T, T'] or [B, #heads, T, T+T'+1]

        if not self.learnable_pos_emb:
            attn_bd = self._rel_shift_bhij(attn_bd, k_len=time_dim_size)  # [B, #heads, T, T']

        attn = attn_ac + attn_bd  # [B, #heads, T, T']
        attn_scaled = attn * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # NOTE: mask applied with masked_fill instead of addition for stable grads for zero-rows
        attn_scaled = attn_scaled.masked_fill(total_mask, float("-inf"))
        # softmax and dropout
        attn_output_weights = self.att_weights_dropout(F.softmax(attn_scaled, dim=-1))  # [B, #heads, T, T']

        attn_output_weights = attn_output_weights.masked_fill(total_mask, 0.0)

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']
        attn_output = torch.einsum("bhij, bjhf -> bihf", attn_output_weights, v).reshape(
            batch_dim_size, -1, self.embed_dim
        )

        output_tensor = self.out_proj(attn_output)

        output_tensor = self.dropout(output_tensor)

        return output_tensor  # [B,T,F]

    @staticmethod
    def _rel_shift_bhij(x, k_len=None):
        """
        :param x: input tensor of shape (B, H, T, L) to apply left shift
        :k_len: length of the key squence
        """
        x_shape = x.shape

        x = torch.nn.functional.pad(x, (1, 0))  # [B, H, T, L+1]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2])  # [B, H, L+1, T]
        x = x[:, :, 1:]  # [B, H, L, T]
        x = x.reshape(x_shape)  # [B, H, T, L]]

        return x[:, :, :, :k_len] if k_len else x  # [B, H, T, T']

    @staticmethod
    def _sinusoidal_pe(pos_seq: torch.Tensor, embed_dim: int):
        """
        :param pos_seq: 1-D position sequence for which to compute embeddings
        :param embed_dim: embedding dimension
        """
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0, device=pos_seq.device) / embed_dim))

        sinusoid_input = torch.outer(pos_seq, inv_freq)

        pos_emb = torch.zeros(pos_seq.shape[0], embed_dim, device=pos_seq.device)

        pos_emb[:, 0::2] = sinusoid_input.sin()
        pos_emb[:, 1::2] = sinusoid_input.cos()

        return pos_emb

    def forward_streaming(
            self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        :param input_tensor: [B, N, C, F]
        :param sequence_mask: [B, N, C]
        :param attn_mask: [N*C, N*C]

        :return: [B, N, C, F]
        """
        assert input_tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = input_tensor.shape

        # [B, N, C, F] -> [B, N*C, F]
        input_tensor = input_tensor.flatten(1, 2)
        sequence_mask = sequence_mask.flatten(1, 2)

        out = self.forward_offline(input_tensor, sequence_mask, attn_mask)

        out = out.view(bsz, num_chunks, chunk_sz, input_tensor.size(-1))

        return out

    def infer(
            self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:

        # x.shape: [t, F]
        attn_mask = torch.ones(x.size(0), x.size(0), device=x.device, dtype=torch.bool)
        y = self.forward_offline(
            input_tensor=x.unsqueeze(0), sequence_mask=seq_mask.unsqueeze(0), attn_mask=attn_mask)
        
        return y[0, -ext_chunk_sz:]  # [C+R, F]


class StreamableConformerBlockRelPosV1(StreamableModule):
    def __init__(self, cfg: ConformerRelPosBlockV1COV1Config, dual_mode: bool):
        super().__init__()

        self.ff1 = ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg)
        self.mhsa = StreamableConformerMHSARelPosV2(cfg=cfg.mhsa_cfg, dual_mode=dual_mode)
        self.conv = StreamableConformerConvolutionV1(model_cfg=cfg.conv_cfg, dual_mode=dual_mode)
        self.ff2 = ConformerPositionwiseFeedForwardV2(cfg=cfg.ff_cfg)
        self.final_layer_norm = StreamableLayerNormV1(cfg.ff_cfg.input_dim, dual_mode=dual_mode)

    def forward_offline(self, x: torch.Tensor, /, sequence_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        T = N*C

        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """

        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  # [B, T, F]
        x = x.masked_fill((~sequence_mask[:, :, None]), 0.0)
        x = self.conv(x) + x  # [B, T, F]
        x = 0.5 * self.ff2(x) + x  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]

        return x

    def forward_streaming(
            self, x: torch.Tensor, /, sequence_mask: torch.Tensor,
            attn_mask: torch.Tensor, lookahead_size: int, carry_over_size: int,
    ) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, N, C, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, N, C]
        :param attn_mask: attention mask [N*C, N*C]
        :param lookahead_size: number of future frames in chunk

        :return: torch.Tensor of shape [B, N, C, F]
        """
        assert x.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = x.shape

        x = 0.5 * self.ff1(x) + x  # [B, N, C, F]
        x = self.mhsa(x, sequence_mask, attn_mask) + x  # [B, N, C, F]

        x = x.masked_fill((~sequence_mask.unsqueeze(-1)), 0.0)
        x = self.conv(x, lookahead_size, carry_over_size) + x

        x = 0.5 * self.ff2(x) + x  # [B, T, F] or [B*N, C, F]
        x = self.final_layer_norm(x)  # [B, T, F] or [B*N, C, F]

        x = x.reshape(bsz, num_chunks, chunk_sz, x.size(-1))  # [B, N, C, F]

        return x

    def infer(
            self,
            input: torch.Tensor,
            sequence_mask: torch.Tensor,
            states: Optional[List[torch.Tensor]],
            curr_layer: Optional[torch.Tensor],
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        input should be assumed to be streamed [C+R, F']
            where C = chunk_size, R = lookahead_size (, K = carry_over_size)
        sequence_mask: [1, C+R]
        states: List[Tensor[C+R, F']] corresponding to previous chunk output of lower layer
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            all_curr_chunks = torch.cat((*states, input), dim=0)  # [(K+1)*(C+R), F'] = [t, F']

            seq_mask = torch.ones(
                all_curr_chunks.size(0), device=input.device, dtype=bool
            ).view(-1, ext_chunk_sz)  # (K+1, C+R)

            if lookahead_size > 0:
                seq_mask[:-1, -lookahead_size:] = False
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.flatten()  # [t]

        else:
            all_curr_chunks = input  # [C+R, F']
            seq_mask = sequence_mask.flatten()  # [C+R]

        #
        # block forward computation
        #
        x = 0.5 * self.ff1(all_curr_chunks) + all_curr_chunks  # [t, F']

        x = self.mhsa.infer(
            x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz
        ) + x[-ext_chunk_sz:]  # [C+R, F']

        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)
        x = self.conv.infer(
            x, states=curr_layer, chunk_sz=ext_chunk_sz, lookahead_sz=lookahead_size
        ) + x  # [C+R, F']

        x = 0.5 * self.ff2(x) + x  # [C+R, F]
        x = self.final_layer_norm(x)  # [C+R, F]

        return x


@dataclass
class StreamableConformerEncoderRelPosV2Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int
    dual_mode: bool

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerRelPosBlockV1COV1Config


class StreamableConformerEncoderRelPosV2(StreamableModule):
    def __init__(self, cfg: StreamableConformerEncoderRelPosV2Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = nn.ModuleList(
            [StreamableConformerBlockRelPosV1(cfg.block_cfg, dual_mode=cfg.dual_mode) for _ in range(cfg.num_layers)]
        )

    def forward_offline(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, *args, **kwargs) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask

    def forward_streaming(self,
                          data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int,
                          carry_over_size: int,
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        B: batch size, N: number of chunks = T/C, C: chunk size, F: feature dim, F': internal and output feature dim,
        T': data time dim, T: down-sampled time dim (internal time dim)

        :param data_tensor: input tensor, shape: [B, N, C, F]
        :param sequence_mask: mask tensor where 1 defines positions within the sequence and 0 padding, shape: [B, T'] or [B, N, C]
        :param lookahead_size: number of lookahead frames chunk is able to attend to

        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]
        """
        assert data_tensor.dim() == 4, ""

        batch_sz, num_chunks, _, _ = data_tensor.shape
        attn_mask = None

        data_tensor = data_tensor.flatten(0, 1)  # [B*N, C, F]
        sequence_mask = sequence_mask.flatten(0, 1)
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)

        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        # [B, N, C', F'] -> [B, N, C'+R, F']
        x, sequence_mask = add_lookahead_v2(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

        attn_mask = create_chunk_mask(seq_len=(x.size(1) * x.size(2)),
                                      chunk_size=x.size(2) - lookahead_size,
                                      lookahead_size=lookahead_size,
                                      carry_over_size=carry_over_size,
                                      device=x.device)

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask=attn_mask,
                       lookahead_size=lookahead_size, carry_over_size=carry_over_size)  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if lookahead_size > 0:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        P = num. of future chunks

        :param input: audio frames [P, C, F], assumed to be streamed with some chunk size and stride (see AudioStreamer)
        :param lengths: [1,] true length of audio frames (without padding)
        :param states: previous' chunk encoder block outputs [P, L, F']
        :param chunk_size: ...
        :param lookahead_size: number of lookahead frames chunk is able to attend to
        """
        if self._mode != Mode.STREAMING:
            self.set_mode_cascaded(Mode.STREAMING)

        # [P, C] where first P is current chunk and rest is for future ac ctx.
        sequence_mask = mask_tensor(tensor=input, seq_len=lengths)  # (1, P*C)

        input = input.view(-1, chunk_size, input.size(-1))  # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size)  # (P, C)

        x, sequence_mask = self.frontend(input, sequence_mask)  # (P, C', F')

        if lookahead_size > 0:
            chunk = x[0]  # [C', F']
            chunk_seq_mask = sequence_mask[0]  # [C',]

            future_ac_ctx = x[1:]  # [P-1, C', F']
            fut_seq_mask = sequence_mask[1:]  # [P-1, C']

            future_ac_ctx = future_ac_ctx.view(-1, x.size(-1))  # [t, F']
            fut_seq_mask = fut_seq_mask.view(-1)  # [t,]

            future_ac_ctx = future_ac_ctx[:lookahead_size]  # [R, F']
            fut_seq_mask = fut_seq_mask[:lookahead_size]  # [R,]

            x = torch.cat((chunk, future_ac_ctx), dim=0)  # [C'+R, F']
            sequence_mask = torch.cat((chunk_seq_mask, fut_seq_mask), dim=0).unsqueeze(0)  # [1, C+R]
        else:
            x = x[0]

        # save layer outs for next chunk (state)
        layer_outs = [x]
        prev_layer = curr_layer = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                prev_layer = [prev_chunk[-1][i] for prev_chunk in states]
                curr_layer = [prev_chunk[-1][i + 1] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=prev_layer, curr_layer=curr_layer, lookahead_size=lookahead_size)
            layer_outs.append(x)

        # remove fac if any
        if lookahead_size > 0:
            x = x[:-lookahead_size]  # [C', F']
            sequence_mask = sequence_mask[:, :-lookahead_size]  # [1, C']

        x = x.unsqueeze(0)  # [1, C', F']

        return x, sequence_mask, layer_outs
