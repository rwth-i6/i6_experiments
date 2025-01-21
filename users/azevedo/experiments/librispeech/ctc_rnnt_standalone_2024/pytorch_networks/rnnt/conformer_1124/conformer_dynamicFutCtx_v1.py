import torch
from torch import nn
from typing import Optional, List, Tuple, Union
from dataclasses import dataclass

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from ..conformer_2.conformer_v3 import (
    mask_tensor,
    ConformerBlockV3,
)

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config

from i6_models.config import ModelConfiguration, ModuleFactoryV1


from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_i6_native import mask_tensor

from returnn.torch.context import get_run_ctx

from ..auxil.functional import create_chunk_mask
from i6_models.util import compat
import math


@dataclass
class AggregatorV1Config(ModelConfiguration):
    input_dim: int
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_dropout: float
    ff_dim: int

    fut_weights_dropout: float
    num_att_heads: int
    with_bias: bool

    noise_std: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return AggregatorV1Config(**d)


@dataclass
class ConformerLearnedFutCtxEncoderV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    aggregator_cfg: AggregatorV1Config
    block_cfg: ConformerBlockV1Config



class AggregatorV1(torch.nn.Module):
    r"""Recurrent neural network transducer (RNN-T) prediction network.

    Taken from torchaudio
    """

    def __init__(self, cfg: AggregatorV1Config) -> None:
        """
        :param cfg: model configuration for the Aggregator
        """
        super().__init__()
        self.lstm_layers = torch.nn.ModuleList(
            [
                nn.LSTM(
                    input_size=cfg.input_dim if idx == 0 else cfg.lstm_hidden_dim,
                    hidden_size=cfg.lstm_hidden_dim,
                )
                for idx in range(cfg.num_lstm_layers)
            ]
        )
        self.dropout = torch.nn.Dropout(p=cfg.lstm_dropout)
        self.lstm_layer_norm = torch.nn.LayerNorm(cfg.lstm_hidden_dim)

        self.lstm_dropout = cfg.lstm_dropout

        # future context posterior stuff
        self.chunk_pivot_linear = torch.nn.Linear(cfg.lstm_hidden_dim, cfg.ff_dim)
        self.future_frames_linear = torch.nn.Linear(cfg.lstm_hidden_dim, cfg.ff_dim)
        #self.logsigmoid = torch.nn.LogSigmoid()
        self.sigmoid = torch.nn.Sigmoid()
        self.std = cfg.noise_std

        # multihead attention stuff
        self.embed_dim = cfg.ff_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads
        self.attn_dropout = torch.nn.Dropout(p=cfg.fut_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.q_proj = torch.nn.Linear(self.embed_dim, self.embed_dim, bias=cfg.with_bias)
        self.kv_proj = torch.nn.Linear(self.embed_dim, 2 * self.embed_dim, bias=cfg.with_bias)
        self.out_proj = torch.nn.Linear(self.embed_dim, cfg.input_dim, bias=cfg.with_bias)

    def forward(
        self,
        input: torch.Tensor,
        sequence_mask: torch.Tensor,
        chunk_size: int,
        state: Optional[List[List[torch.Tensor]]] = None,
        eps: float = 1e-3,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        r"""Forward pass.

        B: batch size;
        N = K: number of chunks per batch;
        C: chunk size;
        T: K*C
        F: current dim of vectors;
        """
        batch_size = input.size(0)

        sequence_mask = sequence_mask.view(batch_size, -1)  # [B, T]

        lstm_out = input.view(batch_size, -1, input.size(-1))
        state_out: List[List[torch.Tensor]] = []
        for layer_idx, lstm in enumerate(self.lstm_layers):
            lstm_out, lstm_state_out = lstm(
                lstm_out, None if state is None else [s.permute(1, 0, 2) for s in state[layer_idx]]
            )
            lstm_out = self.dropout(lstm_out)
            state_out.append([s.permute(1, 0, 2) for s in lstm_state_out])

        lstm_out_normed = self.lstm_layer_norm(lstm_out)

        # get chunk pivots for posteriors
        chunk_summary = lstm_out_normed.view(-1, chunk_size, lstm_out_normed.size(-1))  # [B*K, C, F]
        chunk_summary = torch.sum(chunk_summary, dim=1)  # [B*K, F]
        chunk_pivots = self.chunk_pivot_linear(chunk_summary)
        chunk_pivots = chunk_pivots.view(batch_size, -1, chunk_pivots.size(-1))  # [B, K, F]

        future_frames = self.future_frames_linear(lstm_out_normed)  # [B, T, F]

        # calculate logits of p(R_k^t | R_k^{t-1}, X_1^T) = p(R_k^t | R_k^{t-1}, X_1^t)
        dot_products = torch.matmul(chunk_pivots, future_frames.transpose(1, 2))  # [B, K, T]
        dot_products = dot_products * (math.sqrt(1.0 / float(chunk_pivots.size(-1))))
        # NOTE: noise only in training
        gaussian_noise = torch.randn(dot_products.size(), device=dot_products.device) * self.std
        eof_probs = self.sigmoid(dot_products + gaussian_noise)  # in [-infty, 0]
        
        num_chunks = chunk_pivots.size(1)
        # mask out frames inside or before current chunk
        chunk_endpts = torch.arange(1, num_chunks+1)*chunk_size
        mask = mask_tensor(eof_probs[0], chunk_endpts).unsqueeze(0)  # [1, K, T]
        eof_probs = eof_probs.masked_fill(mask, 0)

        # num_chunks = chunk_pivots.size(1)
        # # mask out frames inside or before current chunk
        # # FIXME: should last frame in chunk be considered (if not, then + 1 and change realisations in train_step)
        # chunk_endpts = torch.arange(1, num_chunks+1)*chunk_size + 1
        # mask = mask_tensor(fut_frame_posteriors[0], chunk_endpts).unsqueeze(0)  # [1, K, T]
        # fut_frame_posteriors = fut_frame_posteriors.masked_fill(mask, 0)

        # # calculate p(R_k^t | X_1^T) for each T x K (in logspace)
        # fut_frame_logits = torch.cumsum(fut_frame_posteriors, dim=-1)  # [B, K, T]
        
        # fut_frame_prods = torch.exp(fut_frame_logits)
        # fut_frame_probs = fut_frame_prods
        # p(R_k^{!>t} | X_1^T) = p(R_k^t | X_1^T) * (1 - p(R_k^{t+1} | R_k^t,  X_1^T))
        # updated_probs = fut_frame_prods[..., :-1] * (1 - torch.exp(fut_frame_posteriors[..., 1:]))
        # fut_frame_probs = torch.cat([updated_probs, fut_frame_prods[..., -1:]], dim=-1)

        # # print(fut_frame_prods[0, 0, :80])
        # # print(fut_frame_probs[0, 0, :80])
        # # print()

        # last_val = sequence_mask.clone()
        # last_val[..., :-1] *= ~sequence_mask[..., 1:]
        # last_val = last_val.unsqueeze(1).expand(-1, num_chunks, -1)

        # fut_frame_probs[last_val] = fut_frame_prods[last_val]
        # >>> [B, K, T]

        # print(f"A: {fut_frame_probs[0] = }")
        # mask "out" padding
        inv_sequence_mask = compat.logical_not(sequence_mask).unsqueeze(1)  # [B, 1, T]
        eof_probs = eof_probs.masked_fill(inv_sequence_mask, 1)

        # print(f"B: {fut_frame_probs[0] = }")
        # print(fut_frame_probs[0, -1, -80:])
        # print(sequence_mask[0, -80:].float())
        # print(sequence_mask[0].sum(dim=-1), sequence_mask.shape)

        in_futctx_probs = torch.ones_like(eof_probs, device=eof_probs.device)
        in_futctx_probs[..., 1:] = 1 - eof_probs[..., :-1]
        expected_fut_ctx = self._expected_mh_attn(
            query=chunk_pivots, keyval=future_frames, probs=in_futctx_probs
        )  # [B, K, F]

        # [old] calc E[R_k] for each k
        # expected_fut_ctx = torch.einsum('bkt, btf -> bkf', fut_frame_probs, input)  # [B, K, F]

        #assert torch.all(torch.abs(fut_frame_probs.sum(dim=-1) - 1) <= eps), f"probs: {fut_frame_probs.sum(dim=-1)}"

        return expected_fut_ctx, in_futctx_probs

    def _expected_mh_attn(self, query: torch.Tensor, keyval: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
        """Expected Multihead attention for each query based on provided probability densities 

        B: batch size
        K: number of chunks
        T: number of frames
        F: feature dim. 

        Args:
            query (torch.Tensor): [B, K, F]
            keyval (torch.Tensor): [B, T, F]
            probs (torch.Tensor): [B, K, T]

        Returns:
            torch.Tensor: [B, K, F] expected attention outputs
        """
        bsz = query.size(0)

        query_seq = self.q_proj(query)  # [B, T, #heads * F']
        key_seq, value_seq = self.kv_proj(keyval).chunk(2, dim=-1)  # [B, T, #heads * F']

        q = query_seq.view(bsz, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']
        k = key_seq.view(bsz, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']
        v = value_seq.view(bsz, -1, self.num_heads, self.embed_dim_per_head)

        scale_factor = 1 * (math.sqrt(1.0 / float(self.embed_dim_per_head)))

        attn_scales = torch.einsum("bkhf, bthf -> bhkt", q, k)
        attn_weights = nn.functional.softmax(scale_factor * attn_scales, dim=-1)
        attn_weights = attn_weights * probs.unsqueeze(1)  # [B, #heads, K, T]
        attn_weights = attn_weights / attn_weights.sum(dim=-1).unsqueeze(-1)
        #print(f"{attn_weights[0, 0] = }")

        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.einsum("bhkt, bthf -> bkhf", attn_weights, v).reshape(
            bsz, -1, self.embed_dim
        )  # [B, K, F]
        
        attn_output = self.out_proj(attn_output)

        return attn_output
    
    @staticmethod
    def _compute_exp_attn_probs(probs: torch.Tensor) -> torch.Tensor:
        """Computes probability that frame j can attend to frame i given the probabilities.

        Args:
            probs (torch.Tensor): [B, K, T] p(R_k^{!>t} | X_1^t) for each K x T and batch entry

        Returns:
            torch.Tensor: b_ij
        """
        P = probs
        T = P.size(-1)

        # compute cumulative product S along the last dimension
        S = torch.cat([torch.ones_like(P[..., :1]), torch.cumprod(1 - P, dim=-1)], dim=-1)  # Shape: [*, T+1]

        # prepare S_i and S_j for broadcasting
        S_i = S[..., :-1].unsqueeze(-1)  # Shape: [*, T, 1]
        S_j = S[..., 1:].unsqueeze(-2)   # Shape: [*, 1, T]

        # compute b_ij matrix using broadcasting
        b_ij = S_j / S_i                 # Shape: [*, T, T]

        # apply masking for i >= j
        indices = torch.arange(T, device=P.device)
        indices_i = indices.view(1, T, 1)  # Shape: [1, T, 1]
        indices_j = indices.view(1, 1, T)  # Shape: [1, 1, T]

        # Compute the mask where i >= j
        mask = indices_i >= indices_j       # Shape: [1, T, T]

        # Expand the mask to match the batch dimensions
        batch_shape = P.shape[:-1]         # Get the batch dimensions
        mask = mask.expand(*batch_shape, T, T)  # Shape: [*, T, T]

        # Set b_ij = 1 where i >= j
        b_ij = b_ij.masked_fill(mask, 1.0)

        # b_ij now contains the desired products for each batch
        return b_ij



    def infer(
        self,
        input,
        sequence_mask,
        lstm_states,
        prob_thr,
    ):
        # see rnntbeamsearch lstm
        # calc probs until thr reached
        #   keep track of lstm states
        pass


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


class ConformerLearnedFutCtxEncoderV1(torch.nn.Module):
    def __init__(self, cfg: ConformerEncoderV1Config):
        super().__init__()

        self.frontend = cfg.frontend()
        self.aggregator = AggregatorV1(cfg.aggregator_cfg)
        self.module_list = torch.nn.ModuleList([ConformerBlockV3COV1(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(
        self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, lookahead_size: int = 1, carry_over_size: int = 1,
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
        use_chunks = (data_tensor.dim() == sequence_mask.dim() + 1 == 4)
        batch_size = data_tensor.size(0)
        num_chunks = data_tensor.size(1) if use_chunks else 1
        chunk_size = data_tensor.size(2) if use_chunks else None
        attn_mask = None

        if use_chunks:
            # chunking by reshaping [B, N, C, F] to [B*N, C, F] for frontend (pooling + conv2d)
            data_tensor = data_tensor.view(-1, chunk_size, data_tensor.size(-1))
            sequence_mask = sequence_mask.view(-1, chunk_size)

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # x shape: [B, T, F'] or [B*N, C', F']

        subs_chunk_size = x.size(1)

        if use_chunks:
            # we are chunking: reshape to [B, N, C', F']
            x = x.view(batch_size, -1, subs_chunk_size, x.size(-1))  # [B, N, C', F']
            sequence_mask = sequence_mask.view(batch_size, -1, subs_chunk_size)  # [B, N, C']

            #
            # NOTE: Aggregator (test)
            #
            expected_fut_ctx, future_frame_probs = self.aggregator(
                x, sequence_mask=sequence_mask, chunk_size=subs_chunk_size
            )  # [B, N, F]

            # [B, N, C', F'] -> [B, N, C'+1, F']
            x = torch.cat((x, expected_fut_ctx.unsqueeze(2)), dim=2)
            new_mask = torch.zeros(x.shape[:-1], device=sequence_mask.device, dtype=sequence_mask.dtype)
            new_mask[..., :-1] = sequence_mask
            new_mask[..., -1] = new_mask[..., -2]
            sequence_mask = new_mask

            # create chunk-causal attn_mask
            attn_mask = create_chunk_mask(
                seq_len=num_chunks*(subs_chunk_size+1),
                chunk_size=subs_chunk_size,
                lookahead_size=1,
                carry_over_size=carry_over_size,
                device=x.device
            )

        for module in self.module_list:
            x = module(x, sequence_mask, attn_mask)  # [B, T, F'] or [B, N, C'+R, F']

        # remove expected future context from every chunk
        if use_chunks:
            x = x[:, :, :-1].contiguous()
            sequence_mask = sequence_mask[:, :, :-1].contiguous()

        return x, sequence_mask, future_frame_probs     # [B, N, C', F']