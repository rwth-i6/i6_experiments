import torch
from typing import Optional, List, Tuple
from dataclasses import dataclass

# from i6_models.parts.conformer import ConformerPositionwiseFeedForwardV2

from i6_models.parts.conformer import (
    ConformerMHSARelPosV1Config,
)
from ...components.feedforward.streamable_conformer_feedforward import (
    StreamableConformerPositionwiseFeedForward,
    StreamableConformerPositionwiseFeedForwardConfig
)
from ...components.attention.streamable_mhsa_relpos import StreamableConformerMHSARelPosV2
from ...components.convolution.streamable_conv import (
    StreamableConformerConvolutionV1,
    StreamableConformerConvolutionV1Config
)
from ...components.streamable_layernorm import StreamableLayerNormV1
from ....streamable_module import StreamableModule
from ....base_config import BaseConfig



@dataclass(kw_only=True)
class StreamableRelPosConformerBlockConfigV1(BaseConfig):
    """
        Attributes:
            ff_cfg:   Configuration for ConformerPositionwiseFeedForwardV1
            mhsa_cfg: Configuration for ConformerMHSAV1
            conv_cfg: Configuration for ConformerConvolutionV1
    """

    ff_cfg: StreamableConformerPositionwiseFeedForwardConfig
    mhsa_cfg: ConformerMHSARelPosV1Config
    conv_cfg: StreamableConformerConvolutionV1Config
    dual_mode: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()

        # TODO: ff and conv only have silu as activations
        # calling from_dict because we know the config module (ie not BasicConfig)
        d["ff_cfg"] = StreamableConformerPositionwiseFeedForwardConfig.from_dict(d["ff_cfg"])
        d["mhsa_cfg"] = ConformerMHSARelPosV1Config(**d["mhsa_cfg"])
        d["conv_cfg"] = StreamableConformerConvolutionV1Config.from_dict(d["conv_cfg"])
        
        return StreamableRelPosConformerBlockConfigV1(**d)

    def module(self):
        return StreamableRelPosConformerBlock


class StreamableRelPosConformerBlock(StreamableModule):
    def __init__(self, cfg: StreamableRelPosConformerBlockConfigV1):
        super().__init__()
        
        dual_mode = cfg.dual_mode
        self.ff1 = StreamableConformerPositionwiseFeedForward(cfg=cfg.ff_cfg)
        self.mhsa = StreamableConformerMHSARelPosV2(cfg=cfg.mhsa_cfg, dual_mode=dual_mode)
        self.conv = StreamableConformerConvolutionV1(model_cfg=cfg.conv_cfg, dual_mode=dual_mode)
        self.ff2 = StreamableConformerPositionwiseFeedForward(cfg=cfg.ff_cfg)
        self.final_layer_norm = StreamableLayerNormV1(cfg.ff_cfg.input_dim, dual_mode=dual_mode)

    def forward_offline(self, x: torch.Tensor, /, sequence_mask: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        TODO
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
        TODO
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
        TODO
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