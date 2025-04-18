import torch
from torch import nn
from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

from i6_models.config import ModelConfiguration, ModuleFactoryV1
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1 import Joiner
from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import SpecaugConfig
from ..conformer_1124.conf_relpos_streaming_v1 import ConformerRelPosBlockV1COV1Config
from ..conformer_0325.conf_dual_0325_v1 import (
    StreamableModule,
    StreamableFeatureExtractorV1,
    StreamableConformerEncoderRelPosV2Config,
    StreamableConformerBlockRelPosV1,
    BroadcastDropout,
    StreamableJoinerV1,
)
from ..auxil.functional import add_lookahead_v2, create_chunk_mask, Mode, mask_tensor
from returnn.torch.context import get_run_ctx



class StreamableJoinerV2(StreamableModule):
    r"""Streamable Monotonic RNN-T joint network.

    Args:
        input_dim (int): source and target input dimension.
        output_dim (int): output dimension.
        activation (str, optional): activation function to use in the joiner.
            Must be one of ("relu", "tanh"). (Default: "relu")
    """

    def __init__(
            self, 
            input_dim: int,
            output_dim: int,
            activation: str = "relu",
            dropout: float = 0.0,
            dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]] = None,
            dual_mode: bool = True
    ) -> None:
        super().__init__()
        self.joiner_off = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            dropout_broadcast_axes=dropout_broadcast_axes,
        )
        self.joiner_on = Joiner(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            dropout=dropout,
            dropout_broadcast_axes=dropout_broadcast_axes,
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



@dataclass
class StreamableConformerEncoderRelPosV3Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int
    dual_mode: bool

    # nested configurations
    feature_extraction_config: LogMelFeatureExtractionV1Config
    specaug_config: SpecaugConfig
    specauc_start_epoch: int
    frontend: ModuleFactoryV1
    block_cfg: ConformerRelPosBlockV1COV1Config


class StreamableConformerEncoderRelPosV3(StreamableModule):
    """
    TODO
    """
    def __init__(self, cfg: StreamableConformerEncoderRelPosV3Config, out_dim: int):
        super().__init__()

        self.feature_extraction = StreamableFeatureExtractorV1(
            cfg=cfg.feature_extraction_config,
            specaug_cfg=cfg.specaug_config,
            specaug_start_epoch=cfg.specauc_start_epoch
        )
        self.frontend = cfg.frontend()
        self.module_list = nn.ModuleList(
            [StreamableConformerBlockRelPosV1(cfg.block_cfg, dual_mode=cfg.dual_mode) for _ in range(cfg.num_layers)]
        )
        self.final_linear = nn.Linear(cfg.block_cfg.ff_cfg.input_dim, out_dim)

    def forward_offline(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        """
        data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len)
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        out = self.final_linear(x)

        return (x, out), torch.sum(sequence_mask, dim=1)

    def forward_streaming(
            self,
            raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            chunk_size: float, lookahead_size: int, carry_over_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
        """
        data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len, chunk_size)

        batch_sz, num_chunks, _, _ = data_tensor.shape

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
            x = module(
                x, sequence_mask, 
                attn_mask=attn_mask,
                lookahead_size=lookahead_size, 
                carry_over_size=carry_over_size
            )  # [B, T, F'] or [B, N, C'+R, F']

        # remove lookahead frames from every chunk
        if lookahead_size > 0:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        x = x.flatten(1, 2)  # [B, C'*N, F']
        sequence_mask = sequence_mask.flatten(1, 2)  # [B, C'*N]

        out = self.final_linear(x)

        return (x, out), torch.sum(sequence_mask, dim=1)

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        """
        if self._mode != Mode.STREAMING:
            raise NotImplementedError("Expected model to be in streaming mode for streaming inference.")

        chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
        audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)
        # [P, C] where first P is current chunk and rest is future ac. ctx.
        sequence_mask = mask_tensor(tensor=audio_features, seq_len=audio_features_len)  # (1, P*C)

        audio_features = audio_features.view(-1, chunk_size_frames, audio_features.size(-1))  # (P, C, F)
        sequence_mask = sequence_mask.view(-1, chunk_size_frames)  # (P, C)

        x, sequence_mask = self.frontend(audio_features, sequence_mask)  # (P, C', F')

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

        x = self.final_linear(x)

        return x, torch.sum(sequence_mask, dim=1), layer_outs
