import torch
from torch import nn
from typing import Tuple, List, Optional, Union, Any
from dataclasses import dataclass

from i6_models.config import ModuleFactoryV1

from ..base_config import BaseConfig
from ..streamable_module import StreamableModule
from ..common import Mode, mask_tensor, create_chunk_mask, add_lookahead



@dataclass(kw_only=True)
class StreamableEncoderConfig(BaseConfig):
    feature_extractor: BaseConfig
    frontend: BaseConfig
    encoder_blocks: BaseConfig

    num_layers: int
    encoder_size: int
    out_dim: Union[int, Any]

    def module(self):
        return StreamableEncoder

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        
        d["feature_extractor"] = BaseConfig.load_config(d["feature_extractor"])
        d["frontend"] = BaseConfig.load_config(d["frontend"])
        d["encoder_blocks"] = BaseConfig.load_config(d["encoder_blocks"])

        return StreamableEncoderConfig(**d)


class StreamableEncoder(StreamableModule):
    """
    TODO
    """
    def __init__(self, config: Union[StreamableEncoderConfig, dict]):
        super().__init__()

        if isinstance(config, dict):
            config: StreamableEncoderConfig = StreamableEncoderConfig.load_config(config)

        self.feature_extraction: StreamableModule = ModuleFactoryV1(
            module_class=config.feature_extractor.module(), 
            cfg=config.feature_extractor
        )()
        self.frontend: StreamableModule = ModuleFactoryV1(
            module_class=config.frontend.module(),
            cfg=config.frontend
        )()
        enc_blocks_factory = ModuleFactoryV1(
            module_class=config.encoder_blocks.module(), 
            cfg=config.encoder_blocks
        )
        self.encoder_blocks: List[StreamableModule] = nn.ModuleList([enc_blocks_factory() for _ in range(config.num_layers)])

        self.final_linear = nn.Linear(config.encoder_blocks.ff_cfg.input_dim, config.out_dim)


    def forward_offline(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor, *args, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param _: 
        :return: _
        """
        data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len)
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']
        for module in self.encoder_blocks:
            x = module(x, sequence_mask)  # [B, T, F']

        out = self.final_linear(x)

        return (x, out), torch.sum(sequence_mask, dim=1)

    def forward_streaming(
            self,
            raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            chunk_size: float, lookahead_size: int, carry_over_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param _: 
        :return: _
        """
        data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len, chunk_size)

        batch_sz, num_chunks, _, _ = data_tensor.shape

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)

        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        # [B, N, C', F'] -> [B, N, C'+R, F']
        x, sequence_mask = add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

        attn_mask = create_chunk_mask(
            seq_len=(x.size(1) * x.size(2)),
            chunk_size=x.size(2) - lookahead_size,
            lookahead_size=lookahead_size,
            carry_over_size=carry_over_size,
            device=x.device
        )

        for module in self.encoder_blocks:
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
        :param _: 
        :return: _
        """
        assert self._mode == Mode.STREAMING, "Expected model to be in streaming mode for streaming inference."

        chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
        audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)
        # [P, C] where first P is current chunk and rest is future ac. ctx.
        #sequence_mask = mask_tensor(tensor=audio_features, seq_len=audio_features_len)  # (1, P*C)

        #audio_features = audio_features.view(-1, chunk_size_frames, audio_features.size(-1))  # (P, C, F)
        #sequence_mask = sequence_mask.view(-1, chunk_size_frames)  # (P, C)

        #x, sequence_mask = self.frontend(audio_features, sequence_mask)  # (P, C', F')
        x, sequence_mask = self.frontend.infer(audio_features, audio_features_len, chunk_size_frames)

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

        for i, module in enumerate(self.encoder_blocks):
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
