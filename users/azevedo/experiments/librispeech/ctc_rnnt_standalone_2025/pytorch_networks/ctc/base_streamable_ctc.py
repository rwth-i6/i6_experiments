import torch
from typing import Dict, Union, Optional, Any, Tuple, List
from dataclasses import dataclass

from i6_models.config import ModuleFactoryV1
from i6_models.parts.dropout import BroadcastDropout

from ..encoders.base_encoder import StreamableEncoder, StreamableEncoderConfig
from ..base_config import BaseConfig
from ..streamable_module import StreamableModule
from ..trainers import train_handler
from ..common import Mode


@dataclass(kw_only=True)
class StreamableCTCConfig(BaseConfig):
    """
    """
    encoder: StreamableEncoderConfig

    label_target_size: Union[int, Any]  # TODO: maybe infer this in Model from some abstract Joiner config (abstract config should have output_dim)
    final_dropout: float

    chunk_size: Optional[float]  # in #samples
    lookahead_size: Optional[int]  # in #frames after frontend subsampling
    carry_over_size: Optional[int]  # in #chunks after frontend subsampling
    dual_mode: Optional[bool]
    streaming_scale: Optional[float]

    train_mode: Union[str, train_handler.TrainMode]

    def module(self):
        return StreamableCTC

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        
        d["encoder"] = BaseConfig.load_config(d["encoder"])
        d["train_mode"] = {str(strat): strat for strat in train_handler.TrainMode}[d["train_mode"]]

        return cls(**d)
    

class StreamableCTC(StreamableModule):
    def __init__(self, model_config_dict: Dict, **kwargs):
        super().__init__()

        self.cfg: StreamableCTCConfig = StreamableCTCConfig.from_dict(model_config_dict)

        self.encoder: StreamableEncoder = ModuleFactoryV1(
            module_class=self.cfg.encoder.module(),
            cfg=self.cfg.encoder
        )()

        self.final_dropout = BroadcastDropout(
            p=self.cfg.final_dropout,
            dropout_broadcast_axes=None,
        )

        self.chunk_size = self.cfg.chunk_size
        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size


    def forward_offline(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        (_, encoder_out), encoder_out_lengths = self.encoder(raw_audio, raw_audio_len)
        logits = encoder_out # = self.final_dropout(encoder_out)

        return logits, encoder_out_lengths

    def forward_streaming(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        (_, encoder_out), encoder_out_lengths = self.encoder(
            raw_audio, raw_audio_len,
            chunk_size=self.chunk_size,
            lookahead_size=self.lookahead_size,
            carry_over_size=self.carry_over_size,
        )  # [B, C'*N, F'], [B, C'*N]
        # logits = self.final_dropout(encoder_out)
        logits = encoder_out

        return logits, encoder_out_lengths

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio samples as [B=1, T, 1]
        :param lengths: length of T as [B=1]
        :param states:
        :param chunk_size:
        :param lookahead_size:
        """
        # NOTE: for ctc we call model explicitely => no if-clause like in rnnt decoder
        assert chunk_size is not None and lookahead_size is not None
        assert input.dim() == 3 and input.size(0) == 1, "Streaming inference expects input with shape [B=1, S, 1]."

        with torch.no_grad():
            encoder_out, encoder_out_lengths, state = self.encoder.infer(
                input, lengths, states, chunk_size=chunk_size, lookahead_size=lookahead_size
            )
        
        # return encoder_out[:, :encoder_out_lengths[0]], encoder_out_lengths, [state]
        return encoder_out, encoder_out_lengths, [state]