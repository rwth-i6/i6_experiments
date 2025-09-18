# Defines the base class for all streamable rnnt"-eque" models (e.g. Standard RNNT, Monotonic RNNT, ...).
# Models to be trained with RETURNN are defined in .models/ and import StreamableRNNT as Model as well as define
# a train_step function

import torch
from typing import Dict, Union, Optional, Any
from dataclasses import dataclass

from i6_models.config import ModuleFactoryV1
from i6_models.parts.dropout import BroadcastDropout

from ..encoders._base_encoder import StreamableEncoder, StreamableEncoderConfig
from .predictors.base_predictor import Predictor

from ..base_config import BaseConfig
from ..streamable_module import StreamableModule

from ..trainers import train_handler



@dataclass(kw_only=True)
class StreamableRNNTConfig(BaseConfig):
    """
    """
    encoder: StreamableEncoderConfig
    predictor: BaseConfig
    joiner: BaseConfig

    encoder_joiner_dim: int
    label_target_size: Union[int, Any]

    chunk_size: Optional[float]  # in #samples
    lookahead_size: Optional[int]  # in #frames after frontend subsampling
    carry_over_size: Optional[int]  # in #chunks after frontend subsampling
    dual_mode: Optional[bool]
    streaming_scale: Optional[float]

    train_mode: Union[str, train_handler.TrainMode]
    ctc_output_loss: float

    def module(self):
        return StreamableRNNT

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        
        d["encoder"] = BaseConfig.load_config(d["encoder"])
        d["predictor"] = BaseConfig.load_config(d["predictor"])
        d["joiner"] = BaseConfig.load_config(d["joiner"])
        d["train_mode"] = {str(strat): strat for strat in train_handler.TrainMode}[d["train_mode"]]

        return cls(**d)


class StreamableRNNT(StreamableModule):
    def __init__(self, model_config_dict: Dict, **kwargs):
        super().__init__()

        self.cfg: StreamableRNNTConfig = StreamableRNNTConfig.from_dict(model_config_dict)

        self.encoder: StreamableEncoder = ModuleFactoryV1(
            module_class=self.cfg.encoder.module(), 
            cfg=self.cfg.encoder
        )()
        self.encoder_out_linear = torch.nn.Linear(self.cfg.encoder.encoder_size, self.cfg.encoder_joiner_dim)

        self.predictor: Predictor = ModuleFactoryV1(
            module_class=self.cfg.predictor.module(), cfg=self.cfg.predictor
        )()

        self.joiner: StreamableModule = ModuleFactoryV1(
            module_class=self.cfg.joiner.module(), cfg=self.cfg.joiner
        )()

        if self.cfg.ctc_output_loss > 0:
            self.encoder_ctc_on = torch.nn.Linear(self.cfg.encoder.encoder_size, self.cfg.label_target_size + 1)
            if self.cfg.dual_mode:
                self.encoder_ctc_off = torch.nn.Linear(self.cfg.encoder.encoder_size, self.cfg.label_target_size + 1)
            else:
                self.encoder_ctc_off = self.encoder_ctc_on

        self.chunk_size = self.cfg.chunk_size
        self.lookahead_size = self.cfg.lookahead_size
        self.carry_over_size = self.cfg.carry_over_size

    def forward_offline(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        encoder_out, encoder_out_lengths = self.encoder(raw_audio, raw_audio_len)
        encoder_out_joiner_in = self.encoder_out_linear(encoder_out)

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, _ = self.joiner(
            source_encodings=encoder_out_joiner_in,
            source_lengths=encoder_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc_off(encoder_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs
    
    def forward_streaming(
            self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor,
            labels: torch.Tensor, labels_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :param labels: [B, N]
        :param labels_len: length of N as [B]
        :return: logprobs [B, T + N, #labels + blank]
        """
        encoder_out, encoder_out_lengths = self.encoder(
            raw_audio, raw_audio_len,
            chunk_size=self.chunk_size,
            lookahead_size=self.lookahead_size,
            carry_over_size=self.carry_over_size
        ) # [B, C'*N, F'], [B, C'*N]
        encoder_out_joiner_in = self.encoder_out_linear(encoder_out)

        predict_out, _, _ = self.predictor(
            input=labels,
            lengths=labels_len,
        )

        output_logits, src_len, _ = self.joiner(
            source_encodings=encoder_out_joiner_in,
            source_lengths=encoder_out_lengths,
            target_encodings=predict_out,
            target_lengths=labels_len,
        )  # output is [B, T, N, #vocab]

        if self.cfg.ctc_output_loss > 0:
            ctc_logprobs = torch.log_softmax(self.encoder_ctc_on(encoder_out), dim=-1)
        else:
            ctc_logprobs = None

        return output_logits, src_len, ctc_logprobs
