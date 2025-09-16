import torch
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from torch import nn
from torchaudio.models import RNNT

from ...streamable_module import StreamableModule
from ...common import Mode, Hypothesis
from ...search._base_decoder import DecoderConfig, ExtraConfig, BaseDecoderModule
from ...base_config import BaseConfig

from .documented_rnnt_beam_search import RNNTBeamSearch



class Transcriber(nn.Module):
    def __init__(
        self,
        encoder: StreamableModule,
        chunk_size: Optional[int] = None,
        lookahead_size: Optional[int] = None,
        carry_over_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.encoder = encoder

        self.chunk_size = chunk_size
        self.lookahead_size = lookahead_size
        self.carry_over_size = carry_over_size

    def forward(self, input: torch.Tensor, lengths: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param input: audio samples as [B, T, 1]
        :param lengths: length of T as [B]
        :return:
        """
        with torch.no_grad():
            # self.encoder.set_mode_cascaded(Mode.OFFLINE)
            (_, encoder_out), encoder_out_lengths = self.encoder(input, lengths)

            return encoder_out, encoder_out_lengths

    def infer(
        self,
        input: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[List[List[torch.Tensor]]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio samples as [B=1, T, 1]
        :param lengths: length of T as [B=1]
        """
        if self.chunk_size is None:
            output, out_lengths = self.forward(input, lengths)
            return output, out_lengths, [[]]

        assert input.dim() == 3 and input.size(0) == 1, "Streaming inference expects input with shape [B=1, S, 1]."

        with torch.no_grad():
            # self.encoder.set_mode_cascaded(Mode.STREAMING)
            encoder_out, encoder_out_lengths, state = self.encoder.infer(
                input, lengths, states, chunk_size=self.chunk_size, lookahead_size=self.lookahead_size
            )

        return encoder_out[:, : encoder_out_lengths[0]], encoder_out_lengths, [state]



@dataclass(kw_only=True)
class RNNTSearchConfig(BaseConfig):
    """
    Given to RNNTDecoder in addition to DecoderConfig from `decoder_module`.
    DecoderConfig defines the "basic" parameters relevant for `decoder_module` while the
    SearchConfig has parameters relevant for the specific decoding algorithm.
    """
    # LM vars e.g. "lm.lstm.some_lstm_variant_file.Model"
    lm_module: Optional[str]
    lm_model_args: Optional[Dict[str, Any]]
    lm_checkpoint: Optional[Union[str, Any]]
    lm_scale: float = 0.0
    zero_ilm_scale: float = 0.0
    lm_package: Optional[str] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return cls(**d)

    def module(self):
        return RNNTDecoder


class RNNTDecoder(BaseDecoderModule):
    """
    Wrapper for RNNTBeamSearch as interface for `decoder_module`.
    """
    def __init__(self, run_ctx):
        super().__init__(run_ctx)
        self.search_config: RNNTSearchConfig = None
        self.decoder = None

        self.lm = None

    def init_decoder(self, decoder_config: DecoderConfig, extra_config: ExtraConfig):
        """
        Initialize the model, LM and decoding algorithm.
        """
        super().init_decoder(decoder_config=decoder_config, extra_config=extra_config)
        self.search_config = decoder_config.search_config

        self.subs_chunk_size = math.ceil(decoder_config.chunk_size / 16e3 * 1000 / 60)
        model: StreamableModule = self.run_ctx.engine._model
        model.set_mode_cascaded(decoder_config.mode)
        self.blank = model.cfg.label_target_size

        rnnt_model = RNNT(
            transcriber=Transcriber(
                encoder=model.encoder,
                chunk_size=decoder_config.chunk_size,
                lookahead_size=model.lookahead_size,
                carry_over_size=decoder_config.carry_over_size,
            ),
            predictor=model.predictor,
            joiner=model.joiner,
        )
        self.model = rnnt_model
        
        # load LM 
        if self.search_config.lm_module is not None:
            assert self.search_config.lm_package is not None
            lm_module_prefix = ".".join(self.search_config.lm_module.split(".")[:-1])
            lm_module_class = self.search_config.lm_module.split(".")[-1]

            LmModule = __import__(
                ".".join([self.search_config.lm_package, lm_module_prefix]),
                fromlist=[lm_module_class],
            )
            LmClass = getattr(LmModule, lm_module_class)

            lm_model = LmClass(**self.search_config.lm_model_args)
            checkpoint_state = torch.load(
                self.search_config.lm_checkpoint,
                map_location=self.run_ctx.device,
            )
            lm_model.load_state_dict(checkpoint_state["model"])
            lm_model.to(device=self.run_ctx.device)
            lm_model.eval()
            self.lm = lm_model

            print("loaded external LM")

        self.decoder = self._build_decoder()
    
    def _build_decoder(self):
        """
        Allows RNN-T type architectures (e.g. mRNN-T) to inherit RNNTDecoder and override this function for different beam-search algorithm.
        """
        if self.model is None or self.blank is None:
            raise ValueError

        return RNNTBeamSearch(
            model=self.model,
            blank=self.blank,
            blank_penalty=self.decoder_config.blank_log_penalty,
            device=self.run_ctx.device,
            lm_model=self.lm,
            lm_scale=self.search_config.lm_scale,
            zero_ilm_scale=self.search_config.zero_ilm_scale,
            lm_sos_token_index=0,
        )
    
    def _step(
            self, audio_chunk: torch.Tensor, chunk_len: torch.Tensor, 
    ) -> Tuple[List[Hypothesis], List[List[torch.Tensor]]]:
        """
        Do streaming (incremental) decoding on audio chunk w.r.t. current state and hypotheses. 
        """
        new_hypotheses, state = self.decoder.infer(
            input=audio_chunk,
            length=chunk_len,
            beam_width=self.decoder_config.beam_size,
            state=tuple(self.states) if len(self.states) > 0 else None,
            hypothesis=self.hypotheses,
        )

        return new_hypotheses, state

    def get_text(self, hypothesis: Hypothesis) -> str:
        sequence = [self.run_ctx.labels[idx] for idx in hypothesis.tokens if idx not in [self.sos, self.blank]]
        text = " ".join(sequence).replace("@@ ", "")

        return text
    
    def __call__(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor) -> List[Hypothesis]:
        """
        Full offline decoding of raw audio.
        """
        hypotheses, _ = self.decoder.infer(
            input=raw_audio,
            length=raw_audio_len,
            beam_width=self.decoder_config.beam_size,
        )
        return hypotheses
