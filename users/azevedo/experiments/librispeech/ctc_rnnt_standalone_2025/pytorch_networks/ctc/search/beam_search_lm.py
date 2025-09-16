import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

from sisyphus import tk

from .._base_streamable_ctc import StreamableCTC
from ...common import _Hypothesis
from ...search._base_decoder import DecoderConfig, ExtraConfig, BaseDecoderModule
from ...base_config import BaseConfig

from .documented_ctc_beam_search_v4 import CTCBeamSearch, Hypothesis


@dataclass(kw_only=True)
class CTCSearchConfig(BaseConfig):
    """
    Given to CTCDecoder in addition to DecoderConfig from `decoder_module`.
    DecoderConfig defines the "basic" parameters relevant for `decoder_module` while the
    SearchConfig has parameters relevant for the specific decoding algorithm.
    """
    # prior correction
    blank_log_penalty: Optional[float] = None
    prior_scale: float = 0.0
    prior_file: Optional[Union[str, Any]] = None

    # LM vars e.g. "lm.lstm.some_lstm_variant_file.Model"
    lm_scale: float = 0.0
    zero_ilm_scale: float = 0.0
    lm_module: Optional[str] = None
    lm_model_args: Optional[Dict[str, Any]] = None
    lm_checkpoint: Optional[Union[str, Any]] = None
    lm_package: Optional[Union[str, Any]] = None

    lm_states_need_label_axis: bool = False

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return cls(**d)

    def module(self):
        return CTCDecoder


class CTCDecoder(BaseDecoderModule):
    """
    Wrapper of torchaudio.models.decode.ctc_decoder as interface for `decoder_module`.
    """

    def __init__(self, run_ctx):
        super().__init__(run_ctx)
        self.search_config: CTCSearchConfig = None
        self.decoder: CTCBeamSearch = None

        self.lm = None

    def init_decoder(self, decoder_config: DecoderConfig, extra_config: ExtraConfig, **kwargs):
        """
        Initialize the model, LM and decoding algorithm.
        """
        super().init_decoder(decoder_config=decoder_config, extra_config=extra_config)
        self.search_config: CTCSearchConfig = decoder_config.search_config

        # self.subs_chunk_size = math.ceil(decoder_config.chunk_size / 16e3 * 1000 / 60)
        model: StreamableCTC = self.run_ctx.engine._model
        model.set_mode_cascaded(decoder_config.mode)
        self.model = model
        self.blank = model.cfg.label_target_size

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

        if self.search_config.prior_file:
            self.run_ctx.prior = torch.tensor(np.loadtxt(self.search_config.prior_file, dtype="float32"), device=self.run_ctx.device)
            self.run_ctx.prior_scale = self.search_config.prior_scale
        else:
            self.run_ctx.prior = None

        self.run_ctx.blank_log_penalty = self.decoder_config.blank_log_penalty

        self.decoder = self._build_decoder()

    def _build_decoder(self):
        """
        Allows CTC type architectures CTCDecoder and override this function for different beam-search algorithm.
        """
        if self.model is None or self.blank is None:
            raise ValueError

        return CTCBeamSearch(
            model=self.model,
            blank=self.model.cfg.label_target_size,
            device=self.run_ctx.device,
            lm_model=self.lm,
            lm_scale=self.search_config.lm_scale,
            lm_sos_token_index=0,
            lm_states_need_label_axis=self.search_config.lm_states_need_label_axis,
            prior=self.run_ctx.prior,
            prior_scale=self.run_ctx.prior_scale,
        )

    def _reset(self):
        return

    def _step(
            self, audio_chunk: torch.Tensor, chunk_len: torch.Tensor,
    ) -> Tuple[List[Hypothesis], List[List[torch.Tensor]]]:
        """
        Do streaming (incremental) decoding on audio chunk w.r.t. current state and hypotheses.
        """
        # init generator for chunks of our raw_audio according to DecoderConfig
        logprobs, audio_features_len, state = self.model.infer(
            input=audio_chunk.unsqueeze(0),
            lengths=chunk_len.unsqueeze(0),
            states=tuple(self.states) if len(self.states) > 0 else None,
            chunk_size=self.decoder_config.chunk_size,
            lookahead_size=self.model.lookahead_size,
        )

        new_hypotheses = self.decoder.infer(
            enc_out=logprobs[:, :audio_features_len[0]],  # NOTE: can do this because bsz = 1
            length=audio_features_len,
            beam_width=self.decoder_config.beam_size,
            hypothesis=self.hypotheses
        )
        
        return new_hypotheses, state

    def get_final_hypotheses(self):
        hypotheses: List[Hypothesis] = self.hypotheses
        _hypotheses = [_Hypothesis(tokens=h.tokens[1:], alignment=h.alignment[1:]) for h in hypotheses]
        return _hypotheses

    def get_text(self, hypothesis: _Hypothesis, **kwargs) -> str:
        sequence = [self.run_ctx.labels[idx] for idx in hypothesis.tokens if idx < len(self.run_ctx.labels)]
        sequence = [s for s in sequence if (not s.startswith("<") and not s.startswith("["))]
        text = " ".join(sequence).replace("@@ ", "")

        return text

    def __call__(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor) -> List[_Hypothesis]:
        """
        Full offline decoding of raw audio.
        """
        logprobs, audio_features_len = self.model(
            raw_audio=raw_audio,
            raw_audio_len=raw_audio_len,
        )

        if isinstance(logprobs, list):
            logprobs = logprobs[-1]

        hypotheses = self.decoder.forward(logprobs, audio_features_len, self.decoder_config.beam_size)
        hypotheses = [_Hypothesis(tokens=h.tokens[1:], alignment=h.alignment[1:]) for h in hypotheses]
        return hypotheses
