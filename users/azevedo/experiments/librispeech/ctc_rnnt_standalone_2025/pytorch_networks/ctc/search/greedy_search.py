import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

from sisyphus import tk

from .._base_streamable_ctc import StreamableCTC
from ...common import CTCHypothesis, _Hypothesis
from ...search._base_decoder import DecoderConfig, ExtraConfig, BaseDecoderModule
from ...base_config import BaseConfig


@dataclass(kw_only=True)
class CTCSearchConfig(BaseConfig):
    """
    Given to CTCDecoder in addition to DecoderConfig from `decoder_module`.
    DecoderConfig defines the "basic" parameters relevant for `decoder_module` while the
    SearchConfig has parameters relevant for the specific decoding algorithm.
    """
    lexicon: Union[str, tk.Path]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return cls(**d)

    def module(self):
        return CTCGreedyDecoder


class CTCGreedyDecoder(BaseDecoderModule):
    """
    Wrapper of torchaudio.models.decode.ctc_decoder as interface for `decoder_module`.
    """

    def __init__(self, run_ctx):
        super().__init__(run_ctx)
        self.search_config: CTCSearchConfig = None
        self.decoder = None

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

        self.run_ctx.blank_log_penalty = self.decoder_config.blank_log_penalty

    def _reset(self):
        return

    def _step(
            self, audio_chunk: torch.Tensor, chunk_len: torch.Tensor,
    ) -> Tuple[List[_Hypothesis], List[List[torch.Tensor]]]:
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

        hypothesis = CTCHypothesis([], [])
        if self.hypotheses is not None: 
            hypothesis = self.hypotheses[0]

        # select tokens greedily
        alignment = torch.argmax(logprobs[0, :audio_features_len[0]], dim=-1).detach().cpu()
        hypothesis.alignment.extend(alignment.tolist())
        # get unaligned tokens with blanks
        hypothesis.tokens.extend(torch.unique_consecutive(alignment, dim=0).tolist())

        return [hypothesis], state

    def get_final_hypotheses(self):
        return self.hypotheses

    def get_text(self, hypothesis: _Hypothesis) -> str:
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
        hypotheses = []
        # iterate over batches
        for lp, l in zip(logprobs, audio_features_len):
            alignment = torch.argmax(lp[:l], dim=-1).detach().cpu()
            tokens = torch.unique_consecutive(alignment, dim=0)
            hypo = CTCHypothesis(alignment=alignment.tolist(), tokens=tokens.tolist())
            hypotheses.append(hypo)

        return hypotheses