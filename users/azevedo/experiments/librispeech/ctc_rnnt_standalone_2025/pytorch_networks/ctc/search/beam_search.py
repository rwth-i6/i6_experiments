import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from torchaudio.models.decoder import CTCDecoderLM, CTCDecoderLMState

from sisyphus import tk

from .._base_streamable_ctc import StreamableCTC
from ...common import CTCHypothesis, _Hypothesis
from ...search._base_decoder import DecoderConfig, ExtraConfig, BaseDecoderModule
from ...base_config import BaseConfig


class CustomLM(CTCDecoderLM):
    def __init__(self, language_model: torch.nn.Module):
        CTCDecoderLM.__init__(self)
        self.language_model = language_model
        self.sil = -1
        self.states = {}

        self.language_model.eval()

    def start(self, start_with_nothing: bool = False):
        state = CTCDecoderLMState()
        with torch.no_grad():
            score = self.language_model(self.sil)

        self.states[state] = score
        return state

    def score(self, state: CTCDecoderLMState, token_index: int):
        outstate = state.child(token_index)
        if outstate not in self.states:
            score = self.language_model(token_index)
            self.states[outstate] = score
        score = self.states[outstate]

        return outstate, score

    def finish(self, state: CTCDecoderLMState):
        return self.score(state, self.sil)


@dataclass(kw_only=True)
class CTCSearchConfig(BaseConfig):
    """
    Given to CTCDecoder in addition to DecoderConfig from `decoder_module`.
    DecoderConfig defines the "basic" parameters relevant for `decoder_module` while the
    SearchConfig has parameters relevant for the specific decoding algorithm.
    """
    lexicon: Union[str, tk.Path]

    # search options
    beam_size_token: int
    beam_threshold: float
    sil_score: float = 0.0
    word_score: float = 0.0

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
        self.decoder = None

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

        from returnn.util.basic import cf
        if self.search_config.lm_package is not None:
            print("initializing arpa lm...")
            lm = cf(self.search_config.lm_package)
        else:
            lm = None

        self.lm = lm  # lm_model

        print("loaded external LM")

        if self.search_config.prior_file:
            self.run_ctx.prior = np.loadtxt(self.search_config.prior_file, dtype="float32")
            self.run_ctx.prior_scale = self.search_config.prior_scale
        else:
            self.run_ctx.prior = None

        self.run_ctx.blank_log_penalty = self.decoder_config.blank_log_penalty

        self.decoder = self._build_decoder()
        self.decoder.decode_begin()

    def _build_decoder(self):
        """
        Allows CTC type architectures CTCDecoder and override this function for different beam-search algorithm.
        """
        if self.model is None or self.blank is None:
            raise ValueError

        from torchaudio.models.decoder import ctc_decoder
        return ctc_decoder(
            lexicon=self.search_config.lexicon,
            lm=self.lm,
            lm_weight=self.search_config.lm_scale,
            tokens=self.run_ctx.labels + ["[blank]"],
            blank_token="[blank]",
            sil_token="[blank]",
            unk_word="[unknown]",
            nbest=1,
            beam_size=self.decoder_config.beam_size,
            beam_size_token=self.search_config.beam_size_token,
            beam_threshold=self.search_config.beam_threshold,
            sil_score=self.search_config.sil_score,
            word_score=self.search_config.word_score,
        )

    def _reset(self):
        self.decoder.decode_begin()

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

        logprobs_cpu, audio_features_len_cpu = logprobs.cpu(), audio_features_len.cpu()
        if self.run_ctx.blank_log_penalty is not None:
            # assumes blank is last
            logprobs_cpu[:, :, -1] -= self.run_ctx.blank_log_penalty
        if self.run_ctx.prior is not None:
            logprobs_cpu -= self.run_ctx.prior_scale * self.run_ctx.prior

        self.decoder.decode_step(logprobs_cpu[0, : audio_features_len_cpu[0]])

        # only returns non-empty list of we do decode_end() beforehand...
        # self.decoder.decode_end()
        hypotheses = self.decoder.get_final_hypothesis()
        # hypotheses = [_Hypothesis(tokens=h.tokens.tolist(), alignment=h.timesteps.tolist()) for h in hypotheses]

        return [], state

    def get_final_hypotheses(self):
        self.decoder.decode_end()
        hypotheses = self.decoder.get_final_hypothesis()  # returns current nbest (=1) hypos
        hypotheses = [_Hypothesis(tokens=h.tokens.tolist(), alignment=h.timesteps.tolist()) for h in hypotheses]
        return hypotheses

    def get_text(self, hypothesis: _Hypothesis) -> str:
        words = self.decoder.idxs_to_tokens(torch.LongTensor(hypothesis.tokens))
        text = " ".join(words).replace("@@ ", "")

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

        logprobs_cpu = logprobs.cpu()
        if self.run_ctx.blank_log_penalty is not None:
            # assumes blank is last
            logprobs_cpu[:, :, -1] -= self.run_ctx.blank_log_penalty
        if self.run_ctx.prior is not None:
            logprobs_cpu -= self.run_ctx.prior_scale * self.run_ctx.prior

        hypotheses = self.decoder(logprobs_cpu, audio_features_len.cpu())
        # convert CTCHypothesis (torch) to Hypothesis (ours)
        # print(hypotheses)
        hypotheses = [_Hypothesis(tokens=h.tokens.tolist(), alignment=h.timesteps.tolist()) for h in hypotheses[0]]
        return hypotheses
