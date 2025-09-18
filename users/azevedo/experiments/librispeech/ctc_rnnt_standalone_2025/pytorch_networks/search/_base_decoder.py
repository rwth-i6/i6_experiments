import copy
import collections
import math
import torch
from torch import Tensor
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from ..base_config import BaseConfig
from ..common import Hypothesis, Mode


@dataclass
class DecoderConfig:
    # files
    returnn_vocab: Union[str, Any]

    # search related options:
    beam_size: Optional[int]

    # defined in each architecture (rnnt, ctc) for each algorithm (e.g. beam search, greedy etc)
    search_config: BaseConfig

    # streaming definitions (mode == Mode.STREAMING)
    mode: Union[Mode, str] = None
    chunk_size: Optional[int] = None
    carry_over_size: Optional[float] = None
    lookahead_size: Optional[int] = None
    stride: Optional[int] = None

    pad_value: Optional[float] = None

    # for new hash
    test_version: Optional[float] = None

    eos_penalty: Optional[float] = None

    # prior correction
    blank_log_penalty: Optional[float] = None

    # batched encoder config
    batched_encoder = False

    # extra compile options
    use_torch_compile: bool = False
    torch_compile_options: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, d):
        d = copy.deepcopy(d)
        d["mode"] = Mode[d["mode"]]
        d["search_config"] = BaseConfig.load_config(d["search_config"])
        return cls(**d)

@dataclass
class ExtraConfig:
    # used for RTF logging
    print_rtf: bool = True
    sample_rate: int = 16000

    # Hypothesis logging
    print_hypothesis: bool = True


class BaseDecoderModule():
    """Inspired by torchaudio.models.decoder._ctc_decoder"""
    def __init__(self, run_ctx):
        self.run_ctx = run_ctx
        self.model = None

        self.blank = None
        self.sos = 0

        self.decoder_config: DecoderConfig = None
        self.extra_config: ExtraConfig = None

        self.hypotheses: Optional[List[Hypothesis]] = None
        self.states: Optional[List[List[Tensor]]] = None

    def init_decoder(self, decoder_config: DecoderConfig, extra_config: ExtraConfig, **kwargs):
        self.decoder_config = decoder_config
        self.extra_config = extra_config

        if decoder_config.mode != Mode.OFFLINE:
            self.states = collections.deque(maxlen=math.ceil(self.decoder_config.carry_over_size))
        self.hypotheses = None
    
    def reset(self):
        if self.decoder_config.mode != Mode.OFFLINE:
            self.states = collections.deque(maxlen=math.ceil(self.decoder_config.carry_over_size))
        self.hypotheses = None

        self._reset()

    def _reset(self):
        raise NotImplementedError

    def step(self, audio_chunk: Tensor, chunk_len: int) -> Tuple[List[Hypothesis], Tensor]:
        self.hypotheses, state = self._step(audio_chunk=audio_chunk, chunk_len=torch.tensor(chunk_len))
        self.states.append(state)

        return self.hypotheses, state
    
    def _step(self, audio_chunk: Tensor, chunk_len: int) -> Tuple[List[Hypothesis], List[Tensor]]:
        raise NotImplementedError
    
    def get_final_hypotheses(self):
        raise NotImplementedError
    
    def get_text(self, hyp, **kwargs) -> str:
        raise NotImplementedError

    def __call__(self, raw_audio, raw_audio_len, **kwargs):
        raise NotImplementedError
