from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from sisyphus import tk

from ..data.bpe import DataConfig
from .pytorch_modules import ConformerCTCConfig


@dataclass
class OCLRConfig:
    init_lr: float
    peak_lr: float
    decayed_lr: float
    final_lr: float
    inc_epochs: int
    dec_epochs: int
    final_epochs: int

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            config={
                "learning_rates": CodeWrapper(
                    f"list(np.linspace({self.init_lr}, {self.peak_lr}, {self.inc_epochs}))"
                    f"+ list(np.linspace({self.peak_lr}, {self.decayed_lr}, {self.dec_epochs}))"
                    f"+ list(np.linspace({self.decayed_lr}, {self.final_lr}, {self.final_epochs}))"
                )
            },
            python_prolog=["import numpy as np"],
        )


@dataclass
class TrainRoutineConfig:
    train_data_config: DataConfig
    cv_data_config: DataConfig
    save_epochs: List[int]
    batch_frames: int
    weight_decay: float
    lr_config: OCLRConfig
    gradient_clip: float


@dataclass
class PriorRoutineConfig:
    prior_data_config: DataConfig
    batch_frames: int


@dataclass
class RecogRoutineConfig:
    descriptor: str
    corpus_name: str
    recog_data_config: DataConfig
    prior_config: PriorRoutineConfig
    epoch: int
    prior_scale: float
    blank_penalty: float
    device: Literal["cpu", "gpu"]


@dataclass
class RasrGreedyRecogRoutineConfig(RecogRoutineConfig):
    vocab_file: tk.Path


@dataclass
class RasrBeamRecogRoutineConfig(RecogRoutineConfig):
    vocab_file: tk.Path
    max_beam_size: int
    top_k_tokens: Optional[int]
    score_threshold: Optional[float]


@dataclass
class FlashlightRecogRoutineConfig(RecogRoutineConfig):
    vocab_file: tk.Path
    lexicon_file: Optional[tk.Path]
    lm_file: Optional[tk.Path]
    beam_size: int
    beam_size_token: Optional[int]
    beam_threshold: float
    lm_scale: float


@dataclass
class PipelineConfig:
    model_config: ConformerCTCConfig
    train_config: TrainRoutineConfig
    recog_configs: Sequence[RecogRoutineConfig]


@dataclass
class ResultItem:
    descriptor: str
    corpus_name: str
    wer: tk.Variable
    am_rtf: tk.Variable
    search_rtf: tk.Variable
    total_rtf: tk.Variable
