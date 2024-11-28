from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence

from i6_core.returnn.config import CodeWrapper, ReturnnConfig
from sisyphus import tk

from ..data.bpe import DataConfig
from .pytorch_modules import FFNNTransducerConfig


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
    accum_grad_multiple_step: int
    weight_decay: float
    enc_loss_scales: Dict[int, float]
    lr_config: OCLRConfig
    gradient_clip: float


@dataclass
class CTCPriorRoutineConfig:
    enc_layer: int
    prior_data_config: DataConfig
    batch_frames: int


@dataclass
class RecogRoutineConfig:
    descriptor: str
    corpus_name: str
    recog_data_config: DataConfig
    epoch: int
    blank_penalty: float
    device: Literal["cpu", "gpu"]


@dataclass
class CTCRecogRoutineConfig(RecogRoutineConfig):
    prior_config: CTCPriorRoutineConfig
    prior_scale: float
    enc_layer: int


@dataclass
class CTCRasrGreedyRecogRoutineConfig(CTCRecogRoutineConfig):
    vocab_file: tk.Path


@dataclass
class CTCFlashlightRecogRoutineConfig(CTCRecogRoutineConfig):
    prior_config: CTCPriorRoutineConfig
    enc_layer: int
    vocab_file: tk.Path
    lexicon_file: Optional[tk.Path]
    lm_file: Optional[tk.Path]
    beam_size: int
    beam_size_token: Optional[int]
    beam_threshold: float
    lm_scale: float


@dataclass
class TransducerRecogRoutineConfig(RecogRoutineConfig):
    ilm_scale: float


@dataclass
class RasrGreedyRecogRoutineConfig(TransducerRecogRoutineConfig):
    vocab_file: tk.Path


@dataclass
class RasrBeamRecogRoutineConfig(TransducerRecogRoutineConfig):
    vocab_file: tk.Path
    max_beam_size: int
    top_k_tokens: Optional[int]
    score_threshold: Optional[float]


@dataclass
class PipelineConfig:
    model_config: FFNNTransducerConfig
    train_config: TrainRoutineConfig
    recog_configs: Sequence[RecogRoutineConfig]


@dataclass
class ResultItem:
    descriptor: str
    corpus_name: str
    wer: tk.Variable
    enc_rtf: tk.Variable
    search_rtf: tk.Variable
    total_rtf: tk.Variable
