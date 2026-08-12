__all__ = ["get_lstm_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from ...experiments.librispeech.training.full_ctx_transducer_bpe import get_model_config
from .pytorch_modules import (
    LstmTransducerConfig,
    LstmTransducerRecogConfig,
    LstmTransducerScorer,
    LstmTransducerStateInitializer,
    LstmTransducerStateUpdater,
)


def get_lstm_transducer_label_scorer_config(
    model_config: LstmTransducerConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:

    label_scorer_type = "stateful-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.start_label_index = model_config.target_size - 1

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.scale = scale
    rasr_config.recognition.model_path = checkpoint
    rasr_config.recognition.experiment = "full_ctx_transducer"

    imports = [
        Import(f"{LstmTransducerScorer.__module__}.{LstmTransducerScorer.__name__}", import_as="ScorerModel"),
        Import(f"{LstmTransducerStateInitializer.__module__}.{LstmTransducerStateInitializer.__name__}", import_as="StateInitializerModel"),
        Import(f"{LstmTransducerStateUpdater.__module__}.{LstmTransducerStateUpdater.__name__}", import_as="StateUpdaterModel"),
        Import(f"{LstmTransducerRecogConfig.__module__}.{LstmTransducerRecogConfig.__name__}", import_as="RecogConfig"),
        Import(f"{get_model_config.__module__}.{get_model_config.__name__}", import_as="get_model_config"),
    ]
    rasr_config.recognition["imports"] = " ".join(
        imp.get().strip() if hasattr(imp, "get") else str(imp).strip() for imp in imports
    )

    return rasr_config

