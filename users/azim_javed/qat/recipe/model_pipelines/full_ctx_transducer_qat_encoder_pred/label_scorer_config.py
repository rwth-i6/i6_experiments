__all__ = ["get_lstm_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from ...experiments.librispeech.training.full_ctx_transducer_qat_encoder_prediction_bpe import get_model_config
from .pytorch_modules import (
    LstmTransducerQATEncoderPredictionConfig,
    LstmTransducerQATEncoderPredictionRecogConfig,
    LstmTransducerQATEncoderPredictionScorer,
    LstmTransducerQATEncoderPredictionStateInitializer,
    LstmTransducerQATEncoderPredictionStateUpdater,
)


def get_lstm_transducer_label_scorer_config(
    model_config: LstmTransducerQATEncoderPredictionConfig,
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
    rasr_config.recognition.experiment = "full_ctx_transducer_qat_encoder"
    rasr_config.recognition.device = "cuda" if use_gpu else "cpu"

    imports = [
        Import(f"{LstmTransducerQATEncoderPredictionScorer.__module__}.{LstmTransducerQATEncoderPredictionScorer.__name__}", import_as="ScorerModel"),
        Import(f"{LstmTransducerQATEncoderPredictionStateInitializer.__module__}.{LstmTransducerQATEncoderPredictionStateInitializer.__name__}", import_as="StateInitializerModel"),
        Import(f"{LstmTransducerQATEncoderPredictionStateUpdater.__module__}.{LstmTransducerQATEncoderPredictionStateUpdater.__name__}", import_as="StateUpdaterModel"),
        Import(f"{LstmTransducerQATEncoderPredictionRecogConfig.__module__}.{LstmTransducerQATEncoderPredictionRecogConfig.__name__}", import_as="RecogConfig"),
        Import(f"{get_model_config.__module__}.{get_model_config.__name__}", import_as="get_model_config"),
    ]
    rasr_config.recognition.imports = " ".join(
        imp.get().strip() if hasattr(imp, "get") else str(imp).strip() for imp in imports
    )

    rasr_config.recognition.qat.weight_bit_prec = 8
    rasr_config.recognition.qat.activation_bit_prec = 8
    rasr_config.recognition.qat.weight_dropout = 0.0
    rasr_config.recognition.qat.weight_pruning_config = None

    return rasr_config