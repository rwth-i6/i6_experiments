__all__ = ["get_ffnn_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from .pytorch_modules import FFNNTransducerQATEncoderPredictionConfig, FFNNTransducerQATEncoderPredictionRecogConfig, FFNNTransducerQATEncoderPredictionScorer
from ...experiments.librispeech.training.ffnn_transducer_qat_encoder_prediction_bpe import get_model_config

def get_ffnn_transducer_label_scorer_config(
    model_config: FFNNTransducerQATEncoderPredictionConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
    max_batch_size: int = 2048,
) -> RasrConfig:

    label_scorer_type = "fixed-context-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.history_length = model_config.context_history_size
    rasr_config.start_label_index = model_config.target_size - 1
    rasr_config.max_batch_size = max_batch_size

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.scale = scale
    rasr_config.recognition.model_path = checkpoint
    rasr_config.recognition.device = "cuda" if use_gpu else "cpu"
    imports = [
        Import(f"{FFNNTransducerQATEncoderPredictionScorer.__module__}.{FFNNTransducerQATEncoderPredictionScorer.__name__}", import_as="ScorerModel"),
        Import(f"{FFNNTransducerQATEncoderPredictionRecogConfig.__module__}.{FFNNTransducerQATEncoderPredictionRecogConfig.__name__}", import_as="RecogConfig"),
        Import(f"{get_model_config.__module__}.{get_model_config.__name__}", import_as="get_model_config"),
    ]
    rasr_config.recognition.imports = " ".join(
        imp.get().strip() if hasattr(imp, "get") else str(imp).strip() for imp in imports
    )

    # TODO: wire these through `run``
    rasr_config.recognition.qat.weight_bit_prec = 8
    rasr_config.recognition.qat.activation_bit_prec = 8
    rasr_config.recognition.qat.weight_dropout = 0.0
    rasr_config.recognition.qat.weight_pruning_config = None

    return rasr_config