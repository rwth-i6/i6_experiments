__all__ = ["get_ffnn_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint


from .pytorch_modules import FFNNTransducerQATEncoderConfig


def get_ffnn_transducer_label_scorer_config(
    model_config: FFNNTransducerQATEncoderConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:

    label_scorer_type = "fixed-context-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.history_length = model_config.context_history_size
    rasr_config.start_label_index = model_config.target_size - 1

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.scale = scale # Unusued; TODO: ask
    rasr_config.recognition.model_path = checkpoint
    rasr_config.recognition.experiment = "ffnn_transducer_qat_encoder"

    rasr_config.qat = RasrConfig()
    rasr_config.qat.weight_bit_prec = 8
    rasr_config.qat.activation_bit_prec = 8
    rasr_config.qat.weight_dropout = 0.0
    rasr_config.qat.weight_pruning_config = None

    return rasr_config