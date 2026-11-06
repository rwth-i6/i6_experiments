__all__ = ["get_lstm_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from ....experiments.librispeech.training.full_ctx_transducer_qat_encoder_prediction_bpe import get_model_config
from .pytorch_modules import (
    LstmTransducerQATEncoderPredictionConfig,
    LstmTransducerQATEncoderPredictionRecogConfig,
    LstmTransducerQATEncoderPredictionScorer,
    LstmTransducerQATEncoderPredictionStateInitializer,
    LstmTransducerQATEncoderPredictionStateUpdater,
)


def _serialize_dac_settings(hardware_settings) -> list:
    return [
        hardware_settings.input_bits,
        hardware_settings.output_precision_bits,
        hardware_settings.output_range_bits,
        hardware_settings.hardware_input_vmax,
        hardware_settings.hardware_output_current_scaling,
    ]


def get_lstm_transducer_label_scorer_config(
    model_config: LstmTransducerQATEncoderPredictionConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
    use_cache: bool = False,
) -> RasrConfig:

    label_scorer_type = "stateful-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.start_label_index = model_config.target_size - 1
    if use_cache:
        rasr_config.use_cache = use_cache

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

    memristor_config = rasr_config.recognition.memristor
    memristor_config.converter_hardware_settings = _serialize_dac_settings(
        model_config.converter_hardware_settings
    )
    memristor_config.pos_enc_converter_hardware_settings = _serialize_dac_settings(
        model_config.pos_enc_converter_hardware_settings
    )
    memristor_config.num_cycles = model_config.num_cycles
    if model_config.correction_settings is not None:
        correction_settings = model_config.correction_settings
        memristor_config.correction_settings = [
            correction_settings.num_cycles if correction_settings.num_cycles is not None else "None",
            correction_settings.test_input_value if correction_settings.test_input_value is not None else "None",
            correction_settings.relative_deviation if correction_settings.relative_deviation is not None else "None",
            correction_settings.ideal_programming,
        ]

    return rasr_config
