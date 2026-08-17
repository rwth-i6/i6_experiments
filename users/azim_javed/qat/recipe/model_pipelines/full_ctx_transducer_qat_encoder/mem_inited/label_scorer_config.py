__all__ = ["get_lstm_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from ....experiments.librispeech.training.full_ctx_transducer_qat_encoder_bpe import get_model_config
from .pytorch_modules import (
    LstmTransducerQATEncoderConfig,
    LstmTransducerQATEncoderRecogConfig,
    LstmTransducerQATEncoderScorer,
    LstmTransducerQATEncoderStateInitializer,
    LstmTransducerQATEncoderStateUpdater,
)


def _get_block_cfg(model_config: LstmTransducerQATEncoderConfig):
    block_cfg = model_config.conformer_cfg.block_cfg
    if isinstance(block_cfg, list):
        assert block_cfg, "empty conformer block_cfg list"
        block_cfg = block_cfg[0]
    return block_cfg


def _serialize_dac_settings(hardware_settings) -> list:
    return [
        hardware_settings.input_bits,
        hardware_settings.output_precision_bits,
        hardware_settings.output_range_bits,
        hardware_settings.hardware_input_vmax,
        hardware_settings.hardware_output_current_scaling,
    ]


def get_lstm_transducer_label_scorer_config(
    model_config: LstmTransducerQATEncoderConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    use_gpu: bool = False,
    max_batch_size: int = 2048,
) -> RasrConfig:

    label_scorer_type = "stateful-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.start_label_index = model_config.target_size - 1
    rasr_config.max_batch_size = max_batch_size

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.model_path = checkpoint
    rasr_config.recognition.device = "cuda" if use_gpu else "cpu"
    rasr_config.recognition.experiment = "full_ctx_transducer_qat_encoder"

    imports = [
        Import(f"{LstmTransducerQATEncoderScorer.__module__}.{LstmTransducerQATEncoderScorer.__name__}", import_as="ScorerModel"),
        Import(f"{LstmTransducerQATEncoderStateInitializer.__module__}.{LstmTransducerQATEncoderStateInitializer.__name__}", import_as="StateInitializerModel"),
        Import(f"{LstmTransducerQATEncoderStateUpdater.__module__}.{LstmTransducerQATEncoderStateUpdater.__name__}", import_as="StateUpdaterModel"),
        Import(f"{LstmTransducerQATEncoderRecogConfig.__module__}.{LstmTransducerQATEncoderRecogConfig.__name__}", import_as="RecogConfig"),
        Import(f"{get_model_config.__module__}.{get_model_config.__name__}", import_as="get_model_config"),
    ]
    rasr_config.recognition.imports = " ".join(
        imp.get().strip() if hasattr(imp, "get") else str(imp).strip() for imp in imports
    )

    block_cfg = _get_block_cfg(model_config)
    ff_cfg = block_cfg.ff_cfg
    mhsa_cfg = block_cfg.mhsa_cfg

    weight_bit_prec = ff_cfg.weight_bit_prec
    if not isinstance(weight_bit_prec, (int, float)):
        raise NotImplementedError(
            "layerwise weight_bit_prec (dict) for the label scorer is not supported, got %r" % (weight_bit_prec,)
        )
    rasr_config.recognition.qat.weight_bit_prec = weight_bit_prec
    rasr_config.recognition.qat.activation_bit_prec = ff_cfg.activation_bit_prec
    rasr_config.recognition.qat.weight_dropout = ff_cfg.weight_dropout
    rasr_config.recognition.qat.weight_pruning_config = None

    memristor_config = rasr_config.recognition.memristor
    if ff_cfg.converter_hardware_settings is not None:
        memristor_config.converter_hardware_settings = _serialize_dac_settings(ff_cfg.converter_hardware_settings)
    if mhsa_cfg.pos_enc_converter_hardware_settings is not None:
        memristor_config.pos_enc_converter_hardware_settings = _serialize_dac_settings(
            mhsa_cfg.pos_enc_converter_hardware_settings
        )
    memristor_config.num_cycles = ff_cfg.num_cycles
    if ff_cfg.correction_settings is not None:
        correction_settings = ff_cfg.correction_settings
        memristor_config.correction_settings = [
            correction_settings.num_cycles if correction_settings.num_cycles is not None else "None",
            correction_settings.test_input_value if correction_settings.test_input_value is not None else "None",
            correction_settings.relative_deviation if correction_settings.relative_deviation is not None else "None",
            correction_settings.ideal_programming,
        ]

    return rasr_config