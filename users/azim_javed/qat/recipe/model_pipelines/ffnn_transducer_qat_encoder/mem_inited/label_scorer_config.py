__all__ = ["get_ffnn_transducer_label_scorer_config"]

from typing import Union

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from .pytorch_modules import FFNNTransducerQATEncoderConfig, FFNNTransducerQATEncoderRecogConfig, FFNNTransducerQATEncoderScorer
from ....experiments.librispeech.training.ffnn_transducer_qat_encoder_bpe import get_model_config

def _get_block_cfg(model_config: FFNNTransducerQATEncoderConfig):
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


def get_ffnn_transducer_label_scorer_config(
    model_config: FFNNTransducerQATEncoderConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
    max_batch_size: int = 2048,
) -> RasrConfig:

    label_scorer_type = "fixed-context-py"
    # NOTE: `scale` is currently not applied. It is written under `recognition.scale`, but RASR's
    # ScaledLabelScorer reads `scale` from the label scorer section root and falls back to 1.0.
    assert scale == 1.0, "scaling is not supported for the label scorer"

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
        Import(f"{FFNNTransducerQATEncoderScorer.__module__}.{FFNNTransducerQATEncoderScorer.__name__}", import_as="ScorerModel"),
        Import(f"{FFNNTransducerQATEncoderRecogConfig.__module__}.{FFNNTransducerQATEncoderRecogConfig.__name__}", import_as="RecogConfig"),
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