__all__ = ["get_ffnn_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from .pytorch_modules import (
    QATFFNNTransducerConfig,
    QATFFNNTransducerRecogConfig,
    QATFFNNTransducerJointADCRecogConfig,
    QATFFNNTransducerScorer,
)
from ....experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config

from synaptogen_ml.memristor_modules import DacAdcHardwareSettings


def _get_block_cfg(model_config: QATFFNNTransducerConfig):
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
    model_config: QATFFNNTransducerConfig,
    checkpoint: PtCheckpoint,
    converter_hardware_settings: DacAdcHardwareSettings,
    pos_enc_converter_hardware_settings: DacAdcHardwareSettings,
    num_cycles: int,
    joint_l1_converter_hardware_settings: DacAdcHardwareSettings = None,
    joint_l2_converter_hardware_settings: DacAdcHardwareSettings = None,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    debug_joint: bool = False,
    scale: float = 1.0,
    use_gpu: bool = False,
    max_batch_size: int = 2048,
    use_cache: bool = False,
    separated_joint_adc: bool = False,
    scorer_model_class=QATFFNNTransducerScorer,
    presoftmax_blank_penalty: bool = False,
    adapter_checkpoint: PtCheckpoint = None,
) -> RasrConfig:

    label_scorer_type = "fixed-context-py-cache" if use_cache else "fixed-context-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.history_length = model_config.context_history_size
    rasr_config.start_label_index = model_config.target_size - 1
    rasr_config.max_batch_size = max_batch_size

    if debug_joint:
        rasr_config.debug_joint = True

    if presoftmax_blank_penalty:
        rasr_config.presoftmax_blank_penalty = True

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.scale = scale
    rasr_config.recognition.model_path = checkpoint
    if adapter_checkpoint is not None:
        rasr_config.recognition.adapter_model_path = adapter_checkpoint
    rasr_config.recognition.device = "cuda" if use_gpu else "cpu"
    recog_cls_cls = QATFFNNTransducerRecogConfig if not separated_joint_adc else QATFFNNTransducerJointADCRecogConfig
    imports = [
        Import(f"{scorer_model_class.__module__}.{scorer_model_class.__name__}", import_as="ScorerModel"),
        Import(
            f"{recog_cls_cls.__module__}.{recog_cls_cls.__name__}",
            import_as="RecogConfig",
        ),
        Import(f"{get_model_config.__module__}.{get_model_config.__name__}", import_as="get_model_config"),
    ]
    rasr_config.recognition.imports = " ".join(
        imp.get().strip() if hasattr(imp, "get") else str(imp).strip() for imp in imports
    )

    block_cfg = _get_block_cfg(model_config)
    ff_cfg = block_cfg.ff_cfg

    weight_bit_prec = ff_cfg.weight_bit_prec
    if not isinstance(weight_bit_prec, (int, float)):
        raise NotImplementedError(
            "layerwise weight_bit_prec (dict) for the label scorer is not supported, got %r" % (weight_bit_prec,)
        )
    rasr_config.recognition.qat.weight_bit_prec = weight_bit_prec
    rasr_config.recognition.qat.activation_bit_prec = ff_cfg.activation_bit_prec
    rasr_config.recognition.qat.weight_dropout = ff_cfg.weight_dropout
    rasr_config.recognition.qat.weight_pruning_config = None
    if (model_config.version_control or "").endswith("-no-joiner-output-bias"):
        rasr_config.recognition.qat.version_control = model_config.version_control

    memristor_config = rasr_config.recognition.memristor
    if converter_hardware_settings is not None:
        memristor_config.converter_hardware_settings = _serialize_dac_settings(converter_hardware_settings)
    if pos_enc_converter_hardware_settings is not None:
        memristor_config.pos_enc_converter_hardware_settings = _serialize_dac_settings(
            pos_enc_converter_hardware_settings
        )
    if separated_joint_adc:
        memristor_config.joint_l1_converter_hardware_settings = _serialize_dac_settings(
            converter_hardware_settings
            if joint_l1_converter_hardware_settings is None
            else joint_l1_converter_hardware_settings
        )
        memristor_config.joint_l2_converter_hardware_settings = _serialize_dac_settings(
            converter_hardware_settings
            if joint_l2_converter_hardware_settings is None
            else joint_l2_converter_hardware_settings
        )

    memristor_config.num_cycles = num_cycles
    if ff_cfg.correction_settings is not None:
        correction_settings = ff_cfg.correction_settings
        memristor_config.correction_settings = [
            correction_settings.num_cycles if correction_settings.num_cycles is not None else "None",
            correction_settings.test_input_value if correction_settings.test_input_value is not None else "None",
            correction_settings.relative_deviation if correction_settings.relative_deviation is not None else "None",
            correction_settings.ideal_programming,
        ]

    return rasr_config
