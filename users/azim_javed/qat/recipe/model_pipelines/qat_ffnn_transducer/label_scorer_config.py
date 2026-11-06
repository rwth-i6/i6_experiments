__all__ = ["get_ffnn_transducer_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import

from .pytorch_modules import QATFFNNTransducerConfig, QATFFNNTransducerRecogConfig, QATFFNNTransducerScorer
from ...experiments.librispeech.training.qat_ffnn_transducer_bpe import get_model_config

def _get_block_cfg(model_config: QATFFNNTransducerConfig):
    block_cfg = model_config.conformer_cfg.block_cfg
    if isinstance(block_cfg, list):
        assert block_cfg, "empty conformer block_cfg list"
        block_cfg = block_cfg[0]
    return block_cfg


def get_ffnn_transducer_label_scorer_config(
    model_config: QATFFNNTransducerConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
    use_gpu: bool = False,
    max_batch_size: int = 2048,
    use_cache: bool = False,
    scorer_model_class=QATFFNNTransducerScorer,
    debug_joint: bool = False,
) -> RasrConfig:

    label_scorer_type = "fixed-context-py-cache" if use_cache else "fixed-context-py"

    rasr_config = RasrConfig()
    rasr_config.type = label_scorer_type
    rasr_config.history_length = model_config.context_history_size
    rasr_config.start_label_index = model_config.target_size - 1
    rasr_config.max_batch_size = max_batch_size

    if debug_joint:
        rasr_config.debug_joint = True

    rasr_config.recognition = RasrConfig()
    rasr_config.recognition.ilm_scale = ilm_scale
    rasr_config.recognition.blank_penalty = blank_penalty
    rasr_config.recognition.scale = scale
    rasr_config.recognition.model_path = checkpoint
    rasr_config.recognition.device = "cuda" if use_gpu else "cpu"
    imports = [
        Import(f"{scorer_model_class.__module__}.{scorer_model_class.__name__}", import_as="ScorerModel"),
        Import(f"{QATFFNNTransducerRecogConfig.__module__}.{QATFFNNTransducerRecogConfig.__name__}", import_as="RecogConfig"),
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
    if (model_config.version_control or "").endswith("-no-joiner-output-bias"):
        rasr_config.recognition.qat.version_control = model_config.version_control

    return rasr_config
