from typing import Optional

from i6_core.rasr.config import RasrConfig
from i6_core.returnn import PtCheckpoint

from .export import (
    export_model_kv_cached,
    export_model_stateless,
)
from .pytorch_modules import TransformerLmConfig

# TODO: Currently using stateless model which works but is very slow. LabelScorer with proper KV caching still needs to be added.


def get_bpe_transformer_lm_stateless_label_scorer_config(
    model_config: TransformerLmConfig,
    checkpoint: PtCheckpoint,
    execution_provider_type: Optional[str] = None,
) -> RasrConfig:
    model = export_model_stateless(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "full-context-onnx"
    rasr_config.max_batch_size = 64
    rasr_config.start_label_index = 0

    rasr_config.onnx_model = RasrConfig()
    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.history = "tokens"
    rasr_config.onnx_model.io_map.history_length = "tokens:size1"
    rasr_config.onnx_model.io_map.scores = "scores"

    if execution_provider_type:
        rasr_config.onnx_model.session.execution_provider_type = execution_provider_type

    return rasr_config


def get_bpe_transformer_lm_label_scorer_config(
    model_config: TransformerLmConfig,
    checkpoint: PtCheckpoint,
    execution_provider_type: Optional[str] = None,
) -> RasrConfig:
    model = export_model_kv_cached(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "state-managed-onnx"
    rasr_config.max_batch_size = 64
    rasr_config.start_labels = 0

    rasr_config.onnx_model = RasrConfig()
    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.token = "tokens"
    rasr_config.onnx_model.io_map.token_length = "tokens:size1"
    rasr_config.onnx_model.io_map.prefix_length = "state_l000_k_in:size1"
    rasr_config.onnx_model.io_map.scores = "scores"

    rasr_config.state_manager = RasrConfig()
    rasr_config.state_manager.type = "transformer"

    if execution_provider_type:
        rasr_config.onnx_model.session.execution_provider_type = execution_provider_type

    return rasr_config
