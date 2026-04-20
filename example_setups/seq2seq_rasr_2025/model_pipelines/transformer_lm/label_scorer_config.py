from typing import Optional

from i6_core.rasr.config import RasrConfig
from i6_core.returnn import PtCheckpoint

from .export import (
    export_model_kv_cached,
    export_model_stateless,
    export_scorer,
    export_state_initializer,
    export_state_updater,
)
from .pytorch_modules import TransformerLmConfig

# TODO: Currently using stateless model which works but is very slow. LabelScorer with proper KV caching still needs to be added.


# def get_bpe_transformer_lm_label_scorer_config(
#     model_config: TransformerLmConfig,
#     checkpoint: PtCheckpoint,
#     execution_provider_type: Optional[str] = None,
# ) -> RasrConfig:
#     model = export_model_stateless(model_config=model_config, checkpoint=checkpoint)
#
#     rasr_config = RasrConfig()
#     rasr_config.type = "full-context-onnx"
#     rasr_config.max_batch_size = 64
#     rasr_config.start_label_index = 0
#
#     rasr_config.onnx_model = RasrConfig()
#     rasr_config.onnx_model.session = RasrConfig()
#     rasr_config.onnx_model.session.file = model
#     rasr_config.onnx_model.session.inter_op_num_threads = 2
#     rasr_config.onnx_model.session.intra_op_num_threads = 2
#
#     rasr_config.onnx_model.io_map = RasrConfig()
#     rasr_config.onnx_model.io_map.history = "tokens"
#     rasr_config.onnx_model.io_map.history_length = "tokens:size1"
#     rasr_config.onnx_model.io_map.scores = "scores"
#
#     if execution_provider_type:
#         rasr_config.onnx_model.session.execution_provider_type = execution_provider_type
#
#     return rasr_config


def get_bpe_transformer_lm_label_scorer_config(
    model_config: TransformerLmConfig,
    checkpoint: PtCheckpoint,
    execution_provider_type: Optional[str] = None,
) -> RasrConfig:
    model = export_model_kv_cached(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "state-managed-onnx"
    rasr_config.max_batch_size = 64
    rasr_config.start_label_index = 0

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


# def get_bpe_transformer_lm_label_scorer_config(
#     model_config: TransformerLmConfig,
#     checkpoint: PtCheckpoint,
#     scale: float = 1.0,
#     execution_provider_type: Optional[str] = None,
# ) -> RasrConfig:
#     scorer_onnx_model = export_scorer(model_config=model_config, checkpoint=checkpoint)
#     state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
#     state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)
#
#     rasr_config = RasrConfig()
#     rasr_config.type = "stateful-onnx"
#     rasr_config.max_batch_size = 64
#     rasr_config.max_cached_score_vectors = 1000
#
#     rasr_config.scorer_model = RasrConfig()
#     rasr_config.scorer_model.session = RasrConfig()
#     rasr_config.scorer_model.session.file = scorer_onnx_model
#     rasr_config.scorer_model.session.inter_op_num_threads = 2
#     rasr_config.scorer_model.session.intra_op_num_threads = 2
#
#     rasr_config.scorer_model.io_map = RasrConfig()
#     rasr_config.scorer_model.io_map.scores = "scores"
#
#     rasr_config.state_initializer_model = RasrConfig()
#     rasr_config.state_initializer_model.session = RasrConfig()
#     rasr_config.state_initializer_model.session.file = state_initializer_onnx_model
#     rasr_config.state_initializer_model.session.inter_op_num_threads = 2
#     rasr_config.state_initializer_model.session.intra_op_num_threads = 2
#
#     rasr_config.state_updater_model = RasrConfig()
#     rasr_config.state_updater_model.session = RasrConfig()
#     rasr_config.state_updater_model.session.file = state_updater_onnx_model
#     rasr_config.state_updater_model.session.inter_op_num_threads = 2
#     rasr_config.state_updater_model.session.intra_op_num_threads = 2
#
#     rasr_config.state_updater_model.io_map = RasrConfig()
#     rasr_config.state_updater_model.io_map.token = "token"
#
#     if scale != 1.0:
#         rasr_config.scale = scale
#
#     if execution_provider_type:
#         rasr_config.scorer_model.session.execution_provider_type = execution_provider_type
#         rasr_config.state_initializer_model.session.execution_provider_type = execution_provider_type
#         rasr_config.state_updater_model.session.execution_provider_type = execution_provider_type
#
#     return rasr_config
