__all__ = ["get_aed_label_scorer_config"]

from typing import Optional
from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_scorer, export_state_initializer, export_state_updater
from .pytorch_modules import AEDConfig


def get_aed_label_scorer_config(
    model_config: AEDConfig,
    checkpoint: PtCheckpoint,
    execution_provider_type: Optional[str] = None,
) -> RasrConfig:
    scorer_onnx_model = export_scorer(model_config=model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "full-input-stateful-onnx"

    rasr_config.scorer_model = RasrConfig()
    rasr_config.scorer_model.session = RasrConfig()
    rasr_config.scorer_model.session.file = scorer_onnx_model
    rasr_config.scorer_model.session.inter_op_num_threads = 2
    rasr_config.scorer_model.session.intra_op_num_threads = 2

    rasr_config.scorer_model.io_map = RasrConfig()
    rasr_config.scorer_model.io_map.scores = "scores"

    rasr_config.state_initializer_model = RasrConfig()
    rasr_config.state_initializer_model.session = RasrConfig()
    rasr_config.state_initializer_model.session.file = state_initializer_onnx_model
    rasr_config.state_initializer_model.session.inter_op_num_threads = 2
    rasr_config.state_initializer_model.session.intra_op_num_threads = 2

    rasr_config.state_initializer_model.io_map = RasrConfig()
    rasr_config.state_initializer_model.io_map.encoder_states = "encoder_states"
    rasr_config.state_initializer_model.io_map.encoder_states_size = "encoder_states:size1"

    rasr_config.state_updater_model = RasrConfig()
    rasr_config.state_updater_model.session = RasrConfig()
    rasr_config.state_updater_model.session.file = state_updater_onnx_model
    rasr_config.state_updater_model.session.inter_op_num_threads = 2
    rasr_config.state_updater_model.session.intra_op_num_threads = 2

    rasr_config.state_updater_model.io_map = RasrConfig()
    rasr_config.state_updater_model.io_map.encoder_states = "encoder_states"
    rasr_config.state_updater_model.io_map.encoder_states_size = "accum_att_weights_in:size1"
    rasr_config.state_updater_model.io_map.token = "token"

    if execution_provider_type:
        rasr_config.scorer_model.session.execution_provider_type = execution_provider_type
        rasr_config.state_initializer_model.session.execution_provider_type = execution_provider_type
        rasr_config.state_updater_model.session.execution_provider_type = execution_provider_type

    return rasr_config
