__all__ = ["get_lstm_lm_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_scorer, export_state_initializer, export_state_updater
from .pytorch_modules import LstmLmConfig


def get_lstm_lm_label_scorer_config(
    model_config: LstmLmConfig,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
) -> RasrConfig:
    scorer_onnx_model = export_scorer(model_config=model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "stateful-onnx"
    if scale != 1:
        rasr_config.scale = scale

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

    rasr_config.state_updater_model = RasrConfig()
    rasr_config.state_updater_model.session = RasrConfig()
    rasr_config.state_updater_model.session.file = state_updater_onnx_model
    rasr_config.state_updater_model.session.inter_op_num_threads = 2
    rasr_config.state_updater_model.session.intra_op_num_threads = 2

    rasr_config.state_updater_model.io_map = RasrConfig()
    rasr_config.state_updater_model.io_map.token = "token"

    return rasr_config
