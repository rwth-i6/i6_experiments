__all__ = ["get_lstm_transducer_label_scorer_config"]

from dataclasses import fields

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_scorer, export_state_initializer, export_state_updater
from .pytorch_modules import LstmTransducerConfig, LstmTransducerRecogConfig


def get_lstm_transducer_label_scorer_config(
    model_config: LstmTransducerConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
) -> RasrConfig:
    recog_model_config = LstmTransducerRecogConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        ilm_scale=ilm_scale,
        blank_penalty=blank_penalty,
    )

    scorer_onnx_model = export_scorer(model_config=recog_model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "stateful-onnx"

    rasr_config.scorer_model = RasrConfig()
    rasr_config.scorer_model.session = RasrConfig()
    rasr_config.scorer_model.session.file = scorer_onnx_model
    rasr_config.scorer_model.session.inter_op_num_threads = 2
    rasr_config.scorer_model.session.intra_op_num_threads = 2

    rasr_config.scorer_model.io_map = RasrConfig()
    rasr_config.scorer_model.io_map.scores = "scores"
    rasr_config.scorer_model.io_map.encoder_state = "encoder_state"

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
