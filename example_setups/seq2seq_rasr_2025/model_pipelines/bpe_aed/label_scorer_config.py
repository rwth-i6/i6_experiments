__all__ = ["get_aed_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_encoder, export_scorer, export_state_initializer, export_state_updater
from .pytorch_modules import AEDConfig


def get_aed_label_scorer_config(
    model_config: AEDConfig,
    checkpoint: PtCheckpoint,
) -> RasrConfig:
    encoder_onnx_model = export_encoder(model_config=model_config, checkpoint=checkpoint)
    scorer_onnx_model = export_scorer(model_config=model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "encoder-decoder"

    rasr_config.encoder = RasrConfig()
    rasr_config.encoder.type = "onnx"

    rasr_config.encoder.onnx_model = RasrConfig()

    rasr_config.encoder.onnx_model.session = RasrConfig()
    rasr_config.encoder.onnx_model.session.file = encoder_onnx_model
    rasr_config.encoder.onnx_model.session.inter_op_num_threads = 2
    rasr_config.encoder.onnx_model.session.intra_op_num_threads = 2

    rasr_config.encoder.onnx_model.io_map = RasrConfig()
    rasr_config.encoder.onnx_model.io_map.features = "audio_samples"
    rasr_config.encoder.onnx_model.io_map.features_size = "audio_samples:size1"
    rasr_config.encoder.onnx_model.io_map.outputs = "encoder_states"

    rasr_config.decoder = RasrConfig()
    rasr_config.decoder.type = "stateful-onnx"
    rasr_config.decoder.start_label_index = 0

    rasr_config.decoder.scorer_model = RasrConfig()
    rasr_config.decoder.scorer_model.session = RasrConfig()
    rasr_config.decoder.scorer_model.session.file = scorer_onnx_model
    rasr_config.decoder.scorer_model.session.inter_op_num_threads = 2
    rasr_config.decoder.scorer_model.session.intra_op_num_threads = 2

    rasr_config.decoder.scorer_model.io_map = RasrConfig()
    rasr_config.decoder.scorer_model.io_map.scores = "scores"

    rasr_config.decoder.state_initializer_model = RasrConfig()
    rasr_config.decoder.state_initializer_model.session = RasrConfig()
    rasr_config.decoder.state_initializer_model.session.file = state_initializer_onnx_model
    rasr_config.decoder.state_initializer_model.session.inter_op_num_threads = 2
    rasr_config.decoder.state_initializer_model.session.intra_op_num_threads = 2

    rasr_config.decoder.state_initializer_model.io_map = RasrConfig()
    rasr_config.decoder.state_initializer_model.io_map.encoder_states = "encoder_states"
    rasr_config.decoder.state_initializer_model.io_map.encoder_states_size = "encoder_states:size1"

    rasr_config.decoder.state_updater_model = RasrConfig()
    rasr_config.decoder.state_updater_model.session = RasrConfig()
    rasr_config.decoder.state_updater_model.session.file = state_updater_onnx_model
    rasr_config.decoder.state_updater_model.session.inter_op_num_threads = 2
    rasr_config.decoder.state_updater_model.session.intra_op_num_threads = 2

    rasr_config.decoder.state_updater_model.io_map = RasrConfig()
    rasr_config.decoder.state_updater_model.io_map.encoder_states = "encoder_states"
    rasr_config.decoder.state_updater_model.io_map.encoder_states_size = "accum_att_weights_in:size1"
    rasr_config.decoder.state_updater_model.io_map.token = "token"

    return rasr_config
