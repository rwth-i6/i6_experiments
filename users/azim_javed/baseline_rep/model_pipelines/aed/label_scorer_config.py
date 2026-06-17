__all__ = ["get_aed_label_scorer_config"]

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_ctc_scorer, export_scorer, export_state_initializer, export_state_updater
from .pytorch_modules import AEDConfig


def get_aed_label_scorer_config(
    model_config: AEDConfig,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:
    scorer_onnx_model = export_scorer(model_config=model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_state_updater(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "stateful-onnx"
    rasr_config.max_cached_score_vectors = 100000

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

    if scale != 1.0:
        rasr_config.scale = scale

    if use_gpu:
        rasr_config.scorer_model.session.execution_provider_type = "cuda"
        rasr_config.state_initializer_model.session.execution_provider_type = "cuda"
        rasr_config.state_updater_model.session.execution_provider_type = "cuda"

    return rasr_config


def get_ctc_label_scorer_config(
    model_config: AEDConfig,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:

    onnx_model = export_ctc_scorer(model_config=model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "no-context-onnx"

    rasr_config.onnx_model = RasrConfig()

    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = onnx_model
    rasr_config.onnx_model.session.inter_op_num_threads = 3
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.input_feature = "encoder_state"
    rasr_config.onnx_model.io_map.scores = "scores"

    if scale != 1.0:
        rasr_config.scale = scale

    if use_gpu:
        rasr_config.onnx_model.session.execution_provider_type = "cuda"

    return rasr_config


def get_ctc_prefix_label_scorer_config(
    model_config: AEDConfig,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:

    rasr_config = RasrConfig()
    rasr_config.type = "ctc-prefix"
    rasr_config.blank_label_index = model_config.label_target_size
    rasr_config.vocab_size = model_config.label_target_size + 1

    rasr_config.ctc_scorer = get_ctc_label_scorer_config(
        model_config=model_config,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
    )

    if scale != 1.0:
        rasr_config.scale = scale

    return rasr_config
