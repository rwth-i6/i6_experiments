__all__ = ["get_ffnn_transducer_label_scorer_config"]

from dataclasses import fields

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import export_scorer
from .pytorch_modules import FFNNTransducerConfig, FFNNTransducerRecogConfig


def get_ffnn_transducer_label_scorer_config(
    model_config: FFNNTransducerConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
) -> RasrConfig:
    recog_model_config = FFNNTransducerRecogConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        ilm_scale=ilm_scale,
        blank_penalty=blank_penalty,
    )

    scorer_onnx_model = export_scorer(model_config=recog_model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "limited-ctx-onnx"
    rasr_config.history_length = model_config.context_history_size
    rasr_config.start_label_index = model_config.target_size - 1

    rasr_config.onnx_model = RasrConfig()
    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = scorer_onnx_model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.encoder_state = "encoder_state"
    rasr_config.onnx_model.io_map.history = "history"
    rasr_config.onnx_model.io_map.scores = "scores"

    return rasr_config
