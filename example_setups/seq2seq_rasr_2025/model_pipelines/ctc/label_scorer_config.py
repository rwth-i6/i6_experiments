__all__ = ["get_ctc_label_scorer_config"]

from dataclasses import fields
from typing import Optional

from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint
from sisyphus import tk

from .export import export_model
from .pytorch_modules import ConformerCTCConfig, ConformerCTCRecogConfig


def get_ctc_label_scorer_config(
    model_config: ConformerCTCConfig,
    checkpoint: PtCheckpoint,
    prior_file: tk.Path,
    prior_scale: float = 0.0,
    blank_penalty: float = 0.0,
    chunk_history: Optional[int] = None,
    chunk_center: Optional[int] = None,
    chunk_future: Optional[int] = None,
) -> RasrConfig:

    recog_model_config = ConformerCTCRecogConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        prior_file=prior_file,
        prior_scale=prior_scale,
        blank_penalty=blank_penalty,
    )

    onnx_model = export_model(model_config=recog_model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "encoder-only"

    rasr_config.encoder = RasrConfig()
    if chunk_history is not None or chunk_center is not None or chunk_future is not None:
        rasr_config.encoder.type = "chunked-onnx"
        if chunk_history is not None:
            rasr_config.encoder.chunk_history = chunk_history
        if chunk_center is not None:
            rasr_config.encoder.chunk_center = chunk_center
        if chunk_future is not None:
            rasr_config.encoder.chunk_future = chunk_future
    else:
        rasr_config.encoder.type = "onnx"

    rasr_config.encoder.onnx_model = RasrConfig()

    rasr_config.encoder.onnx_model.session = RasrConfig()
    rasr_config.encoder.onnx_model.session.file = onnx_model
    rasr_config.encoder.onnx_model.session.inter_op_num_threads = 2
    rasr_config.encoder.onnx_model.session.intra_op_num_threads = 2

    rasr_config.encoder.onnx_model.io_map = RasrConfig()
    rasr_config.encoder.onnx_model.io_map.features = "audio_samples"
    rasr_config.encoder.onnx_model.io_map.features_size = "audio_samples:size1"
    rasr_config.encoder.onnx_model.io_map.outputs = "scores"

    return rasr_config
