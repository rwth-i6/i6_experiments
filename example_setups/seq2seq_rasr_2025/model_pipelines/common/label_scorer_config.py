from typing import Optional
from i6_core.rasr.config import RasrConfig
from sisyphus import tk


def get_no_op_label_scorer_config(scale: float = 1.0) -> RasrConfig:
    config = RasrConfig()
    config.type = "no-op"
    if scale != 1.0:
        config.scale = scale

    return config


def get_encoder_decoder_label_scorer_config(
    encoder_onnx_model: tk.Path,
    decoder_label_scorer_config: Optional[RasrConfig] = None,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:
    rasr_config = RasrConfig()
    if decoder_label_scorer_config is None:
        rasr_config.type = "encoder-only"
    else:
        rasr_config.type = "encoder-decoder"
        rasr_config.decoder = decoder_label_scorer_config

    rasr_config.encoder = RasrConfig()
    rasr_config.encoder.type = "onnx"

    rasr_config.encoder.onnx_model = RasrConfig()

    rasr_config.encoder.onnx_model.session = RasrConfig()
    rasr_config.encoder.onnx_model.session.file = encoder_onnx_model
    rasr_config.encoder.onnx_model.session.inter_op_num_threads = 2
    rasr_config.encoder.onnx_model.session.intra_op_num_threads = 2

    rasr_config.encoder.onnx_model.io_map = RasrConfig()
    rasr_config.encoder.onnx_model.io_map.features = "features"
    rasr_config.encoder.onnx_model.io_map.features_size = "features:size1"
    rasr_config.encoder.onnx_model.io_map.outputs = "enc_out"

    if use_gpu:
        rasr_config.encoder.onnx_model.session.execution_provider_type = "cuda"

    if scale != 1:
        rasr_config.scale = scale

    return rasr_config
