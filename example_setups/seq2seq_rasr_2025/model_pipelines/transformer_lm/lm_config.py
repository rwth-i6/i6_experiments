from i6_core.rasr.config import RasrConfig
from sisyphus import tk


def get_lm_config(onnx_model: tk.Path, vocab_file: tk.Path, lm_scale: float, use_gpu: bool = False) -> RasrConfig:
    rasr_config = RasrConfig()
    rasr_config.type = "onnx-stateless"
    rasr_config.scale = lm_scale
    rasr_config.vocab_file = vocab_file
    rasr_config.max_batch_size = 1
    rasr_config.vocab_unknown_word = "<UNK>"

    rasr_config.onnx_model = RasrConfig()
    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = onnx_model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2
    if use_gpu:
        rasr_config.onnx_model.session.execution_provider_type = "cuda"

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.tokens = "tokens"
    rasr_config.onnx_model.io_map.lengths = "tokens:size1"
    rasr_config.onnx_model.io_map.scores = "scores"

    return rasr_config
