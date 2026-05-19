from i6_core.rasr.config import RasrConfig
from i6_core.returnn import PtCheckpoint

from .export import export_ctc_scorer, export_initializer_model, export_step_model


def get_speech_lm_label_scorer_config(
    model_kwargs: dict,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:
    initializer_model = export_initializer_model(model_kwargs=model_kwargs, checkpoint=checkpoint)
    step_model = export_step_model(model_kwargs=model_kwargs, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "prefix-speech-lm-onnx"
    rasr_config.initial_prompt_labels = [6448, 25, 220]  # Currently hardcoded tokenization of "USER:"
    rasr_config.suffix_prompt_labels = [
        3167,
        3114,
        8806,
        311,
        1467,
        13,
        35560,
        3846,
        2821,
        25,
        220,
        151643,
    ]  # Currently hardcoded tokenization of "Transcribe speech to text. ASSISTANT:" + <bos>

    rasr_config.initializer_model = RasrConfig()
    rasr_config.initializer_model.session = RasrConfig()
    rasr_config.initializer_model.session.file = initializer_model
    rasr_config.initializer_model.session.inter_op_num_threads = 2
    rasr_config.initializer_model.session.intra_op_num_threads = 2

    rasr_config.initializer_model.io_map = RasrConfig()
    rasr_config.initializer_model.io_map.initial_prompt = "initial_prompt"
    rasr_config.initializer_model.io_map.encoder_states = "encoder_states"
    rasr_config.initializer_model.io_map.suffix_prompt = "suffix_prompt"
    rasr_config.initializer_model.io_map.scores = "scores"

    rasr_config.step_model = RasrConfig()
    rasr_config.step_model.session = RasrConfig()
    rasr_config.step_model.session.file = step_model
    rasr_config.step_model.session.inter_op_num_threads = 2
    rasr_config.step_model.session.intra_op_num_threads = 2

    rasr_config.step_model.io_map = RasrConfig()
    rasr_config.step_model.io_map.token = "token"
    rasr_config.step_model.io_map.prefix_length = "state_l000_k_in:size2"
    rasr_config.step_model.io_map.scores = "scores"

    rasr_config.state_manager = RasrConfig()
    rasr_config.state_manager.type = "transformer"

    if scale != 1.0:
        rasr_config.scale = scale

    if use_gpu:
        rasr_config.initializer_model.session.execution_provider_type = "cuda"
        rasr_config.step_model.session.execution_provider_type = "cuda"

    return rasr_config


def get_ctc_label_scorer_config(
    model_kwargs: dict,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:
    onnx_model = export_ctc_scorer(model_kwargs=model_kwargs, checkpoint=checkpoint)

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
    model_kwargs: dict,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
    use_gpu: bool = False,
) -> RasrConfig:

    rasr_config = RasrConfig()
    rasr_config.type = "ctc-prefix"
    rasr_config.blank_label_index = 151936
    rasr_config.vocab_size = 151937

    rasr_config.ctc_scorer = get_ctc_label_scorer_config(
        model_kwargs=model_kwargs,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
    )

    if scale != 1.0:
        rasr_config.scale = scale

    return rasr_config
