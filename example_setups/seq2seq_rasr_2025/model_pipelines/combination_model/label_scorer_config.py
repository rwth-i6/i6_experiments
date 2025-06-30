__all__ = [
    "get_ctc_label_scorer_config",
    "get_ctc_prefix_label_scorer_config",
    "get_transducer_label_scorer_config",
    "get_attention_label_scorer_config",
    "get_combine_label_scorer_config",
]

from dataclasses import fields
from typing import Optional
from ...model_pipelines.common.recog_rasr_config import (
    get_combine_label_scorer_config as _get_combine_label_scorer_config,
)
from ..bpe_lstm_lm.label_scorer_config import get_lstm_lm_label_scorer_config
from ..bpe_lstm_lm.pytorch_modules import LstmLmConfig
from sisyphus import tk
from i6_core.rasr.config import RasrConfig
from i6_core.returnn.training import PtCheckpoint

from .export import (
    export_encoder,
    export_ctc_scorer,
    export_transducer_scorer,
    export_attention_scorer,
    export_attention_state_initializer,
    export_attention_state_updater,
)
from .pytorch_modules import (
    CombinationModelCTCRecogConfig,
    CombinationModelConfig,
    CombinationModelTransducerRecogConfig,
)


def get_ctc_label_scorer_config(
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
    prior_file: tk.Path,
    prior_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
) -> RasrConfig:

    recog_model_config = CombinationModelCTCRecogConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        prior_file=prior_file,
        prior_scale=prior_scale,
        blank_penalty=blank_penalty,
    )

    onnx_model = export_ctc_scorer(model_config=recog_model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "no-ctx-onnx"
    if scale != 1:
        rasr_config.scale = scale

    rasr_config.onnx_model = RasrConfig()

    rasr_config.onnx_model.session = RasrConfig()
    rasr_config.onnx_model.session.file = onnx_model
    rasr_config.onnx_model.session.inter_op_num_threads = 2
    rasr_config.onnx_model.session.intra_op_num_threads = 2

    rasr_config.onnx_model.io_map = RasrConfig()
    rasr_config.onnx_model.io_map.encoder_state = "encoder_state"
    rasr_config.onnx_model.io_map.scores = "scores"

    return rasr_config


def get_ctc_prefix_label_scorer_config(
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
    prior_file: tk.Path,
    prior_scale: float = 0.0,
    scale: float = 1.0,
) -> RasrConfig:

    rasr_config = RasrConfig()
    rasr_config.type = "ctc-prefix"
    rasr_config.blank_label_index = model_config.target_size - 1
    if scale != 1:
        rasr_config.scale = scale

    rasr_config.ctc_scorer = get_ctc_label_scorer_config(
        model_config=model_config,
        checkpoint=checkpoint,
        prior_file=prior_file,
        prior_scale=prior_scale,
    )

    return rasr_config


def get_transducer_label_scorer_config(
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
    ilm_scale: float = 0.0,
    blank_penalty: float = 0.0,
    scale: float = 1.0,
) -> RasrConfig:
    recog_model_config = CombinationModelTransducerRecogConfig(
        **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
        ilm_scale=ilm_scale,
        blank_penalty=blank_penalty,
    )

    scorer_onnx_model = export_transducer_scorer(model_config=recog_model_config, checkpoint=checkpoint)

    rasr_config = RasrConfig()
    rasr_config.type = "limited-ctx-onnx"
    rasr_config.history_length = model_config.transducer_context_history_size
    rasr_config.start_label_index = 0
    if scale != 1:
        rasr_config.scale = scale

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


def get_attention_label_scorer_config(
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
    scale: float = 1.0,
) -> RasrConfig:
    scorer_onnx_model = export_attention_scorer(model_config=model_config, checkpoint=checkpoint)
    state_initializer_onnx_model = export_attention_state_initializer(model_config=model_config, checkpoint=checkpoint)
    state_updater_onnx_model = export_attention_state_updater(model_config=model_config, checkpoint=checkpoint)

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

    return rasr_config


def get_combine_label_scorer_config(
    model_config: CombinationModelConfig,
    checkpoint: PtCheckpoint,
    include_encoder: bool = False,
    ctc_score_scale: float = 0.0,
    ctc_prefix_score_scale: float = 0.0,
    transducer_score_scale: float = 0.0,
    attention_score_scale: float = 0.0,
    ctc_blank_penalty: float = 0.0,
    ctc_prior_file: Optional[tk.Path] = None,
    ctc_prior_scale: float = 0.0,
    transducer_ilm_scale: float = 0.0,
    transducer_blank_penalty: float = 0.0,
    lm_checkpoint: Optional[PtCheckpoint] = None,
    lm_config: Optional[LstmLmConfig] = None,
    lm_scale: float = 0.0,
) -> RasrConfig:
    sub_scorers = []
    if ctc_score_scale != 0:
        assert ctc_prior_file is not None
        ctc_scorer = get_ctc_label_scorer_config(
            model_config=model_config,
            checkpoint=checkpoint,
            prior_file=ctc_prior_file,
            prior_scale=ctc_prior_scale,
            blank_penalty=ctc_blank_penalty,
        )
        sub_scorers.append((ctc_scorer, ctc_score_scale))

    if ctc_prefix_score_scale != 0:
        assert ctc_prior_file is not None
        ctc_prefix_scorer = get_ctc_prefix_label_scorer_config(
            model_config=model_config,
            checkpoint=checkpoint,
            prior_file=ctc_prior_file,
            prior_scale=ctc_prior_scale,
        )
        sub_scorers.append((ctc_prefix_scorer, ctc_prefix_score_scale))

    if transducer_score_scale != 0:
        transducer_scorer = get_transducer_label_scorer_config(
            model_config=model_config,
            checkpoint=checkpoint,
            ilm_scale=transducer_ilm_scale,
            blank_penalty=transducer_blank_penalty,
        )
        sub_scorers.append((transducer_scorer, transducer_score_scale))

    if attention_score_scale != 0:
        attention_scorer = get_attention_label_scorer_config(
            model_config=model_config,
            checkpoint=checkpoint,
        )
        sub_scorers.append((attention_scorer, attention_score_scale))

    if lm_scale != 0:
        assert lm_config is not None
        assert lm_checkpoint is not None
        lm_scorer = get_lstm_lm_label_scorer_config(
            model_config=lm_config,
            checkpoint=lm_checkpoint,
        )
        sub_scorers.append((lm_scorer, lm_scale))

    if len(sub_scorers) == 1 and sub_scorers[0][1] == 1:
        decoder_config = sub_scorers[0][0]
    else:
        decoder_config = _get_combine_label_scorer_config(sub_scorers)

    if include_encoder:
        encoder_onnx_model = export_encoder(model_config=model_config, checkpoint=checkpoint)

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

        rasr_config.decoder = decoder_config
    else:
        rasr_config = decoder_config

    return rasr_config
