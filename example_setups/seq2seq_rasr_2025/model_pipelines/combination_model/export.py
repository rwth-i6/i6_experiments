__all__ = [
    "export_encoder",
    "export_ctc_scorer",
    "export_transducer_scorer",
    "export_attention_scorer",
    "export_attention_state_initializer",
    "export_attention_state_updater",
]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.export import export_model as _export_model
from ..common.imports import get_model_serializers
from .pytorch_modules import (
    CombinationModelAttentionScorer,
    CombinationModelAttentionStateInitializer,
    CombinationModelAttentionStateUpdater,
    CombinationModelConfig,
    CombinationModelCTCRecogConfig,
    CombinationModelCTCScorer,
    CombinationModelEncoder,
    CombinationModelTransducerRecogConfig,
    CombinationModelTransducerScorer,
)

# -----------------------
# --- Forward steps -----
# -----------------------


def _encoder_forward_step(*, model: CombinationModelEncoder, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    audio_samples = extern_data["audio_samples"].raw_tensor  # [B, T, 1]
    assert audio_samples is not None

    assert extern_data["audio_samples"].dims[1].dyn_size_ext is not None
    audio_samples_size = extern_data["audio_samples"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert audio_samples_size is not None

    encoder_states = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
    )

    run_ctx.mark_as_output(name="encoder_states", tensor=encoder_states)


def _ctc_scorer_forward_step(*, model: CombinationModelCTCScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor
    assert encoder_state is not None

    scores = model.forward(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _transducer_scorer_forward_step(*, model: CombinationModelTransducerScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor
    assert encoder_state is not None
    history = extern_data["history"].raw_tensor
    assert history is not None

    scores = model.forward(encoder_state=encoder_state, history=history)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _attention_scorer_forward_step(*, model: CombinationModelAttentionScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    token_embedding = extern_data["token_embedding"].raw_tensor
    assert token_embedding is not None
    lstm_state_h = extern_data["lstm_state_h"].raw_tensor
    assert lstm_state_h is not None
    att_context = extern_data["att_context"].raw_tensor
    assert att_context is not None

    scores = model.forward(
        token_embedding=token_embedding,
        lstm_state_h=lstm_state_h,
        att_context=att_context,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _attention_state_initializer_forward_step(
    *, model: CombinationModelAttentionStateInitializer, extern_data: TensorDict, **_
):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_states = extern_data["encoder_states"].raw_tensor
    assert encoder_states is not None
    assert extern_data["encoder_states"].dims[1].dyn_size_ext is not None
    encoder_states_size = extern_data["encoder_states"].dims[1].dyn_size_ext.raw_tensor
    assert encoder_states_size is not None

    token_embedding, lstm_state_h, lstm_state_c, att_context, accum_att_weights = model.forward(
        encoder_states=encoder_states, encoder_states_size=encoder_states_size
    )

    assert run_ctx.expected_outputs
    run_ctx.expected_outputs["accum_att_weights"].dims[1].dyn_size_ext = (
        extern_data["encoder_states"].dims[1].dyn_size_ext
    )

    run_ctx.mark_as_output(name="token_embedding", tensor=token_embedding)
    run_ctx.mark_as_output(name="lstm_state_c", tensor=lstm_state_c)
    run_ctx.mark_as_output(name="lstm_state_h", tensor=lstm_state_h)
    run_ctx.mark_as_output(name="att_context", tensor=att_context)
    run_ctx.mark_as_output(name="accum_att_weights", tensor=accum_att_weights)


def _attention_state_updater_forward_step(
    *, model: CombinationModelAttentionStateUpdater, extern_data: TensorDict, **_
):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_states = extern_data["encoder_states"].raw_tensor
    assert encoder_states is not None
    assert extern_data["encoder_states"].dims[1].dyn_size_ext is not None
    encoder_states_size = extern_data["encoder_states"].dims[1].dyn_size_ext.raw_tensor
    assert encoder_states_size is not None
    token = extern_data["token"].raw_tensor
    assert token is not None
    lstm_state_c_in = extern_data["lstm_state_c_in"].raw_tensor
    assert lstm_state_c_in is not None
    lstm_state_h_in = extern_data["lstm_state_h_in"].raw_tensor
    assert lstm_state_h_in is not None
    att_context_in = extern_data["att_context_in"].raw_tensor
    assert att_context_in is not None
    accum_att_weights_in = extern_data["accum_att_weights_in"].raw_tensor
    assert accum_att_weights_in is not None

    token_embedding_out, lstm_state_h_out, lstm_state_c_out, att_context_out, accum_att_weights_out = model.forward(
        encoder_states=encoder_states,
        encoder_states_size=encoder_states_size,
        token=token,
        lstm_state_c=lstm_state_c_in,
        lstm_state_h=lstm_state_h_in,
        att_context=att_context_in,
        accum_att_weights=accum_att_weights_in,
    )

    assert run_ctx.expected_outputs
    run_ctx.expected_outputs["accum_att_weights_out"].dims[1].dyn_size_ext = (
        extern_data["accum_att_weights_in"].dims[1].dyn_size_ext
    )

    run_ctx.mark_as_output(name="token_embedding_out", tensor=token_embedding_out)
    run_ctx.mark_as_output(name="lstm_state_c_out", tensor=lstm_state_c_out)
    run_ctx.mark_as_output(name="lstm_state_h_out", tensor=lstm_state_h_out)
    run_ctx.mark_as_output(name="att_context_out", tensor=att_context_out)
    run_ctx.mark_as_output(name="accum_att_weights_out", tensor=accum_att_weights_out)


# -----------------------
# --- Export routines ---
# -----------------------


def export_encoder(model_config: CombinationModelConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=CombinationModelEncoder, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_encoder_forward_step.__module__}.{_encoder_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "audio_samples": {
                    "dim": 1,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "encoder_states": {
                    "dim": model_config.enc_dim + model_config.attention_decoder_config.attention_cfg.attention_dim + 1,
                    "dtype": "float32",
                },
            },
        },
        input_names=["audio_samples", "audio_samples:size1"],
        output_names=["encoder_states", "encoder_states:size1"],
    )


def export_ctc_scorer(model_config: CombinationModelCTCRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=CombinationModelCTCScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_ctc_scorer_forward_step.__module__}.{_ctc_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "encoder_state": {
                    "shape": (
                        1,
                        model_config.enc_dim + model_config.attention_decoder_config.attention_cfg.attention_dim + 1,
                    ),
                    "dim": model_config.enc_dim,
                    "dtype": "float32",
                    "time_dim_axis": None,
                    "batch_dim_axis": None,
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.target_size,
                    "shape": (1, model_config.target_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_state"],
        output_names=["scores"],
    )


def export_transducer_scorer(model_config: CombinationModelTransducerRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=CombinationModelTransducerScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_transducer_scorer_forward_step.__module__}.{_transducer_scorer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "encoder_state": {
                    "shape": (
                        1,
                        model_config.enc_dim + model_config.attention_decoder_config.attention_cfg.attention_dim + 1,
                    ),
                    "dim": model_config.enc_dim,
                    "dtype": "float32",
                    "batch_dim_axis": None,
                },
                "history": {
                    "dim": model_config.target_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "shape": (model_config.transducer_context_history_size,),
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.target_size,
                    "dtype": "float32",
                    "time_dim_axis": None,
                },
            },
            "backend": "torch",
        },
        input_names=["encoder_state", "history"],
        output_names=["scores"],
    )


def export_attention_scorer(model_config: CombinationModelConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=CombinationModelAttentionScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_attention_scorer_forward_step.__module__}.{_attention_scorer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "token_embedding": {
                    "dim": model_config.attention_decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "att_context": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.attention_decoder_config.vocab_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["token_embedding", "lstm_state_h", "att_context"],
        output_names=["scores"],
        metadata={
            "token_embedding": "token_embedding",
            "lstm_state_h": "lstm_state_h",
            "att_context": "att_context",
        },
    )


def export_attention_state_initializer(model_config: CombinationModelConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(
        model_class=CombinationModelAttentionStateInitializer, model_config=model_config
    )

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_attention_state_initializer_forward_step.__module__}.{_attention_state_initializer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "encoder_time_dim": CodeWrapper('Dim(dimension=None, description="EncTime")'),
            "encoder_feature_dim": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.attention_decoder_config.attention_cfg.attention_dim + 1}, description="EncTime")'
            ),
            "accum_att_weights_dim": CodeWrapper('Dim(dimension=1, description="AttWeights")'),
            "extern_data": {
                "encoder_states": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, encoder_feature_dim)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "token_embedding": {
                    "dim": model_config.attention_decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_c": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "att_context": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "accum_att_weights": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_states", "encoder_states:size1"],
        output_names=[
            "token_embedding",
            "lstm_state_c",
            "lstm_state_h",
            "att_context",
            "accum_att_weights",
            "accum_att_weights:size1",
        ],
        metadata={
            "token_embedding": "token_embedding",
            "lstm_state_c": "lstm_state_c",
            "lstm_state_h": "lstm_state_h",
            "att_context": "att_context",
            "accum_att_weights": "accum_att_weights",
        },
    )


def export_attention_state_updater(model_config: CombinationModelConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(
        model_class=CombinationModelAttentionStateUpdater, model_config=model_config
    )

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_attention_state_updater_forward_step.__module__}.{_attention_state_updater_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "encoder_time_dim": CodeWrapper('Dim(dimension=None, description="EncTime")'),
            "encoder_feature_dim": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.attention_decoder_config.attention_cfg.attention_dim + 1}, description="EncTime")'
            ),
            "accum_att_weights_dim": CodeWrapper('Dim(dimension=1, description="AttWeights")'),
            "extern_data": {
                "encoder_states": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, encoder_feature_dim)"),
                    "dtype": "float32",
                },
                "token": {
                    "dim": model_config.attention_decoder_config.vocab_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "dtype": "int32",
                },
                "lstm_state_c_in": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h_in": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "att_context_in": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "accum_att_weights_in": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "token_embedding_out": {
                    "dim": model_config.attention_decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_c_out": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h_out": {
                    "dim": model_config.attention_decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "att_context_out": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "accum_att_weights_out": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=[
            "encoder_states",
            "token",
            "lstm_state_c_in",
            "lstm_state_h_in",
            "att_context_in",
            "accum_att_weights_in",
            "accum_att_weights_in:size1",
        ],
        output_names=[
            "token_embedding_out",
            "lstm_state_c_out",
            "lstm_state_h_out",
            "att_context_out",
            "accum_att_weights_out",
            "accum_att_weights_out:size1",
        ],
        metadata={
            "lstm_state_c_in": "lstm_state_c",
            "lstm_state_h_in": "lstm_state_h",
            "att_context_in": "att_context",
            "accum_att_weights_in": "accum_att_weights",
            "token_embedding_out": "token_embedding",
            "lstm_state_c_out": "lstm_state_c",
            "lstm_state_h_out": "lstm_state_h",
            "att_context_out": "att_context",
            "accum_att_weights_out": "accum_att_weights",
        },
    )
