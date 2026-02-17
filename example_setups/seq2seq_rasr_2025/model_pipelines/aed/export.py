__all__ = ["export_encoder", "export_scorer", "export_state_initializer", "export_state_updater", "export_ctc_scorer"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import AEDCTCScorer, AEDConfig, AEDExportEncoder, AEDScorer, AEDStateInitializer, AEDStateUpdater

# -----------------------
# --- Forward steps -----
# -----------------------


def _encoder_forward_step(*, model: AEDExportEncoder, extern_data: TensorDict, **_):
    import returnn.frontend as rf
    from returnn.tensor.dim import batch_dim

    run_ctx = rf.get_run_ctx()

    features = extern_data["features"].raw_tensor  # [B, T, F]
    assert features is not None

    assert extern_data["features"].dims[1].dyn_size_ext is not None
    features_size = extern_data["features"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert features_size is not None

    encoder_states, encoder_states_size = model.forward(
        features=features,
        features_size=features_size,
    )

    run_ctx.mark_as_output(name="enc_out", tensor=encoder_states)
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["enc_out"].dims[1].dyn_size_ext = rf.Tensor(
            "enc_out_time", dims=[batch_dim], raw_tensor=encoder_states_size.long(), dtype="int64"
        )


def _scorer_forward_step(*, model: AEDScorer, extern_data: TensorDict, **_):
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


def _state_initializer_forward_step(*, model: AEDStateInitializer, extern_data: TensorDict, **_):
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

    run_ctx.mark_as_output(name="accum_att_weights", tensor=accum_att_weights)
    run_ctx.mark_as_output(name="att_context", tensor=att_context)
    run_ctx.mark_as_output(name="lstm_state_c", tensor=lstm_state_c)
    run_ctx.mark_as_output(name="lstm_state_h", tensor=lstm_state_h)
    run_ctx.mark_as_output(name="token_embedding", tensor=token_embedding)


def _state_updater_forward_step(*, model: AEDStateUpdater, extern_data: TensorDict, **_):
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

    run_ctx.mark_as_output(name="accum_att_weights_out", tensor=accum_att_weights_out)
    run_ctx.mark_as_output(name="att_context_out", tensor=att_context_out)
    run_ctx.mark_as_output(name="lstm_state_c_out", tensor=lstm_state_c_out)
    run_ctx.mark_as_output(name="lstm_state_h_out", tensor=lstm_state_h_out)
    run_ctx.mark_as_output(name="token_embedding_out", tensor=token_embedding_out)


def _ctc_scorer_forward_step(*, model: AEDCTCScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor
    assert encoder_state is not None

    scores = model.forward(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)


# -----------------------
# --- Export routines ---
# -----------------------


def export_encoder(model_config: AEDConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=AEDExportEncoder, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_encoder_forward_step.__module__}.{_encoder_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_enc_out_time": CodeWrapper('Dim(name="enc_out_time", dimension=None)'),
            "dim_enc_out_feature": CodeWrapper(
                f'Dim(name="enc_out_feature", dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1})'
            ),
            "extern_data": {
                "features": {
                    "dim": model_config.logmel_cfg.num_filters,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "enc_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_enc_out_time, dim_enc_out_feature)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["features", "features:size1"],
        output_names=["enc_out"],
    )


def export_scorer(model_config: AEDConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=AEDScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "att_context": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "token_embedding": {
                    "dim": model_config.decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.label_target_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["att_context", "lstm_state_h", "token_embedding"],
        output_names=["scores"],
        metadata={
            "att_context": "att_context",
            "lstm_state_h": "lstm_state_h",
            "token_embedding": "token_embedding",
        },
    )


def export_state_initializer(model_config: AEDConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=AEDStateInitializer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_initializer_forward_step.__module__}.{_state_initializer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "encoder_time_dim": CodeWrapper('Dim(dimension=None, description="EncTime")'),
            "encoder_feature_dim": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, description="EncFeature")'
            ),
            "accum_att_weights_dim": CodeWrapper('Dim(dimension=1, description="AttWeights")'),
            "extern_data": {
                "encoder_states": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, encoder_feature_dim)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "accum_att_weights": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
                "att_context": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_c": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "token_embedding": {
                    "dim": model_config.decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_states", "encoder_states:size1"],
        output_names=[
            "accum_att_weights",
            "accum_att_weights:size1",
            "att_context",
            "lstm_state_c",
            "lstm_state_h",
            "token_embedding",
        ],
        metadata={
            "accum_att_weights": "accum_att_weights",
            "att_context": "att_context",
            "lstm_state_c": "lstm_state_c",
            "lstm_state_h": "lstm_state_h",
            "token_embedding": "token_embedding",
        },
    )


def export_state_updater(model_config: AEDConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=AEDStateUpdater, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_updater_forward_step.__module__}.{_state_updater_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        extra_imports=[Import("returnn.tensor.Tensor")],
        returnn_config_dict={
            "encoder_batch_dim": CodeWrapper('Dim(dimension=1, description="EncBatch")'),
            "encoder_time_dim": CodeWrapper(
                'Dim(dimension=None, dyn_size_ext=Tensor("EncTime", dims=[encoder_batch_dim], dtype="int32"), description="EncTime")'
            ),
            "encoder_feature_dim": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, description="EncFeature")'
            ),
            "accum_att_weights_dim": CodeWrapper('Dim(dimension=1, description="AttWeights")'),
            "extern_data": {
                "accum_att_weights_in": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
                "att_context_in": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "encoder_states": {
                    "dim_tags": CodeWrapper("(encoder_batch_dim, encoder_time_dim, encoder_feature_dim)"),
                    "dtype": "float32",
                },
                "lstm_state_c_in": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h_in": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "token": {
                    "dim": model_config.label_target_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "accum_att_weights_out": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_time_dim, accum_att_weights_dim)"),
                    "dtype": "float32",
                },
                "att_context_out": {
                    "dim": model_config.enc_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_c_out": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_state_h_out": {
                    "dim": model_config.decoder_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "token_embedding_out": {
                    "dim": model_config.decoder_config.target_embed_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=[
            "accum_att_weights_in",
            "accum_att_weights_in:size1",
            "att_context_in",
            "encoder_states",
            "lstm_state_c_in",
            "lstm_state_h_in",
            "token",
        ],
        output_names=[
            "accum_att_weights_out",
            "accum_att_weights_out:size1",
            "att_context_out",
            "lstm_state_c_out",
            "lstm_state_h_out",
            "token_embedding_out",
        ],
        metadata={
            "accum_att_weights_in": "accum_att_weights",
            "accum_att_weights_out": "accum_att_weights",
            "att_context_in": "att_context",
            "att_context_out": "att_context",
            "lstm_state_c_in": "lstm_state_c",
            "lstm_state_c_out": "lstm_state_c",
            "lstm_state_h_in": "lstm_state_h",
            "lstm_state_h_out": "lstm_state_h",
            "token_embedding_out": "token_embedding",
        },
    )


def export_ctc_scorer(model_config: AEDConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=AEDCTCScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_ctc_scorer_forward_step.__module__}.{_ctc_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "encoder_feature_dim": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, description="EncFeature")'
            ),
            "extern_data": {
                "encoder_state": {
                    "dim_tags": CodeWrapper("(batch_dim, encoder_feature_dim)"),
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.label_target_size + 1,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_state"],
        output_names=["scores"],
    )
