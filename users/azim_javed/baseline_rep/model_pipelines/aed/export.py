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
            "tensor_in_time": CodeWrapper('Tensor(name="in_time", dims=[batch_dim], dtype="int32")'),
            "dim_in_time": CodeWrapper('Dim(dimension=tensor_in_time, name="in_time")'),
            "dim_feature": CodeWrapper(f'Dim(dimension={model_config.logmel_cfg.num_filters}, name="feature")'),
            "tensor_out_time": CodeWrapper('Tensor(name="out_time", dims=[batch_dim], dtype="int32")'),
            "dim_out_time": CodeWrapper('Dim(dimension=tensor_out_time, name="out_time")'),
            "dim_enc": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, name="enc")'
            ),
            "extern_data": {
                "features": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_in_time, dim_feature)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "enc_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_out_time, dim_enc)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["features", "features:size1"],
        output_names=["enc_out", "enc_out:size1"],
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
            "dim_att_context": CodeWrapper(f'Dim(dimension={model_config.enc_dim}, name="att_context")'),
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.decoder_config.lstm_hidden_size}, name="lstm")'),
            "dim_embed": CodeWrapper(f'Dim(dimension={model_config.decoder_config.target_embed_dim}, name="embed")'),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.label_target_size}, name="target")'),
            "extern_data": {
                "att_context": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_att_context)"),
                    "dtype": "float32",
                },
                "lstm_h": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
                "token_embedding": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_embed)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_target)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["att_context", "lstm_h", "token_embedding"],
        output_names=["scores"],
        metadata={
            "att_context": "att_context",
            "lstm_h": "lstm_h",
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
            "dim_enc_batch": CodeWrapper('Dim(dimension=1, name="enc_batch")'),
            "tensor_enc_time": CodeWrapper('Tensor(name="enc_time", dims=[dim_enc_batch], dtype="int32")'),
            "dim_enc_time": CodeWrapper('Dim(dimension=tensor_enc_time, name="enc_time")'),
            "dim_enc_feature": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, name="enc_feature")'
            ),
            "dim_accum": CodeWrapper('Dim(dimension=1, name="accum")'),
            "dim_context": CodeWrapper(f'Dim(dimension={model_config.enc_dim}, name="context")'),
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.decoder_config.lstm_hidden_size}, name="lstm")'),
            "dim_embed": CodeWrapper(f'Dim(dimension={model_config.decoder_config.target_embed_dim}, name="embed")'),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.label_target_size}, name="target")'),
            "extern_data": {
                "encoder_states": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_enc_time, dim_enc_feature)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "accum_att_weights": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_enc_time, dim_accum)"),
                    "dtype": "float32",
                },
                "att_context": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_context)"),
                    "dtype": "float32",
                },
                "lstm_c": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_lstm)"),
                    "dtype": "float32",
                },
                "token_embedding": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_embed)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_states", "encoder_states:size1"],
        output_names=[
            "accum_att_weights",
            "accum_att_weights:size1",
            "att_context",
            "lstm_c",
            "lstm_h",
            "token_embedding",
        ],
        metadata={
            "accum_att_weights": "accum_att_weights",
            "att_context": "att_context",
            "lstm_c": "lstm_c",
            "lstm_h": "lstm_h",
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
        returnn_config_dict={
            "dim_enc_batch": CodeWrapper('Dim(dimension=1, name="enc_batch")'),
            "tensor_enc_time": CodeWrapper('Tensor(name="enc_time", dims=[dim_enc_batch], dtype="int32")'),
            "dim_enc_time": CodeWrapper('Dim(dimension=tensor_enc_time, name="enc_time")'),
            "dim_enc_feature": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, name="enc_feature")'
            ),
            "dim_context": CodeWrapper(f'Dim(dimension={model_config.enc_dim}, name="context")'),
            "dim_accum": CodeWrapper('Dim(dimension=1, name="accum")'),
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.decoder_config.lstm_hidden_size}, name="lstm")'),
            "dim_embed": CodeWrapper(f'Dim(dimension={model_config.decoder_config.target_embed_dim}, name="embed")'),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.label_target_size}, name="target")'),
            "extern_data": {
                "accum_att_weights_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_enc_time, dim_accum)"),
                    "dtype": "float32",
                },
                "att_context_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_context)"),
                    "dtype": "float32",
                },
                "encoder_states": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_enc_time, dim_enc_feature)"),
                    "dtype": "float32",
                },
                "lstm_c_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
                "token": {
                    "dim_tags": CodeWrapper("(batch_dim,)"),
                    "sparse_dim": CodeWrapper("dim_target"),
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "accum_att_weights_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_enc_time, dim_accum)"),
                    "dtype": "float32",
                },
                "att_context_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_context)"),
                    "dtype": "float32",
                },
                "lstm_c_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
                "token_embedding_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_embed)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=[
            "accum_att_weights_in",
            "accum_att_weights_in:size1",
            "att_context_in",
            "encoder_states",
            "lstm_c_in",
            "lstm_h_in",
            "token",
        ],
        output_names=[
            "accum_att_weights_out",
            "accum_att_weights_out:size1",
            "att_context_out",
            "lstm_c_out",
            "lstm_h_out",
            "token_embedding_out",
        ],
        metadata={
            "accum_att_weights_in": "accum_att_weights",
            "accum_att_weights_out": "accum_att_weights",
            "att_context_in": "att_context",
            "att_context_out": "att_context",
            "lstm_c_in": "lstm_c",
            "lstm_c_out": "lstm_c",
            "lstm_h_in": "lstm_h",
            "lstm_h_out": "lstm_h",
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
            "dim_enc_feature": CodeWrapper(
                f'Dim(dimension={model_config.enc_dim + model_config.decoder_config.attention_cfg.attention_dim + 1}, description="enc_feature")'
            ),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.label_target_size + 1}, description="target")'),
            "extern_data": {
                "encoder_state": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_enc_feature)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_target)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["encoder_state"],
        output_names=["scores"],
    )


# -----------------------
# --- Forward steps -----
# -----------------------


def _encoder_forward_step(*, model: AEDExportEncoder, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    features = extern_data["features"].raw_tensor  # [B, T, F]
    assert features is not None

    assert extern_data["features"].dims[1].dyn_size_ext is not None
    features_size = extern_data["features"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert features_size is not None

    encoder_states, encoder_states_size = model(
        features=features,
        features_size=features_size,
    )

    expected = run_ctx.expected_outputs
    assert expected is not None

    assert expected["enc_out"].dims[1].dyn_size_ext is not None
    expected["enc_out"].dims[1].dyn_size_ext.raw_tensor = encoder_states_size

    run_ctx.mark_as_output(name="enc_out", tensor=encoder_states, dims=expected["enc_out"].dims)


def _scorer_forward_step(*, model: AEDScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    att_context = extern_data["att_context"].raw_tensor
    assert att_context is not None

    lstm_h = extern_data["lstm_h"].raw_tensor
    assert lstm_h is not None

    token_embedding = extern_data["token_embedding"].raw_tensor
    assert token_embedding is not None

    scores = model(
        att_context=att_context,
        lstm_state_h=lstm_h,
        token_embedding=token_embedding,
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

    token_embedding, lstm_h, lstm_c, att_context, accum_att_weights = model(
        encoder_states=encoder_states, encoder_states_size=encoder_states_size
    )

    run_ctx.mark_as_output(name="accum_att_weights", tensor=accum_att_weights)
    run_ctx.mark_as_output(name="att_context", tensor=att_context)
    run_ctx.mark_as_output(name="lstm_c", tensor=lstm_c)
    run_ctx.mark_as_output(name="lstm_h", tensor=lstm_h)
    run_ctx.mark_as_output(name="token_embedding", tensor=token_embedding)


def _state_updater_forward_step(*, model: AEDStateUpdater, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    accum_att_weights_in = extern_data["accum_att_weights_in"].raw_tensor
    assert accum_att_weights_in is not None

    att_context_in = extern_data["att_context_in"].raw_tensor
    assert att_context_in is not None

    encoder_states = extern_data["encoder_states"].raw_tensor
    assert encoder_states is not None

    assert extern_data["encoder_states"].dims[1].dyn_size_ext is not None
    encoder_states_size = extern_data["encoder_states"].dims[1].dyn_size_ext.raw_tensor
    assert encoder_states_size is not None

    lstm_c_in = extern_data["lstm_c_in"].raw_tensor
    assert lstm_c_in is not None

    lstm_h_in = extern_data["lstm_h_in"].raw_tensor
    assert lstm_h_in is not None

    token = extern_data["token"].raw_tensor
    assert token is not None

    token_embedding_out, lstm_h_out, lstm_c_out, att_context_out, accum_att_weights_out = model(
        accum_att_weights=accum_att_weights_in,
        att_context=att_context_in,
        encoder_states=encoder_states,
        encoder_states_size=encoder_states_size,
        lstm_state_c=lstm_c_in,
        lstm_state_h=lstm_h_in,
        token=token,
    )

    run_ctx.mark_as_output(name="accum_att_weights_out", tensor=accum_att_weights_out)
    run_ctx.mark_as_output(name="att_context_out", tensor=att_context_out)
    run_ctx.mark_as_output(name="lstm_c_out", tensor=lstm_c_out)
    run_ctx.mark_as_output(name="lstm_h_out", tensor=lstm_h_out)
    run_ctx.mark_as_output(name="token_embedding_out", tensor=token_embedding_out)


def _ctc_scorer_forward_step(*, model: AEDCTCScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor
    assert encoder_state is not None

    scores = model(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)
