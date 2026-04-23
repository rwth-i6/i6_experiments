__all__ = ["export_scorer", "export_state_initializer", "export_state_updater"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import LstmLmConfig, LstmLmScorer, LstmLmStateInitializer, LstmLmStateUpdater

# -----------------------
# --- Export routines ---
# -----------------------


def export_scorer(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.lstm_hidden_size}, name="lstm")'),
            "dim_vocab": CodeWrapper(f'Dim(dimension={model_config.vocab_size}, name="vocab")'),
            "extern_data": {
                "lstm_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_vocab)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["lstm_out"],
        output_names=["scores"],
        metadata={
            "lstm_out": "lstm_out",
        },
    )


def export_state_initializer(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmStateInitializer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_initializer_forward_step.__module__}.{_state_initializer_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_state_batch": CodeWrapper('Dim(dimension=1, name="state_batch")'),
            "dim_lstm_layers": CodeWrapper(f'Dim(dimension={model_config.lstm_layers}, name="lstm_layers")'),
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.lstm_hidden_size}, name="lstm")'),
            "extern_data": {},
            "model_outputs": {
                "lstm_c": {
                    "dim_tags": CodeWrapper("(dim_state_batch, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h": {
                    "dim_tags": CodeWrapper("(dim_state_batch, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_out": {
                    "dim_tags": CodeWrapper("(dim_state_batch, dim_lstm)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=[],
        output_names=["lstm_c", "lstm_h", "lstm_out"],
        metadata={
            "lstm_c": "lstm_c",
            "lstm_h": "lstm_h",
            "lstm_out": "lstm_out",
        },
    )


def export_state_updater(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmStateUpdater, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_state_updater_forward_step.__module__}.{_state_updater_forward_step.__name__}",
            import_as="forward_step",
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_lstm_layers": CodeWrapper(f'Dim(dimension={model_config.lstm_layers}, name="lstm_layers")'),
            "dim_lstm": CodeWrapper(f'Dim(dimension={model_config.lstm_hidden_size}, name="lstm")'),
            "dim_vocab": CodeWrapper(f'Dim(dimension={model_config.vocab_size}, name="vocab")'),
            "extern_data": {
                "lstm_c_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h_in": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "token": {
                    "dim_tags": CodeWrapper("(batch_dim,)"),
                    "sparse_dim": CodeWrapper("dim_vocab"),
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "lstm_c_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_h_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm_layers, dim_lstm)"),
                    "dtype": "float32",
                },
                "lstm_out": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_lstm)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=[
            "lstm_c_in",
            "lstm_h_in",
            "token",
        ],
        output_names=[
            "lstm_c_out",
            "lstm_h_out",
            "lstm_out",
        ],
        metadata={
            "lstm_c_in": "lstm_c",
            "lstm_c_out": "lstm_c",
            "lstm_h_in": "lstm_h",
            "lstm_h_out": "lstm_h",
            "lstm_out": "lstm_out",
        },
    )


# -----------------------
# --- Forward steps -----
# -----------------------


def _scorer_forward_step(*, model: LstmLmScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out = extern_data["lstm_out"].raw_tensor
    assert lstm_out is not None

    scores = model(lstm_out=lstm_out)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _state_initializer_forward_step(*, model: LstmLmStateInitializer, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out, lstm_h, lstm_c = model()

    run_ctx.mark_as_output(name="lstm_c", tensor=lstm_c)
    run_ctx.mark_as_output(name="lstm_h", tensor=lstm_h)
    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)


def _state_updater_forward_step(*, model: LstmLmStateUpdater, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_c_in = extern_data["lstm_c_in"].raw_tensor
    assert lstm_c_in is not None

    lstm_h_in = extern_data["lstm_h_in"].raw_tensor
    assert lstm_h_in is not None

    token = extern_data["token"].raw_tensor
    assert token is not None

    lstm_out, lstm_h_out, lstm_c_out = model(token=token, lstm_h=lstm_h_in, lstm_c=lstm_c_in)

    run_ctx.mark_as_output(name="lstm_c_out", tensor=lstm_c_out)
    run_ctx.mark_as_output(name="lstm_h_out", tensor=lstm_h_out)
    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
