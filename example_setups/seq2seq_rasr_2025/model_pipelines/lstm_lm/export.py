__all__ = ["export_scorer", "export_state_initializer", "export_state_updater"]

from i6_core.returnn import PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.export import export_model as _export_model
from ..common.imports import get_model_serializers
from .pytorch_modules import LstmLmConfig, LstmLmScorer, LstmLmStateInitializer, LstmLmStateUpdater

# -----------------------
# --- Forward steps -----
# -----------------------


def _scorer_forward_step(*, model: LstmLmScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out = extern_data["lstm_out"].raw_tensor
    assert lstm_out is not None

    scores = model.forward(lstm_out=lstm_out)

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _state_initializer_forward_step(*, model: LstmLmStateInitializer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    lstm_out, lstm_h, lstm_c = model.forward()

    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
    run_ctx.mark_as_output(name="lstm_h", tensor=lstm_h)
    run_ctx.mark_as_output(name="lstm_c", tensor=lstm_c)


def _state_updater_forward_step(*, model: LstmLmStateUpdater, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    token = extern_data["token"].raw_tensor
    assert token is not None
    lstm_c_in = extern_data["lstm_c_in"].raw_tensor
    assert lstm_c_in is not None
    lstm_h_in = extern_data["lstm_h_in"].raw_tensor
    assert lstm_h_in is not None

    lstm_out, lstm_h_out, lstm_c_out = model.forward(token=token, lstm_h=lstm_h_in, lstm_c=lstm_c_in)

    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
    run_ctx.mark_as_output(name="lstm_h_out", tensor=lstm_h_out)
    run_ctx.mark_as_output(name="lstm_c_out", tensor=lstm_c_out)


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
            "extern_data": {
                "lstm_out": {
                    "dim": model_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.vocab_size + 1,
                    "time_dim_axis": None,
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
            "extern_data": {},
            "model_outputs": {
                "lstm_out": {
                    "dim": model_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_h": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
            },
        },
        input_names=[],
        output_names=["lstm_out", "lstm_h", "lstm_c"],
        metadata={
            "lstm_out": "lstm_out",
            "lstm_h": "lstm_h",
            "lstm_c": "lstm_c",
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
            "extern_data": {
                "token": {
                    "dim": model_config.vocab_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "dtype": "int32",
                },
                "lstm_h_in": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c_in": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "lstm_out": {
                    "dim": model_config.lstm_hidden_size,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_h_out": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c_out": {
                    "dim": model_config.lstm_hidden_size,
                    "shape": (model_config.lstm_layers, model_config.lstm_hidden_size),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
            },
        },
        input_names=[
            "token",
            "lstm_h_in",
            "lstm_c_in",
        ],
        output_names=[
            "lstm_out",
            "lstm_h_out",
            "lstm_c_out",
        ],
        metadata={
            "lstm_h_in": "lstm_h",
            "lstm_c_in": "lstm_c",
            "lstm_out": "lstm_out",
            "lstm_h_out": "lstm_h",
            "lstm_c_out": "lstm_c",
        },
    )
