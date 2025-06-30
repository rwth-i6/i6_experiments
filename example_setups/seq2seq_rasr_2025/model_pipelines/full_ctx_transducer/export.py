__all__ = ["export_encoder", "export_scorer", "export_state_initializer", "export_state_updater"]

from i6_core.returnn import PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.export import export_model as _export_model
from ..common.imports import get_model_serializers
from .pytorch_modules import (
    LstmTransducerConfig,
    LstmTransducerRecogConfig,
    LstmTransducerEncoder,
    LstmTransducerScorer,
    LstmTransducerStateInitializer,
    LstmTransducerStateUpdater,
)

# -----------------------
# --- Forward steps -----
# -----------------------


def _encoder_forward_step(*, model: LstmTransducerEncoder, extern_data: TensorDict, **_):
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


def _scorer_forward_step(*, model: LstmTransducerScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, E]
    assert encoder_state is not None
    lstm_out = extern_data["lstm_out"].raw_tensor  # [B, P]
    assert lstm_out is not None

    scores = model.forward(
        encoder_state=encoder_state,
        lstm_out=lstm_out,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)


def _state_initializer_forward_step(*, model: LstmTransducerStateInitializer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_states = extern_data["encoder_states"].raw_tensor
    assert encoder_states is not None
    assert extern_data["encoder_states"].dims[1].dyn_size_ext is not None
    encoder_states_size = extern_data["encoder_states"].dims[1].dyn_size_ext.raw_tensor
    assert encoder_states_size is not None

    lstm_out, lstm_h, lstm_c = model.forward()

    run_ctx.mark_as_output(name="lstm_out", tensor=lstm_out)
    run_ctx.mark_as_output(name="lstm_h", tensor=lstm_h)
    run_ctx.mark_as_output(name="lstm_c", tensor=lstm_c)


def _state_updater_forward_step(*, model: LstmTransducerStateUpdater, extern_data: TensorDict, **_):
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


def export_encoder(model_config: LstmTransducerConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmTransducerEncoder, model_config=model_config)

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
                    "dim": model_config.enc_dim,
                    "dtype": "float32",
                },
            },
            "backend": "torch",
        },
        input_names=["audio_samples", "audio_samples:size1"],
        output_names=["encoder_states"],
    )


def export_scorer(model_config: LstmTransducerRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmTransducerScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "lstm_out": {
                    "dim": model_config.pred_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim": model_config.target_size,
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


def export_state_initializer(model_config: LstmTransducerConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmTransducerStateInitializer, model_config=model_config)

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
                    "dim": model_config.pred_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_h": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
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


def export_state_updater(model_config: LstmTransducerConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmTransducerStateUpdater, model_config=model_config)

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
                    "dim": model_config.target_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "dtype": "int32",
                },
                "lstm_h_in": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c_in": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "lstm_out": {
                    "dim": model_config.pred_dim,
                    "time_dim_axis": None,
                    "dtype": "float32",
                },
                "lstm_h_out": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
                    "time_dim_axis": None,
                    "batch_dim_axis": 1,
                    "dtype": "float32",
                },
                "lstm_c_out": {
                    "dim": model_config.pred_dim,
                    "shape": (model_config.pred_num_layers, model_config.pred_dim),
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
