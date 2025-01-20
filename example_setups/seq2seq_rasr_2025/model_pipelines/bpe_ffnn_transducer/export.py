__all__ = ["export_encoder", "export_scorer"]

from i6_core.returnn import PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.export import export_model as _export_model
from ..common.imports import get_model_serializers
from .pytorch_modules import (
    FFNNTransducerConfig,
    FFNNTransducerEncoder,
    FFNNTransducerRecogConfig,
    FFNNTransducerScorer,
)

# -----------------------
# --- Forward steps -----
# -----------------------


def _encoder_forward_step(*, model: FFNNTransducerEncoder, extern_data: TensorDict, **_):
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


def _scorer_forward_step(*, model: FFNNTransducerScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, F]
    assert encoder_state is not None
    history = extern_data["history"].raw_tensor  # [B, S]
    assert history is not None

    scores = model.forward(
        encoder_state=encoder_state,
        history=history,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)


# -----------------------
# --- Export routines ---
# -----------------------


def export_encoder(model_config: FFNNTransducerConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=FFNNTransducerEncoder, model_config=model_config)

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


def export_scorer(model_config: FFNNTransducerRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=FFNNTransducerScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "extern_data": {
                "encoder_state": {
                    "shape": (1, model_config.enc_dim),
                    "dim": model_config.enc_dim,
                    "dtype": "float32",
                    "batch_dim_axis": None,
                },
                "history": {
                    "dim": model_config.target_size,
                    "time_dim_axis": None,
                    "sparse": True,
                    "shape": (model_config.context_history_size,),
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
