__all__ = ["export_scorer"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import FFNNTransducerRecogConfig, FFNNTransducerScorer


# -----------------------
# --- Export routines ---
# -----------------------


def export_scorer(model_config: FFNNTransducerRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=FFNNTransducerScorer, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_enc_batch": CodeWrapper('Dim(dimension=1, name="enc_batch")'),
            "dim_enc_feature": CodeWrapper(f'Dim(dimension={model_config.enc_dim}, name="enc_feature")'),
            "dim_context": CodeWrapper(f'Dim(dimension={model_config.context_history_size}, name="context")'),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.target_size}, name="target")'),
            "extern_data": {
                "encoder_state": {
                    "dim_tags": CodeWrapper("(dim_enc_batch, dim_enc_feature)"),
                    "dtype": "float32",
                },
                "history": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_context)"),
                    "sparse_dim": CodeWrapper("dim_target"),
                    "dtype": "int32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_target)"),
                    "dtype": "float32",
                },
            },
            "backend": "torch",
        },
        input_names=["encoder_state", "history"],
        output_names=["scores"],
    )


# -----------------------
# --- Forward steps -----
# -----------------------


def _scorer_forward_step(*, model: FFNNTransducerScorer, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, V]
    assert encoder_state is not None

    history = extern_data["history"].raw_tensor  # [B, H]
    assert history is not None

    scores = model(
        encoder_state=encoder_state,
        history=history,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)
