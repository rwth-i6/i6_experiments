__all__ = ["export_scorer"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import (
    ConformerCTCMultiOutputScorerConfig,
    ConformerCTCMultiOutputScorerModel,
)

# -----------------------
# --- Export routines ---
# -----------------------


def export_scorer(model_config: ConformerCTCMultiOutputScorerConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=ConformerCTCMultiOutputScorerModel, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_scorer_forward_step.__module__}.{_scorer_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_in_feature": CodeWrapper(
                f'Dim(dimension={model_config.dim * len(model_config.layer_idx_target_size_list)}, name="in_feature")'
            ),
            "dim_target": CodeWrapper(
                f'Dim(dimension={model_config.layer_idx_target_size_list[model_config.output_idx][1]}, name="target")'
            ),
            "extern_data": {
                "encoder_state": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_in_feature)"),
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


def _scorer_forward_step(*, model: ConformerCTCMultiOutputScorerModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, F]
    assert encoder_state is not None

    scores = model(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)
