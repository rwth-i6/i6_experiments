__all__ = ["export_model"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import ConformerCTCRecogConfig, ConformerCTCRecogExportModel


def export_model(model_config: ConformerCTCRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=ConformerCTCRecogExportModel, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_model_forward_step.__module__}.{_model_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "tensor_in_time": CodeWrapper('Tensor(name="in_time", dims=[batch_dim], dtype="int32")'),
            "dim_in_time": CodeWrapper('Dim(dimension=tensor_in_time, name="in_time")'),
            "dim_feature": CodeWrapper(f'Dim(dimension={model_config.logmel_cfg.num_filters}, name="feature")'),
            "tensor_out_time": CodeWrapper('Tensor(name="out_time", dims=[batch_dim], dtype="int32")'),
            "dim_out_time": CodeWrapper('Dim(dimension=tensor_out_time, name="out_time")'),
            "dim_target": CodeWrapper(f'Dim(dimension={model_config.target_size}, name="target")'),
            "extern_data": {
                "features": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_in_time, dim_feature)"),
                    "dtype": "float32",
                },
            },
            "model_outputs": {
                "scores": {
                    "dim_tags": CodeWrapper("(batch_dim, dim_out_time, dim_target)"),
                    "dtype": "float32",
                },
            },
        },
        input_names=["features", "features:size1"],
        output_names=["scores", "scores:size1"],
    )


def _model_forward_step(*, model: ConformerCTCRecogExportModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    features = extern_data["features"].raw_tensor  # [B, T, 1]
    assert features is not None

    assert extern_data["features"].dims[1].dyn_size_ext is not None
    features_size = extern_data["features"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert features_size is not None

    scores, scores_size = model(
        features=features,
        features_size=features_size,
    )

    assert run_ctx.expected_outputs is not None
    assert run_ctx.expected_outputs["scores"].dims[1].dyn_size_ext is not None
    run_ctx.expected_outputs["scores"].dims[1].dyn_size_ext.raw_tensor = scores_size

    run_ctx.mark_as_output(name="scores", tensor=scores)
