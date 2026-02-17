__all__ = ["export_model"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import ConformerCTCRecogConfig, ConformerCTCRecogExportModel


def _model_forward_step(*, model: ConformerCTCRecogExportModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf
    from returnn.tensor.dim import batch_dim

    run_ctx = rf.get_run_ctx()

    features = extern_data["features"].raw_tensor  # [B, T, 1]
    assert features is not None

    assert extern_data["features"].dims[1].dyn_size_ext is not None
    features_size = extern_data["features"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert features_size is not None

    scores, scores_size = model.forward(
        features=features,
        features_size=features_size,
    )

    run_ctx.mark_as_output(
        name="enc_out",
        tensor=scores,
    )
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["enc_out"].dims[1].dyn_size_ext = rf.Tensor(
            "enc_out_time", dims=[batch_dim], raw_tensor=scores_size.long(), dtype="int64"
        )


def export_model(model_config: ConformerCTCRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=ConformerCTCRecogExportModel, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_model_forward_step.__module__}.{_model_forward_step.__name__}", import_as="forward_step"
        ),
        checkpoint=checkpoint,
        returnn_config_dict={
            "dim_enc_out_time": CodeWrapper('Dim(name="enc_out_time", dimension=None)'),
            "dim_enc_out_feature": CodeWrapper(f'Dim(name="enc_out_feature", dimension={model_config.target_size})'),
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
