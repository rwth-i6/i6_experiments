__all__ = ["export_encoder", "export_scorer"]

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.config import CodeWrapper
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.onnx_export import export_model as _export_model
from ..common.serializers import get_model_serializers
from .pytorch_modules import (
    ConformerCTCMultiOutputConfig,
    ConformerCTCMultiOutputEncoderModel,
    ConformerCTCMultiOutputScorerConfig,
    ConformerCTCMultiOutputScorerModel,
)

# -----------------------
# --- Export routines ---
# -----------------------


def export_encoder(model_config: ConformerCTCMultiOutputConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(
        model_class=ConformerCTCMultiOutputEncoderModel, model_config=model_config
    )

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
                f'Dim(dimension={model_config.dim * len(model_config.layer_idx_target_size_list)}, name="enc")'
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
            "backend": "torch",
        },
        input_names=["features", "features:size1"],
        output_names=["enc_out", "enc_out:size1"],
    )


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


def _encoder_forward_step(*, model: ConformerCTCMultiOutputEncoderModel, extern_data: TensorDict, **_):
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
    )  # [B, T, O*E], [B]

    assert run_ctx.expected_outputs is not None
    assert run_ctx.expected_outputs["enc_out"].dims[1].dyn_size_ext is not None
    run_ctx.expected_outputs["enc_out"].dims[1].dyn_size_ext.raw_tensor = encoder_states_size

    run_ctx.mark_as_output(name="enc_out", tensor=encoder_states)


def _scorer_forward_step(*, model: ConformerCTCMultiOutputScorerModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    encoder_state = extern_data["encoder_state"].raw_tensor  # [B, F]
    assert encoder_state is not None

    scores = model(encoder_state=encoder_state)

    run_ctx.mark_as_output(name="scores", tensor=scores)
