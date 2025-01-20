__all__ = ["export_model"]

from i6_core.returnn import PtCheckpoint
from returnn.tensor.tensor_dict import TensorDict
from sisyphus import tk

from i6_experiments.common.setups.serialization import Import

from ..common.export import export_model as _export_model
from ..common.imports import get_model_serializers
from .pytorch_modules import ConformerCTCRecogConfig, ConformerCTCRecogModel


def _model_forward_step(*, model: ConformerCTCRecogModel, extern_data: TensorDict, **_):
    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()

    audio_samples = extern_data["audio_samples"].raw_tensor  # [B, T, 1]
    assert audio_samples is not None

    assert extern_data["audio_samples"].dims[1].dyn_size_ext is not None
    audio_samples_size = extern_data["audio_samples"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert audio_samples_size is not None

    scores = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
    )

    run_ctx.mark_as_output(name="scores", tensor=scores)


def export_model(model_config: ConformerCTCRecogConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=ConformerCTCRecogModel, model_config=model_config)

    return _export_model(
        model_serializers=model_serializers,
        forward_step_import=Import(
            f"{_model_forward_step.__module__}.{_model_forward_step.__name__}", import_as="forward_step"
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
                "scores": {
                    "dim": model_config.target_size,
                    "dtype": "float32",
                },
            },
        },
        input_names=["audio_samples", "audio_samples:size1"],
        output_names=["scores", "scores:size1"],
    )
