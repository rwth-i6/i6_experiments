from typing import Dict
from i6_core.returnn.compile import TorchOnnxExportJob
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from i6_experiments.common.setups.serialization import Import
from sisyphus import Job, Task, tk

from ....tools import returnn_python_exe, returnn_root
from ..pytorch_modules import (
    LstmLmScorerModel,
    LstmLmStateInitializerModel,
    LstmLmStateUpdaterModel,
    LstmLmConfig,
)
from .returnn_steps.export import scorer_forward_step, state_initializer_forward_step, state_updater_forward_step
from ...common.imports import get_model_serializers, recipe_imports


class AddOnnxMetadataJob(Job):
    """
    Add metadata to a given Onnx model
    """

    def __init__(
        self,
        model_path: tk.Path,
        metadata: Dict[str, str],
    ):
        """
        :param model_path: path of the onnx model for which add metadata
        :param metadata: model metadata dict
        """
        self.model_path = model_path
        self.metadata = metadata

        self.out_model = self.output_path("model.onnx")

    def tasks(self):
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        import onnx

        model = onnx.load(self.model_path)
        for key, value in self.metadata.items():
            meta = model.metadata_props.add()
            meta.key = key
            meta.value = value

        onnx.checker.check_model(model)
        onnx.save(model, self.out_model.get_path())


def export_scorer(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmScorerModel, model_config=model_config)

    returnn_config = ReturnnConfig(
        config={
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
            "backend": "torch",
        },
        python_prolog=recipe_imports,
        python_epilog=[
            *model_serializers,
            Import(
                code_object_path=f"{scorer_forward_step.__module__}.{scorer_forward_step.__name__}",
                import_as="forward_step",
            ),
        ],  # type: ignore
    )

    export_job = TorchOnnxExportJob(
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        input_names=["lstm_out"],
        output_names=["scores"],
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )

    add_metadata_job = AddOnnxMetadataJob(
        export_job.out_onnx_model,
        {
            "lstm_out": "lstm_out",
        },
    )

    return add_metadata_job.out_model


def export_state_initializer(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmStateInitializerModel, model_config=model_config)

    returnn_config = ReturnnConfig(
        config={
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
            "backend": "torch",
        },
        python_prolog=recipe_imports,
        python_epilog=[
            *model_serializers,
            Import(
                code_object_path=f"{state_initializer_forward_step.__module__}.{state_initializer_forward_step.__name__}",
                import_as="forward_step",
            ),
        ],  # type: ignore
    )

    export_job = TorchOnnxExportJob(
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        input_names=[],
        output_names=[
            "lstm_out",
            "lstm_h",
            "lstm_c",
        ],
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )

    add_metadata_job = AddOnnxMetadataJob(
        export_job.out_onnx_model,
        {
            "lstm_out": "lstm_out",
            "lstm_h": "lstm_h",
            "lstm_c": "lstm_c",
        },
    )

    return add_metadata_job.out_model


def export_state_updater(model_config: LstmLmConfig, checkpoint: PtCheckpoint) -> tk.Path:
    model_serializers = get_model_serializers(model_class=LstmLmStateUpdaterModel, model_config=model_config)

    returnn_config = ReturnnConfig(
        config={
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
            "backend": "torch",
        },
        python_prolog=recipe_imports,
        python_epilog=[
            *model_serializers,
            Import(
                code_object_path=f"{state_updater_forward_step.__module__}.{state_updater_forward_step.__name__}",
                import_as="forward_step",
            ),
        ],  # type: ignore
    )

    export_job = TorchOnnxExportJob(
        returnn_config=returnn_config,
        checkpoint=checkpoint,
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
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )

    add_metadata_job = AddOnnxMetadataJob(
        export_job.out_onnx_model,
        {
            "lstm_h_in": "lstm_h",
            "lstm_c_in": "lstm_c",
            "lstm_out": "lstm_out",
            "lstm_h_out": "lstm_h",
            "lstm_c_out": "lstm_c",
        },
    )

    return add_metadata_job.out_model
