__all__ = ["export_model"]

from typing import Dict, List, Optional

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.compile import TorchOnnxExportJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import Job, Task, tk

from ...tools import returnn_python_exe, returnn_root
from ..common.imports import recipe_imports


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


def export_model(
    model_serializers: Collection,
    forward_step_import: Import,
    checkpoint: PtCheckpoint,
    returnn_config_dict: dict,
    input_names: List[str],
    output_names: List[str],
    metadata: Optional[dict] = None,
) -> tk.Path:
    returnn_config = ReturnnConfig(
        config={
            **returnn_config_dict,
            "backend": "torch",
        },
        python_prolog=recipe_imports
        + [
            Import("returnn.tensor.dim.Dim"),
            Import("returnn.tensor.dim.batch_dim"),
        ],
        python_epilog=[
            model_serializers,
            forward_step_import,
        ],  # type: ignore
    )

    exported_model = TorchOnnxExportJob(
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        input_names=input_names,
        output_names=output_names,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    ).out_onnx_model

    if metadata is not None:
        exported_model = AddOnnxMetadataJob(model_path=exported_model, metadata=metadata).out_model

    return exported_model
