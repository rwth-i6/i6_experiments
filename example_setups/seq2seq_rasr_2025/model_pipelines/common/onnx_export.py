__all__ = ["export_model"]

from typing import Dict, List, Optional

from i6_core.returnn import PtCheckpoint
from i6_core.returnn.compile import TorchOnnxExportJob
from i6_core.returnn.config import ReturnnConfig
from i6_experiments.common.setups.serialization import Collection, Import
from sisyphus import Job, Task, tk

from ...tools import returnn_python_exe, returnn_root
from ..common.serializers import recipe_imports


def export_model(
    model_serializers: Collection,
    forward_step_import: Import,
    checkpoint: PtCheckpoint,
    returnn_config_dict: dict,
    input_names: List[str],
    output_names: List[str],
    extra_imports: Optional[List[Import]] = None,
    metadata: Optional[dict] = None,
) -> tk.Path:
    if extra_imports is None:
        extra_imports = []
    returnn_config = ReturnnConfig(
        config={
            **returnn_config_dict,
            "backend": "torch",
        },
        python_prolog=recipe_imports
        + [
            Import("returnn.tensor.dim.batch_dim"),
            Import("returnn.tensor.dim.Dim"),
            Import("returnn.tensor.Tensor"),
        ]
        + extra_imports,
        python_epilog=[
            model_serializers,
            forward_step_import,
        ],  # type: ignore
        sort_config=False,
    )
    export_job = TorchOnnxExportJob(
        returnn_config=returnn_config,
        checkpoint=checkpoint,
        input_names=input_names,
        output_names=output_names,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
    )
    export_job.update_rqmt("run", {"mem": 16})

    exported_model = export_job.out_onnx_model

    if metadata is not None:
        metadata_job = AddOnnxMetadataJob(model_path=exported_model, metadata=metadata)
        metadata_job.update_rqmt("run", {"mem": 16})
        exported_model = metadata_job.out_model

    return exported_model


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
        import os
        import shutil

        import onnx
        from onnx import external_data_helper

        model_path = self.model_path.get_path()
        out_model_path = self.out_model.get_path()
        model = onnx.load(model_path, load_external_data=False)
        for key, value in self.metadata.items():
            meta = model.metadata_props.add()
            meta.key = key
            meta.value = value

        model_dir = os.path.dirname(model_path)
        out_model_dir = os.path.dirname(out_model_path)
        external_data_locations = {
            entry.value
            for tensor in external_data_helper._get_all_tensors(model)
            if external_data_helper.uses_external_data(tensor)
            for entry in tensor.external_data
            if entry.key == "location"
        }
        for location in external_data_locations:
            if os.path.isabs(location):
                raise ValueError(f"External ONNX tensor data location must be relative, got: {location!r}")
            src = os.path.normpath(os.path.join(model_dir, location))
            dst = os.path.normpath(os.path.join(out_model_dir, location))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)

        onnx.save(model, out_model_path)
        onnx.checker.check_model(out_model_path)
