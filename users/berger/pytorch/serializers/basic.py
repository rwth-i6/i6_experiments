from typing import Optional

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection,
    PyTorchModel,
)
from i6_experiments.common.setups.serialization import Import


def get_basic_pt_network_serializer(
        model_name: str,
        train_step_name: str,
        model_kwargs: Optional[dict] = None,
        module_import_path: str = "i6_experiments.users.berger.pytorch.models",
        train_step_import_path: str = "i6_experiments.users.berger.pytorch.train_steps",
        additional_serializer_objects: Optional[list] = None,
        additional_packages: Optional[set] = None,
) -> Collection:
    if model_kwargs is None:
        model_kwargs = {}
    if additional_serializer_objects is None:
        additional_serializer_objects = []
    if additional_packages is None:
        additional_packages = set()
    model_import = Import(f"{module_import_path}.{model_name}.Model")
    train_step_import = Import(f"{train_step_import_path}.{train_step_name}.train_step")

    pytorch_model = PyTorchModel(
        model_class_name=model_import.object_name,
        model_kwargs=model_kwargs
    )

    i6_models_import = Import("i6_models")

    serializer = Collection(
        serializer_objects=[
            model_import,
            train_step_import,
            i6_models_import,
            *additional_serializer_objects,
            pytorch_model,
        ],
        make_local_package_copy=True,
        packages={module_import_path, train_step_import_path}.union(additional_packages),
    )

    return serializer
