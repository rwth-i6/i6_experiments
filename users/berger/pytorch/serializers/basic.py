from typing import Optional
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection,
    PyTorchModel,
)
from i6_experiments.common.setups.serialization import Import
from .model_configuration import get_config_constructor
from i6_models.config import ModelConfiguration


def get_basic_pt_network_serializer(
    module_import_path: str,
    train_step_import_path: str,
    model_config: ModelConfiguration,
    model_kwargs: Optional[dict] = None,
    additional_serializer_objects: Optional[list] = None,
    additional_packages: Optional[set] = None,
) -> Collection:
    if model_kwargs is None:
        model_kwargs = {}
    if additional_serializer_objects is None:
        additional_serializer_objects = []
    if additional_packages is None:
        additional_packages = set()
    model_import = Import(module_import_path)
    train_step_import = Import(train_step_import_path)

    constructor_call, imports = get_config_constructor(model_config, "cfg")
    model_kwargs["cfg"] = CodeWrapper("cfg")

    pytorch_model = PyTorchModel(model_class_name=model_import.object_name, model_kwargs=model_kwargs)

    serializer = Collection(
        serializer_objects=[
            model_import,
            train_step_import,
            *imports,
            *additional_serializer_objects,
            constructor_call,
            pytorch_model,
        ],
        packages=additional_packages,
    )

    return serializer
