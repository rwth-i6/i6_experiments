from typing import List, Optional

from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import Collection, PyTorchModel
from i6_experiments.common.setups.serialization import Import, SerializerObject
from i6_experiments.users.berger.pytorch.serializers.model import ExportPyTorchModel
from i6_models.config import ModelConfiguration

from .model_configuration import get_config_constructor


def get_basic_pt_network_train_serializer(
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
    serializer_objects: List[SerializerObject] = [model_import]
    serializer_objects.append(Import(train_step_import_path))

    constructor_call, imports = get_config_constructor(model_config, "cfg")
    serializer_objects.extend(imports)
    serializer_objects.append(constructor_call)

    model_kwargs["cfg"] = CodeWrapper("cfg")
    serializer_objects.append(PyTorchModel(model_class_name=model_import.object_name, model_kwargs=model_kwargs))

    serializer_objects.extend(additional_serializer_objects)

    serializer = Collection(
        serializer_objects=serializer_objects,
        packages=additional_packages,
    )

    return serializer


def get_basic_pt_network_recog_serializer(
    module_import_path: str,
    model_config: ModelConfiguration,
    model_kwargs: Optional[dict] = None,
    export_import_path: Optional[str] = None,
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
    serializer_objects: List[SerializerObject] = [model_import]
    serializer_objects.append(Import(export_import_path))

    constructor_call, imports = get_config_constructor(model_config, "cfg")
    serializer_objects.extend(imports)
    serializer_objects.append(constructor_call)

    model_kwargs["cfg"] = CodeWrapper("cfg")
    serializer_objects.append(
        PyTorchModel(
            model_class_name=model_import.object_name,
            model_kwargs=model_kwargs,
        )
    )

    serializer_objects.extend(additional_serializer_objects)

    serializer = Collection(
        serializer_objects=serializer_objects,
        packages=additional_packages,
    )

    return serializer
