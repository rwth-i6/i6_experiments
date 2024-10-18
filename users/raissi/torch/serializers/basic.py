# This code is copied from Simon Berger
from typing import List, Optional
from enum import Enum, auto

from i6_core.returnn.config import CodeWrapper
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection,
    PyTorchModel,
    build_config_constructor_serializers,
)
from i6_experiments.common.setups.serialization import Import, SerializerObject
from i6_experiements.users.raissi.torch.args import SerializationAndHashArgs


from i6_models.config import ModelConfiguration


class SerialConfigVariant(Enum):
    TRAIN = auto()
    PRIOR = auto()
    ALIGN = auto()
    RECOG = auto()


def get_basic_pt_network_serializer(
    module_import_path: str,
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

    constructor_call, imports = build_config_constructor_serializers(model_config, "cfg")
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
