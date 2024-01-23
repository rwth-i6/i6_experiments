from typing import List, Optional
from i6_core.returnn.config import CodeWrapper

from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection,
    PyTorchModel,
    SerializerObject
)
from i6_experiments.common.setups.serialization import Import
from i6_experiments.users.rossenbach.common_setups.returnn.serializer import PartialImport
from .model_configuration import get_config_constructor
from i6_models.config import ModelConfiguration

def get_basic_pt_network_serializer(
    module_import_path: str,
    steps_import_path: str,
    model_config: ModelConfiguration,
    model_kwargs: Optional[dict] = None,
    additional_train_serializer_objects: Optional[list] = None,
    additional_forward_serializer_objects: Optional[list] = None,
    additional_packages: Optional[set] = None,
) -> Collection:

    package = __package__

    if model_kwargs is None:
        model_kwargs = {}
    if additional_train_serializer_objects is None:
        additional_train_serializer_objects = []
    if additional_forward_serializer_objects is None:
        additional_forward_serializer_objects = []
    if additional_packages is None:
        additional_packages = set()
    model_import = Import(f"{module_import_path}.ConformerCTCModel")

    train_step_import = Import(f"{steps_import_path}.train_step")

    # Just a hack to enable PT search serialization
    forward_step_import = Import(f"{steps_import_path}.search_step",
                                 import_as="forward_step")

    init_hook_import = Import(f"{steps_import_path}.search_init_hook",
                              import_as="forward_init_hook")
    finish_hook_import = Import(f"{steps_import_path}.search_finish_hook",
                                import_as="forward_finish_hook")


    constructor_call, imports = get_config_constructor(model_config, "cfg")
    model_kwargs["cfg"] = CodeWrapper("cfg")

    pytorch_model = PyTorchModel(model_class_name=model_import.object_name, model_kwargs=model_kwargs)

    train_serializer = Collection(
        serializer_objects=[
            model_import,
            train_step_import,
            *imports,
            *additional_train_serializer_objects,
            constructor_call,
            pytorch_model,
        ],
        packages=additional_packages,
    )

    forward_serializer = Collection(
        serializer_objects=[
            model_import,
            forward_step_import,
            init_hook_import,
            finish_hook_import,
            *imports,
            *additional_forward_serializer_objects,
            constructor_call,
            pytorch_model
        ],
        packages=additional_packages
    )

    return train_serializer, forward_serializer

########################################################################

def get_basic_pt_network_train_serializer(
    module_import_path: str,
    train_step_import_path: str,
    model_config: ModelConfiguration,
    model_kwargs: Optional[dict] = None,
    additional_serializer_objects: Optional[list] = None,
    additional_packages: Optional[set] = None,
    debug: bool = False,
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
    serializer_objects.append(
        PyTorchModel(
            model_class_name=model_import.object_name,
            model_kwargs=model_kwargs
        )
    )

    serializer_objects.extend(additional_serializer_objects)
    print("additional_packages:", additional_packages)
    serializer = Collection(
        serializer_objects=serializer_objects,
        packages=additional_packages,
        make_local_package_copy=not debug,
    )

    return serializer


def get_basic_pt_network_recog_serializer(
    module_import_path: str,
    model_config: ModelConfiguration,
    import_kwargs: Optional[dict] = None,
    model_kwargs: Optional[dict] = None,
    recog_step_import_path: Optional[str] = None,
    additional_serializer_objects: Optional[list] = None,
    additional_packages: Optional[set] = None,
    debug: bool = False,
) -> Collection:
    if model_kwargs is None:
        model_kwargs = {}
    if additional_serializer_objects is None:
        additional_serializer_objects = []
    if additional_packages is None:
        additional_packages = set()

    PACKAGE = __package__.rsplit('.', 1)[0]

    model_import = Import(module_import_path)
    serializer_objects: List[SerializerObject] = [model_import]
    # serializer_objects.append(Import(recog_step_import_path, import_as="forward_step"))

    # Problematic due to additional arguments for forward_step(), i.e. text_lexicon
    # forward_step_import = Import(f"{recog_step_import_path}.search_step", import_as="forward_step")
    # init_hook_import = Import(f"{recog_step_import_path}.search_init_hook", import_as="forward_init_hook")
    # finish_hook_import = Import(f"{recog_step_import_path}.search_finish_hook", import_as="forward_finish_hook")
    # serializer_objects.extend([forward_step_import, init_hook_import, finish_hook_import])

    init_hook_import = PartialImport(
        code_object_path=f"{recog_step_import_path}.search_init_hook",
        unhashed_package_root=PACKAGE,
        hashed_arguments=import_kwargs,
        unhashed_arguments={},
        import_as="forward_init_hook",
    )

    forward_step_import = Import(f"{recog_step_import_path}.search_step",import_as="forward_step")
    finish_hook_import = Import(f"{recog_step_import_path}.search_finish_hook", import_as="forward_finish_hook")
    serializer_objects.extend([init_hook_import, forward_step_import, finish_hook_import])

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
    print("additional_packages:", additional_packages)
    serializer = Collection(
        serializer_objects=serializer_objects,
        packages=additional_packages,
        make_local_package_copy=not debug,
    )

    return serializer