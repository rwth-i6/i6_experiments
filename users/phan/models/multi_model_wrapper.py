import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, Optional, Any, List

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection, Call, PyTorchModel,
    build_config_constructor_serializers,
)
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.users.berger.systems.dataclasses import ConfigVariant
from i6_experiments.common.setups.serialization import (
    Import, CodeFromFunction, PartialImport,
    SerializerObject
)
from i6_core.returnn.config import CodeWrapper

@dataclass
class MultiModelWrapperConfig(ModelConfiguration):
    module_class: Dict[str, Any] # values are subclasses of nn.Module
    module_config: Dict[str, ModelConfiguration]
    module_preload: Optional[Dict[str, str]] # module -> preload file checkpoint of module

class MultiModelWrapper(nn.Module):
    """
    Wrapper module for holding several modules.
    Used in for example transfer learning (teacher, student),
    seq train (am, lm), etc.
    """

    def __init__(self, cfg: MultiModelWrapperConfig, **kwargs):
        """
        :param cfg: Config
        """
        super().__init__()
        self.cfg = cfg
        self.module_dict = nn.ModuleDict()
        for module in cfg.module_class:
            self.module_dict[module] = cfg.module_class[module](0, cfg.module_config[module])
        if cfg.module_preload is not None:
            for module, checkpoint in cfg.module_preload.items():
                self.load_returnn_pt_checkpoint(module, checkpoint)


    def load_returnn_pt_checkpoint(self, module, ckpt_path):
        """
        Load model from a returnn torch checkpoint

        :param module: Module name as in the module dict
        :param ckpt_path: Returnn pt checkpoint
        """
        checkpoint = torch.load(ckpt_path)
        self.module_dict[module].load_state_dict(checkpoint["model"])


    def forward(self, args, kwargs, module, inference):
        """
        :param args: Args passed to module's forward
        :param kwargs: Kwargs passed to module's forward
        :param module: The module to call forward
        :param inference: Whether to use inference mode or not.
        For example, teacher forward during training.
        """
        if inference:
            with torch.inference_mode():
                x = self.module_dict[module](*args, **kwargs)
        else:
            x = self.module_dict[module](*args, **kwargs)
        return x


def serialize_config(
    model_config: MultiModelWrapperConfig,
    config_variable_name: str = "cfg",
) -> Collection:
    """
    Serialize the MultiModelWrapperConfig and assign it
    to config_variable_name
    """
    module_class_str = "{"
    for module, imported_class in model_config.module_class.items():
        module_class_str += f"\"{module}\": {imported_class.__name__},"
    module_class_str += "}"

    module_config_str = "{"
    for module in model_config.module_class:
        module_config_str += f"\"{module}\": {module}_config,"
    module_config_str += "}"
    config_call = Call(
        callable_name=MultiModelWrapperConfig.__name__,
        kwargs=[
            ("module_class", CodeWrapper(module_class_str)), # let's hope this work
            ("module_config", CodeWrapper(module_config_str)), # sisyphus will try to call .get on a dict
            ("module_preload", CodeWrapper(model_config.module_preload.__repr__() + "\n")), # str to str dict, repr should be fine
        ],
        return_assign_variables=config_variable_name,
    )
    return config_call

wrapper_config_import_obj = Import(f"{__name__}.{MultiModelWrapperConfig.__name__}")
wrapper_model_import_obj = Import(f"{__name__}.{MultiModelWrapper.__name__}")


def get_base_serializer(
    model_config: MultiModelWrapperConfig,
    module_class_import: Dict[str, str],
    prologue_serializers: List[SerializerObject],
    epilogue_serializers: List[SerializerObject],
    config_variable_name: str = "cfg",
) -> Collection:
    """
    Base Serializer object for MultiModelWrapper
    Based on the config variant (train, prior, recog, etc.),
    additional serializer objects will be appended to make a complete config.
    :param model_config: config used in model init
    :param module_class_import: dict of module name -> where to import them
    e.g. `"student_lm": i6_experiments.users.phan.models.lstm_lm.LSTMLM`
    :param prologue_serializer_objects: extra imports, calls. etc. at the beginning
    :param epilogue_serializer_objects: extra imports, calls. etc. at the end
    :return: a serializer object for the model
    """

    # import model configuration and wrapper class, and train step package
    serializers = [
        wrapper_config_import_obj,
        wrapper_model_import_obj,
    ]
    # This exists because hashes...
    serializers.extend(prologue_serializers)

    # Import the needed classes in model config
    for module in model_config.module_class:
        serializers.append(Import(f"{module_class_import[module]}.{model_config.module_class[module].__name__}"))

    # Imports all needed package for the config
    call_objs = []
    for module, config in model_config.module_config.items():
        call_obj, other_imports = build_config_constructor_serializers(
            config,
            variable_name=f"{module}_config",
        )
        serializers.extend(other_imports)
        call_objs.append(call_obj)
    
    serializers.extend(call_objs)

    # Serialize the MultiModelWrapperConfig
    module_class_str = "{"
    for module, imported_class in model_config.module_class.items():
        module_class_str += f"\"{module}\": {imported_class.__name__},"
    module_class_str += "}"

    module_config_str = "{"
    for module in model_config.module_class:
        module_config_str += f"\"{module}\": {module}_config,"
    module_config_str += "}"
    config_call = Call(
        callable_name=MultiModelWrapperConfig.__name__,
        kwargs=[
            ("module_class", CodeWrapper(module_class_str)), # let's hope this work
            ("module_config", CodeWrapper(module_config_str)), # sisyphus will try to call .get on a dict
            ("module_preload", CodeWrapper(model_config.module_preload.__repr__() + "\n")), # str to str dict, repr should be fine
        ],
        return_assign_variables=config_variable_name,
    )
    serializers.append(config_call)

    # get model, model kwargs, ....
    model_kwargs = {"cfg": CodeWrapper("cfg")}
    serializers.append(PyTorchModel(
        model_class_name=MultiModelWrapper.__name__,
        model_kwargs=model_kwargs,
    ))
    serializers.extend(epilogue_serializers)
    return Collection(
        serializer_objects=serializers,
    )


def get_serializer(
    model_config: MultiModelWrapper,
    module_class_import: Dict[str, Any],
    variant: ConfigVariant,
    prologue_serializers_kwargs: dict,
    epilogue_serializers: List[SerializerObject] = [],
) -> Collection:
    """
    Get serializer of the model based on config variant.
    """
    if variant == ConfigVariant.TRAIN:
        prologue_serializers = get_train_extra_serializers(**prologue_serializers_kwargs)
    elif variant == ConfigVariant.PRIOR:
        prologue_serializers = get_prior_extra_serializers(**prologue_serializers_kwargs)
    elif variant == ConfigVariant.RECOG:
        prologue_serializers = get_recog_extra_serializers(**prologue_serializers_kwargs)
    else:
        raise NotImplementedError("variant must be TRAIN, PRIOR, or RECOG")
    return get_base_serializer(
        model_config=model_config,
        module_class_import=module_class_import,
        prologue_serializers=prologue_serializers,
        epilogue_serializers=epilogue_serializers,
    )

# All the partial_kwargs should be a dict with keys
# "hashed_arguments" and "unhashed_arguments"
def get_train_extra_serializers(
    train_step_package: str,
    partial_train_step: bool = False,
    partial_kwargs: dict = {},
) -> List[SerializerObject]:
    """
    For this config variant, only the train step function matters.
    Additional train params can be added via partial import.
    """
    if partial_train_step:
        train_step_import = PartialImport(
            code_object_path=f"{train_step_package}.train_step",
            unhashed_package_root=train_step_package,
            import_as="train_step",
            **partial_kwargs,
        )
    else:
        train_step_import = Import(f"{train_step_package}.train_step")
    serializers = [train_step_import]
    return serializers


def get_prior_extra_serializers(
    forward_step_package: str,
    prior_package: str,
    partial_forward_step: bool = False,
    partial_kwargs: dict = {},
) -> List[SerializerObject]:
    """
    For prior, import the forward step and ComputePriorCallback.
    """
    if partial_forward_step:
        forward_step_import = PartialImport(
            code_object_path=f"{forward_step_package}.forward_step",
            unhashed_package_root=forward_step_package,
            import_as="forward_step",
            **partial_kwargs,
        )
    else:
        forward_step_import = Import(f"{forward_step_package}.forward_step")
    serializers = [
        forward_step_import,
        Import(f"{prior_package}.ComputePriorCallback", import_as="forward_callback"),
    ]
    return serializers


def get_recog_extra_serializers(
    export_package: str,
    partial_export: bool = False,
    partial_kwargs: dict = {},
) -> List[SerializerObject]:
    """
    For recog, the export to ONNX is important. Like train, parameters can be
    passed via partial_export_kwargs.
    """
    if partial_export:
        export_func_import = PartialImport(
            code_object_path=f"{export_package}.export",
            unhashed_package_root=export_package,
            import_as="export",
            **partial_kwargs,
        )
    else:
        export_func_import = Import(f"{export_package}.export")
    serializers = [export_func_import]
    return serializers
