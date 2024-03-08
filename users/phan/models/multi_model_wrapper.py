import torch
from torch import nn
from dataclasses import dataclass
from typing import Dict, Optional, Any

from i6_models.config import ModelConfiguration
from i6_experiments.common.setups.returnn_pytorch.serialization import (
    Collection, Call, PyTorchModel,
    build_config_constructor_serializers,
)
from i6_experiments.users.berger.pytorch.serializers.basic import (
    get_basic_pt_network_serializer,
)
from i6_experiments.common.setups.serialization import (
    Import, CodeFromFunction, PartialImport
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

def get_train_serializer(
    model_config: MultiModelWrapperConfig,
    module_class_import: Dict[str, str],
    train_step_package: str,
    partial_train_step_func: bool = False,
    partial_import_kwargs: dict = {},
) -> Collection:
    """
    Serializer object for MultiModelWrapper
    :param model_config: config used in model init
    :param module_class_import: dict of module name -> where to import them
    e.g. `"student_lm": i6_experiments.users.phan.models.lstm_lm.LSTMLM`
    :param train_step_package: where to import the train step
    :param partial_train_step_func: If True, use a partial tain step function.
    This is to add parameters to the training without attaching them to the model.
    :param partial_import_kwargs: Args passed to PartialImport, currently a dict
    ith keys "hashed_arguments" and "unhashed_arguments"
    :return: a serializer object for the model
    """

    # import model configuration and wrapper class, and train step package
    serializers = [
        Import(f"{__name__}.{MultiModelWrapperConfig.__name__}"),
        Import(f"{__name__}.{MultiModelWrapper.__name__}"),
    ]
    if not partial_train_step_func:
        train_step_import = PartialImport(
            code_object_path=f"{train_step_package}.train_step",
            unhashed_package_root=train_step_package,
            import_as="train_step",
            **partial_import_kwargs,
        )
    else:
        train_step_import = Import(f"{train_step_package}.train_step")
    serializers.append(train_step_import)

    # Import the needed classes in model config
    for module in model_config.module_class:
        serializers.append(Import(f"{module_class_import[module]}.{model_config.module_class[module].__name__}"))

    # Use CodeWrapper to insert literal code

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
        return_assign_variables="cfg",
    )
    serializers.append(config_call)

    # get model, model kwargs, ....
    model_kwargs = {"cfg": CodeWrapper("cfg")}
    serializers.append(PyTorchModel(
        model_class_name=MultiModelWrapper.__name__,
        model_kwargs=model_kwargs,
    ))
    return Collection(
        serializer_objects=serializers,
    )
