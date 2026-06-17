from collections import OrderedDict
from dataclasses import dataclass, field, fields
from inspect import isfunction
from typing import List, Optional, Tuple, Union

import torch
from i6_experiments.common.setups.serialization import Import, SerializerObject
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from sisyphus.delayed_ops import DelayedBase
from sisyphus.hash import sis_hash_helper
from sisyphus.tools import try_get


@dataclass
class FunctionCall(SerializerObject):
    """
    SerializerObject that serializes the call of a function with a given name with given arguments.
    The return value is optionally assigned to a variable of a given name.
    Example:
    FunctionCall(func_name="range", args=[1, 10], kwargs={"step": 2}, variable_name="number_range")
    ->
    number_range = range(1, 10, step=2)
    """

    func_name: str
    args: List[Union[str, DelayedBase]] = field(default_factory=list)
    kwargs: "OrderedDict[str, Union[str, DelayedBase]]" = field(default_factory=OrderedDict)
    variable_name: Optional[str] = None

    def get(self) -> str:
        result = ""

        # Variable assignment
        if self.variable_name is not None:
            result += f"{self.variable_name} = "

        # Function call
        result += f"{self.func_name}("

        args_str = ", ".join([str(try_get(val)) for val in self.args])
        kwargs_str = ", ".join([f"{key}={try_get(val)}" for key, val in self.kwargs.items()])

        # Account for calls where args and/or kwargs is empty
        result += ", ".join(filter(lambda s: s != "", [args_str, kwargs_str]))

        result += ")"

        return result

    def _sis_hash(self):
        h = {
            "func_name": self.func_name,
            "args": self.args,
            "kwargs": self.kwargs,
            "variable_name": self.variable_name,
        }
        return sis_hash_helper(h)


def get_config_constructor(
    cfg: ModelConfiguration, variable_name: Optional[str] = None
) -> Tuple[FunctionCall, List[Import]]:
    """
    Creates a SerializerObject that constructs a ModelConfiguration instance
    and optionally assigns it to a variable.

    :param cfg: ModelConfiguration object that will be re-created by the serializer
    :param variable_name: Name of the variable which the constructed ModelConfiguration
                          will be assigned to. If None, the result will not be assigned
                          to a variable.
    :return: Tuple of ConstructorCall serializer object and list of all imports
             that are necessary to construct the ModelConfiguration.
    """

    # Import the class of <cfg>
    imports = [Import(f"{type(cfg).__module__}.{type(cfg).__name__}")]

    attrs = OrderedDict()

    # Iterate over all dataclass fields
    for key in fields(type(cfg)):
        # Value corresponding to dataclass field name
        attr = getattr(cfg, key.name)

        # Switch over serialization logic for different subtypes
        if isinstance(attr, ModelConfiguration):
            # Example:
            # ConformerBlockConfig(mhsa_config=ConformerMHSAConfig(...))
            # -> Sub-Constructor-Call and imports for ConformerMHSAConfig
            subcall, subimports = get_config_constructor(attr)
            imports += subimports
            attrs[key.name] = subcall
        elif isinstance(attr, ModuleFactoryV1):
            # Example:
            # ConformerEncoderConfig(
            #     frontend=ModuleFactoryV1(module_class=VGGFrontend, cfg=VGGFrontendConfig(...)))
            # -> Import classes ModuleFactoryV1, VGGFrontend and VGGFrontendConfig
            # -> Sub-Constructor-Call for VGGFrontendConfig
            subcall, subimports = get_config_constructor(attr.cfg)
            imports += subimports
            imports.append(Import(f"{attr.module_class.__module__}.{attr.module_class.__name__}"))
            imports.append(Import(f"{ModuleFactoryV1.__module__}.ModuleFactoryV1"))
            attrs[key.name] = FunctionCall(
                func_name="ModuleFactoryV1",
                kwargs=OrderedDict([("module_class", attr.module_class.__name__), ("cfg", subcall)]),
            )
        elif isinstance(attr, torch.nn.Module):
            # Example:
            # ConformerConvolutionConfig(norm=BatchNorm1d(...))
            # -> Import class BatchNorm1d
            # -> Sub-serialization of BatchNorm1d object.
            #       The __str__ function of torch.nn.Module already does this in the way we want.
            imports.append(Import(f"{attr.__module__}.{type(attr).__name__}"))
            attrs[key.name] = str(attr)
        elif isfunction(attr):
            # Example:
            # ConformerConvolutionConfig(activation=torch.nn.functional.silu)
            # -> Import function silu
            # Builtins (e.g. 'sum') do not need to be imported
            if attr.__module__ != "builtins":
                imports.append(Import(f"{attr.__module__}.{attr.__name__}"))
            attrs[key.name] = attr.__name__
        elif isinstance(attr, DelayedBase):
            # sisyphus variables are just given as-is and will be instanciated only when calling "get".
            attrs[key.name] = attr
        else:
            # No special case (usually python primitives)
            # -> Just get string representation
            attrs[key.name] = str(attr)

    return FunctionCall(func_name=type(cfg).__name__, kwargs=attrs, variable_name=variable_name), imports
