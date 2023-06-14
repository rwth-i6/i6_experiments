from dataclasses import dataclass, fields
from inspect import isfunction
from typing import Any, Dict, List, Optional, Tuple

import torch
from i6_experiments.common.setups.serialization import Import, SerializerObject
from i6_models.config import ModelConfiguration, SubassemblyWithOptions
from sisyphus.hash import sis_hash_helper
from sisyphus.tools import try_get


@dataclass
class ConstructorCall(SerializerObject):
    class_name: str
    args: Dict[str, Any]
    variable_name: Optional[str] = None

    def get(self) -> str:
        result = ""
        if self.variable_name is not None:
            result += f"{self.variable_name} = "
        result += f"{self.class_name}("
        arg_strings = []
        for key, val in self.args.items():
            arg_strings.append(f"{key}={try_get(val)}")
        result += ", ".join(arg_strings)
        result += ")"
        return result

    def _sis_hash(self):
        h = {
            "class_name": self.class_name,
            "args": self.args,
        }
        return sis_hash_helper(h)


def get_config_constructor(
    cfg: ModelConfiguration, variable_name: Optional[str] = None
) -> Tuple[ConstructorCall, List[Import]]:
    """
    Creates a SerializerObject that constructs a ModelConfiguration instance
    and assigns it to a variable.

    :param cfg: ModelConfiguration object that will be re-created by the serializer
    :param variable_name: Name of the variable which the constructed ModelConfiguration
                          will be assigned to. If none, the result will not be assigned
                          to a variable.
    :return: Tuple of ConstructorCall serializer object and list of all imports
             that are necessary to construct the ModelConfiguration.
    """

    # Import the class of <cfg>
    imports = [Import(f"{type(cfg).__module__}.{type(cfg).__name__}")]

    attrs = {}

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
        elif isinstance(attr, SubassemblyWithOptions):
            # Example:
            # ConformerEncoderConfig(
            #     frontend=SubassemblyWithOptions(module_class=VGGFrontend, cfg=VGGFrontendConfig(...)))
            # -> Import class VGGFrontend
            # -> Sub-Constructor-Call and imports for VGGFrontendConfig
            subcall, subimports = get_config_constructor(attr.cfg)
            imports += subimports
            imports.append(Import(f"{attr.module_class.__module__}.{attr.module_class.__name__}"))
            attrs[key.name] = ConstructorCall(
                class_name="SubassemblyWithOptions",
                args={"module_class": attr.module_class.__name__, "cfg": subcall},
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
        else:
            # No special case (usually python primitives)
            # -> Just get string representation
            attrs[key.name] = str(attr)

    return ConstructorCall(class_name=type(cfg).__name__, args=attrs, variable_name=variable_name), imports
