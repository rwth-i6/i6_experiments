"""
Vendored from `i6_experiments.common.setups.returnn_pytorch.serialization` on branch
`berger_template_setups`. Only `build_config_constructor_serializers_v2` is needed here;
`Call`, `Import` and everything else are imported from the shared package as usual.

Compared to `build_config_constructor_serializers` in the shared package, this version can
also serialize enum members and values of type list, tuple or dict, and it fixes import
deduplication.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from inspect import isfunction
from typing import Any, List, Optional, Tuple, Union

import torch
from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase, DelayedFormat

from i6_experiments.common.setups.serialization import Call, Import

__all__ = ["build_config_constructor_serializers_v2"]


def build_config_constructor_serializers_v2(
    cfg: Any,
    variable_name: Optional[str] = None,
    unhashed_package_root: Optional[str] = None,
) -> Tuple[Call, List[Import]]:
    """
    Creates a Call object that will re-construct the given ModelConfiguration when serialized and
    optionally assigns the resulting config object to a variable. Automatically generates a list of all
    necessary imports in order to perform the constructor call.

    Compared to the previous version, this function can also serialize enum members and values of type
    list, tuple or dict. It also fixes import deduplication.

    :param cfg: ModelConfiguration or dataclass object that will be re-constructed by the Call serializer
    :param variable_name: Name of the variable which the constructed ModelConfiguration
                          will be assigned to. If None, the result will not be assigned
                          to a variable.
    :param unhashed_package_root: Will be passed to all generated Import objects.
    :return: Call object and list of necessary imports.
    """
    from i6_models.config import ModuleFactoryV1

    # Helper function which can call itself recursively for nested types
    def serialize_value(value: Any) -> Tuple[Union[str, DelayedBase], List[Import]]:
        # Switch over serialization logic for different subtypes

        if isinstance(value, ModuleFactoryV1):
            # Example:
            # ConformerEncoderConfig(
            #     frontend=ModuleFactoryV1(module_class=VGGFrontend, cfg=VGGFrontendConfig(...)))
            # -> Import classes ModuleFactoryV1, VGGFrontend and VGGFrontendConfig
            # -> Sub-Constructor-Call for VGGFrontendConfig
            subcall, subimports = build_config_constructor_serializers_v2(
                value.cfg, unhashed_package_root=unhashed_package_root
            )
            subimports.append(
                Import(
                    code_object_path=f"{value.module_class.__module__}.{value.module_class.__name__}",
                    unhashed_package_root=(
                        unhashed_package_root
                        if unhashed_package_root is not None
                        and value.module_class.__module__.startswith(unhashed_package_root)
                        else None
                    ),
                )
            )
            subimports.append(
                Import(
                    code_object_path=f"{ModuleFactoryV1.__module__}.{ModuleFactoryV1.__name__}",
                    unhashed_package_root=(
                        unhashed_package_root
                        if unhashed_package_root is not None
                        and ModuleFactoryV1.__module__.startswith(unhashed_package_root)
                        else None
                    ),
                )
            )
            return (
                Call(
                    callable_name=ModuleFactoryV1.__name__,
                    kwargs=[
                        ("module_class", value.module_class.__name__),
                        ("cfg", subcall),
                    ],
                ),
                subimports,
            )
        elif is_dataclass(value):
            # Example:
            # ConformerBlockConfig(mhsa_config=ConformerMHSAConfig(...))
            # -> Sub-Constructor-Call and imports for ConformerMHSAConfig
            return build_config_constructor_serializers_v2(value, unhashed_package_root=unhashed_package_root)
        elif isinstance(value, torch.nn.Module):
            # Example:
            # ConformerConvolutionConfig(norm=BatchNorm1d(...))
            # -> Import class BatchNorm1d
            # -> Sub-serialization of BatchNorm1d object.
            #       The __str__ function of torch.nn.Module already does this in the way we want.
            return str(value), [
                Import(
                    code_object_path=f"{value.__module__}.{type(value).__name__}",
                    unhashed_package_root=(
                        unhashed_package_root
                        if unhashed_package_root is not None and value.__module__.startswith(unhashed_package_root)
                        else None
                    ),
                )
            ]
        elif isfunction(value):
            # Example:
            # ConformerConvolutionConfig(activation=torch.nn.functional.silu)
            # -> Import function silu
            # Builtins (e.g. 'sum') do not need to be imported
            if value.__module__ != "builtins":
                subimports = [
                    Import(
                        code_object_path=f"{value.__module__}.{value.__name__}",
                        unhashed_package_root=(
                            unhashed_package_root
                            if unhashed_package_root is not None and value.__module__.startswith(unhashed_package_root)
                            else None
                        ),
                    )
                ]
            else:
                subimports = []
            return value.__name__, subimports
        elif isinstance(value, Enum):
            # Example:
            # FrontendLayerType.Conv2d
            # -> Import enum class FrontendLayerType
            subimports = [
                Import(
                    code_object_path=f"{value.__class__.__module__}.{value.__class__.__name__}",
                    unhashed_package_root=(
                        unhashed_package_root
                        if unhashed_package_root is not None
                        and value.__class__.__module__.startswith(unhashed_package_root)
                        else None
                    ),
                )
            ]
            return f"{value.__class__.__name__}.{value.name}", subimports
        elif isinstance(value, list):
            # -> Serialize list values individually, collect subimports
            list_items = []
            list_imports = []
            for item in value:
                item_serialized, item_imports = serialize_value(item)
                list_items.append(item_serialized)
                list_imports += item_imports
            return DelayedFormat(f"[{', '.join(['{}'] * len(list_items))}]", *list_items), list_imports
        elif isinstance(value, tuple):
            # -> Serialize tuple values individually, collect subimports
            tuple_items = []
            tuple_imports = []
            for item in value:
                item_serialized, item_imports = serialize_value(item)
                tuple_items.append(item_serialized)
                tuple_imports += item_imports
            return DelayedFormat(f"({', '.join(['{}'] * len(tuple_items))})", *tuple_items), tuple_imports
        elif isinstance(value, dict):
            # -> Serialize dict values individually, collect subimports
            dict_items = []  # Will alternatingly contain key and value of all dict items
            dict_imports = []
            for key, val in value.items():
                val_serialized, item_imports = serialize_value(val)
                dict_items += [repr(key), val_serialized]
                dict_imports += item_imports
            return (
                DelayedFormat("{{" + ", ".join(["{}: {}"] * (len(dict_items) // 2)) + "}}", *dict_items),
                dict_imports,
            )
        elif isinstance(value, tk.Path):
            return DelayedFormat('tk.Path("{}")', value), [Import("sisyphus.tk")]
        elif isinstance(value, DelayedBase):
            # sisyphus variables are just given as-is and will be instanciated only when calling "get".
            return value, []
        elif isinstance(value, str):
            return f'"{value}"', []
        else:
            # No special case (usually python primitives)
            # -> Just get string representation
            return str(value), []

    # Import the class of `cfg`
    imports = [
        Import(
            code_object_path=f"{type(cfg).__module__}.{type(cfg).__name__}",
            unhashed_package_root=(
                unhashed_package_root
                if unhashed_package_root is not None and type(cfg).__module__.startswith(unhashed_package_root)
                else None
            ),
        )
    ]

    call_kwargs = []

    # Iterate over all dataclass fields and apply helper function to all values
    for key in fields(type(cfg)):
        # Value corresponding to dataclass field name
        value = getattr(cfg, key.name)

        serialized_value, value_imports = serialize_value(value)
        call_kwargs.append((key.name, serialized_value))
        imports += value_imports

    # Deduplicate imports
    seen_hashes = set()
    unique_imports = []
    for imp in imports:
        imp_hash = hash(imp)
        if imp_hash not in seen_hashes:
            seen_hashes.add(imp_hash)
            unique_imports.append(imp)

    unique_imports.sort(key=lambda imp: str(imp))

    return (
        Call(
            callable_name=type(cfg).__name__,
            kwargs=call_kwargs,
            return_assign_variables=variable_name,
        ),
        unique_imports,
    )
