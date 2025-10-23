"""
This allows to easily put i6_models into your setup,
without the need to import i6_models in the Sisyphus manager
(which would import torch and other heavy dependencies).

This is maybe not strictly RETURNN-specific, but this subpackage seemed most appropriate.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Any, Type, Dict
import types
import os
import sys
import functools
import importlib
import dataclasses

if TYPE_CHECKING:
    from i6_models.config import ModelConfiguration, ModuleFactoryV1


def wrap_i6_models_module_as_delayed_factory(
    cls_name: str, config_cls_name: Optional[str] = None, **kwargs
) -> _WrappedI6ModelsModuleDelayedFactory:
    """
    Wrap an i6_models class together with its config as a delayed factory,
    so that the actual object is only created when first accessed.
    """
    assert cls_name.startswith("i6_models.")
    return _WrappedI6ModelsModuleDelayedFactory(cls_name=cls_name, config_cls_name=config_cls_name, config=kwargs)


class _WrappedI6ModelsModuleDelayedFactory:
    """
    This is a Dim where the dim is only computed when it is first accessed,
    e.g. because it depends on a vocab that we don't want to load in the Sis manager.

    This is for pickle / serialization_v2.

    Note: We don't use Sisyphus DelayedBase here,
    because this would be run in a separate create_files Sisyphus job task (see ReturnnTrainingJob)
    where we don't want to load the vocab yet.
    CodeWrapper also does not really work properly in many cases.
    """

    def __init__(
        self,
        *,
        i6_models_path: Optional[str] = None,
        cls_name: str,
        config_cls_name: Optional[str] = None,
        config: Dict[str, Any],
    ):
        i6_models = _import_i6_models(i6_models_path=i6_models_path)

        # Update the path, even if it was given.
        i6_models_path = os.path.dirname(os.path.dirname(os.path.realpath(i6_models.__file__)))

        self.i6_models_path = i6_models_path
        self.cls_name = cls_name
        self.config_cls_name = config_cls_name
        self.config = config

    def __getstate__(self):
        return {
            "i6_models_path": self.i6_models_path,
            "cls_name": self.cls_name,
            "config_cls_name": self.config_cls_name,
            "config": self.config,
        }

    def _sis_hash(self) -> bytes:
        from sisyphus.hash import sis_hash_helper  # noqa

        return b"(_WrappedI6ModelsModuleDelayedFactory, " + sis_hash_helper(self.__getstate__()) + b")"

    def __reduce__(self):
        from i6_experiments.users.zeyer import serialization_v2

        if serialization_v2.in_serialize_config():
            # This is usually run in a separate create_files Sisyphus job task (see ReturnnTrainingJob).
            # But the actual reduce code in that case is run when the config is loaded.
            return functools.partial(_build_i6_models_module_factory, **self.__getstate__()), ()

        # Generic fallback: Serialize as-is (as DelayedReduceDim).
        # E.g. when we pickle the job state.
        # Otherwise, we would create the Dim already during unpickling of the job state,
        # which we don't want.
        # We basically only want the Dim to be created when we actually use it in RETURNN,
        # i.e. the reduce logic when we serialize for the RETURNN config.
        return functools.partial(_WrappedI6ModelsModuleDelayedFactory, **self.__getstate__()), ()


def _build_i6_models_module_factory(
    *,
    i6_models_path: Optional[str] = None,
    cls_name: str,
    config_cls_name: Optional[str] = None,
    config: Dict[str, Any],
) -> ModuleFactoryV1:
    _import_i6_models(i6_models_path=i6_models_path)
    config_obj = _build_i6_models_config_object(config, base_cls_name=cls_name, config_cls_name=config_cls_name)
    cls = _get_cls(cls_name)

    from i6_models.config import ModuleFactoryV1

    # noinspection PyTypeChecker
    return ModuleFactoryV1(cls, config_obj)


def _import_i6_models(*, i6_models_path: Optional[str] = None) -> types.ModuleType:
    try:
        # Try to import first. Maybe the environment is different,
        # and there is a different i6_models available.
        # If so, prefer the new one.
        import i6_models
    except ImportError:
        if i6_models_path is None:
            raise
        sys.path.append(i6_models_path)

        import i6_models

    return i6_models


def _build_i6_models_config_object(
    build_dict: Dict[str, Any],
    *,
    config_cls: Optional[Type] = None,
    config_cls_name: Optional[str] = None,
    base_cls_name: Optional[str] = None,
) -> ModelConfiguration:
    build_dict = build_dict.copy()
    if config_cls is not None:
        build_dict.pop("class", None)
    elif "class" in build_dict:
        if config_cls_name:
            assert config_cls_name == build_dict["class"]
        config_cls = _get_cls(build_dict.pop("class"))
    elif config_cls_name:
        config_cls = _get_cls(config_cls_name)
    elif base_cls_name:
        config_cls = _get_cls(base_cls_name + "Config")
    else:
        raise ValueError("Cannot determine config class")

    if dataclasses.is_dataclass(config_cls):
        # Maybe recursively build nested config objects.
        # noinspection PyDataclass
        for field in dataclasses.fields(config_cls):
            if isinstance(field.type, str):
                # Later Python versions use forward references, and then field.type is a str object.
                cls_mod = sys.modules[config_cls.__module__]
                t = _attr_chain(cls_mod, field.type)
            else:
                t = field.type
            v = build_dict.get(field.name)
            if isinstance(v, dict) and not isinstance(v, t):
                build_dict[field.name] = _build_i6_models_config_object(v, config_cls=t)

    return config_cls(**build_dict)


def _get_cls(cls_name: str) -> Type:
    if "." not in cls_name:
        raise ValueError(f"Expected '.' in class name: {cls_name}")
    mod_name, cls_name = cls_name.rsplit(".", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)


def _attr_chain(obj: Any, attr_chain: str) -> Any:
    for attr in attr_chain.split("."):
        obj = getattr(obj, attr)
    return obj
