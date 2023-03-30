"""
Serialization helpers for RETURNN, such as ReturnnConfig
"""


from __future__ import annotations

from typing import TypeVar, Any, Optional, Set, Dict
from copy import deepcopy
from types import FunctionType
from i6_core.returnn.config import ReturnnConfig

# The code here does not need the user to use returnn_common.
# However, we internally make use of some helper code from returnn_common.
from returnn_common.nn.naming import ReturnnDimTagsProxy

from .. import serialization as base_serialization


T = TypeVar("T")


def get_serializable_config(config: ReturnnConfig, *, hash_full_python_code: bool = False) -> ReturnnConfig:
    """
    Takes the config, goes through the config (e.g. network dict)
    and replaces some non-serializable objects (e.g. dim tags) with serializable ones.
    (Currently, it is all about dim tags.)

    :param config: the existing config
    :param hash_full_python_code: if True, the full python code is used for hashing via :class:`CodeFromFunction`,
        otherwise it uses only the module name and function qualname for the hash of functions
    :return: either config itself if no change needed, or otherwise new adapted config
    """
    config = deepcopy(config)

    # Collect taken Python variable names (or function names).
    # See if there are already some existing functions in the prolog.
    reserved_names = set()
    dim_tag_proxy = ReturnnDimTagsProxy(reserved_names=reserved_names)
    proxy = _ProxyHandler(reserved_names=reserved_names)
    if isinstance(config.python_prolog, (list, tuple)):
        for obj in config.python_prolog:
            if isinstance(obj, base_serialization.CodeFromFunction):
                proxy.register_obj(obj.name, obj.func, serializer_obj=obj)

    # Collect all dim tags.
    config.config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.config)
    config.post_config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.post_config)
    config.staged_network_dict = dim_tag_proxy.collect_dim_tags_and_transform_config(config.staged_network_dict)

    # Collect some other objects (currently only functions).
    proxy.obj_refs_by_name.clear()  # reset such that we only have the new ones in here
    config.config = proxy.collect_objs_and_transform_config(config.config)
    config.post_config = proxy.collect_objs_and_transform_config(config.post_config)
    config.staged_network_dict = proxy.collect_objs_and_transform_config(config.staged_network_dict)

    if not dim_tag_proxy.dim_refs_by_name and not proxy.obj_refs_by_name:
        # No dim tags or other special objects found, just return as-is.
        return config

    if proxy.obj_refs_by_name:
        assert not config.hash_full_python_code, (
            f"For extended serialization ({proxy}), you must not use hash_full_python_code=True in the ReturnnConfig."
            " We use the python_prolog with DelayedObjects to serialize the code."
        )

    # Prepare object to use config.update(),
    # because config.update() does reasonable logic for python_epilog code merging,
    # including handling of python_epilog_hash.
    python_prolog_ext = []
    for code in [
        # Probably we should use base_serialization.NonhashedCode for this here...
        _ImportPyCodeStr,
        # Also this should probably be split for each individual dim tag definition,
        # and those wrapped in some own code wrappers,
        # such that we have control over the hash and can make sure it will stay stable,
        # even with code changes.
        dim_tag_proxy.py_code_str(),
    ]:
        if not code:
            continue
        if config.python_prolog and code in config.python_prolog:
            continue
        python_prolog_ext.append(code)
    if proxy.obj_refs_by_name:
        for obj in proxy.obj_refs_by_name.values():
            python_prolog_ext.append(obj.get_serializer_obj(hash_full_python_code=hash_full_python_code))
    config_update = ReturnnConfig(
        {},
        python_prolog=python_prolog_ext,
        hash_full_python_code=config.hash_full_python_code,
    )
    config.update(config_update)

    return config


class _ProxyHandler:
    """
    When serialized via __repr__, this represents a dict unique_name -> obj.
    All usages in the network and extern_data will also get proxies when serialized point to this dict.
    Adapted from ReturnnDimTagsProxy but for other object types such as function.
    """

    _HandledTypes = (FunctionType,)

    class Proxy:
        """
        This will be a reference to the global functions.
        """

        def __init__(
            self,
            *,
            name: str,
            obj: Any,
            # Currently the only type we handle is CodeFromFunction.
            # If we need sth else, we probably should create some base class.
            serializer_obj: Optional[base_serialization.CodeFromFunction] = None,
        ):
            self.name = name  # Python identifier
            self.obj = obj
            self.serializer_obj = serializer_obj

        def __repr__(self):
            return self.py_id_name()

        def py_id_name(self) -> str:
            """
            :return: valid Python identifier
            """
            assert self.name
            return self.name

        def get_serializer_obj(self, *, hash_full_python_code: bool) -> base_serialization.CodeFromFunction:
            """
            :param hash_full_python_code:
            :return: SerializerObject
            """
            if self.serializer_obj is None:
                assert isinstance(self.obj, FunctionType)  # only case implemented here
                self.serializer_obj = base_serialization.CodeFromFunction(
                    name=self.name, func=self.obj, hash_full_python_code=hash_full_python_code
                )
            return self.serializer_obj

    def __init__(self, *, reserved_names: Set[str]):
        self.obj_refs_by_name = {}  # type: Dict[str, _ProxyHandler.Proxy]
        self.obj_refs_by_id = {}  # type: Dict[int, _ProxyHandler.Proxy]
        self.reserved_names = reserved_names

    def _sis_hash(self):
        raise Exception("unexpected")

    def collect_objs_and_transform_config(self, config: T) -> T:
        """
        Go through the config and collect all relevant objects, replace them by proxies.

        :return: new config
        """
        import re

        def _unique_name(obj) -> str:
            assert id(obj) not in self.obj_refs_by_id
            name_ = obj.__qualname__ or obj.__class__.__qualname__
            name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
            if not name_ or name_[:1].isdigit():
                name_ = "_" + name_
            if name_ not in self.reserved_names:
                return name_
            i = 0
            while True:
                name__ = f"{name_}_{i}"
                if name__ not in self.reserved_names:
                    return name__
                i += 1

        # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
        def _map(path, value):
            if isinstance(value, self._HandledTypes):
                if id(value) in self.obj_refs_by_id:
                    return self.obj_refs_by_id[id(value)]
                name = _unique_name(value)
                return self.register_obj(name=name, obj=value)
            if isinstance(value, dict):
                return {
                    _map(path + (key, "key"), key): _map(path + (key, "value"), value_) for key, value_ in value.items()
                }
            if isinstance(value, list):
                return [_map(path + (i,), value_) for i, value_ in enumerate(value)]
            if isinstance(value, tuple) and type(value) is tuple:
                return tuple(_map(path + (i,), value_) for i, value_ in enumerate(value))
            if isinstance(value, tuple) and type(value) is not tuple:
                # noinspection PyProtectedMember,PyUnresolvedReferences,PyArgumentList
                return type(value)(*(_map(path + (key,), getattr(value, key)) for key in value._fields))
            if isinstance(value, set):
                values = [_map(path + (value,), value_) for value_ in value]
                return set(values)
            return value

        config = _map((), config)
        return config

    def register_obj(
        self, name: str, obj: Any, *, serializer_obj: Optional[base_serialization.CodeFromFunction] = None
    ) -> _ProxyHandler.Proxy:
        """
        :param name:
        :param obj:
        :param serializer_obj:
        """
        assert name not in self.reserved_names
        assert name not in self.obj_refs_by_name
        assert id(obj) not in self.obj_refs_by_id
        assert isinstance(obj, self._HandledTypes)
        ref = _ProxyHandler.Proxy(name=name, obj=obj, serializer_obj=serializer_obj)
        self.obj_refs_by_name[name] = ref
        self.obj_refs_by_id[id(obj)] = ref
        self.reserved_names.add(name)
        return ref


# This is like returnn-common ReturnnConfigSerializer.ImportPyCodeStr.
# We copy it here because in the current implementation of get_serializable_config,
# any change in here will also change the config hash,
# but in returnn-common, we do not guarantee that this code will not change.
_ImportPyCodeStr = (
    "from returnn.tf.util.data import (\n"
    "  Dim, batch_dim, single_step_dim,"
    " SpatialDim, FeatureDim, ImplicitDynSizeDim, ImplicitSparseDim)\n\n"
)
