"""
Serialization helpers for RETURNN, such as ReturnnConfig
"""


from __future__ import annotations

from typing import TypeVar, Any, Tuple, Dict
from copy import deepcopy
from types import FunctionType
from i6_core.returnn.config import ReturnnConfig

# The code here does not need the user to use returnn_common.
# However, we internally make use of some helper code from returnn_common.
from returnn_common.nn.naming import ReturnnDimTagsProxy, ReturnnConfigSerializer


T = TypeVar("T")


def get_serializable_config(config: ReturnnConfig) -> ReturnnConfig:
    """
    Takes the config, goes through the config (e.g. network dict)
    and replaces some non-serializable objects (e.g. dim tags) with serializable ones.
    (Currently, it is all about dim tags.)
    """
    config = deepcopy(config)
    dim_tag_proxy = ReturnnDimTagsProxy()
    config.config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.config)
    config.post_config = dim_tag_proxy.collect_dim_tags_and_transform_config(config.post_config)
    config.staged_network_dict = dim_tag_proxy.collect_dim_tags_and_transform_config(config.staged_network_dict)
    proxy = _ProxyHandler()
    config.config = proxy.collect_objs_and_transform_config(config.config)
    config.post_config = proxy.collect_objs_and_transform_config(config.post_config)
    config.staged_network_dict = proxy.collect_objs_and_transform_config(config.staged_network_dict)

    if not dim_tag_proxy.dim_refs_by_name and not proxy.obj_refs_by_name:
        # No dim tags or other special objects found, just return as-is.
        return config

    # Prepare object to use config.update(),
    # because config.update() does reasonable logic for python_epilog code merging,
    # including handling of python_epilog_hash.
    python_prolog_ext = []
    dim_tag_def_code = dim_tag_proxy.py_code_str()
    obj_def_code = proxy.py_code_str()
    for code in [
        ReturnnConfigSerializer.ImportPyCodeStr,
        dim_tag_def_code,
        obj_def_code,
    ]:
        if not code:
            continue
        if config.python_prolog and code in config.python_prolog:
            continue
        python_prolog_ext.append(code)
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

        def __init__(self, *, name: str, path: Tuple[Any, ...], obj: Any):
            self.name = name  # Python identifier
            self.path = path
            self.obj = obj

        def __repr__(self):
            return self.py_id_name()

        def py_id_name(self) -> str:
            """
            :return: valid Python identifier
            """
            assert self.name
            return self.name

        def py_code(self):
            """
            :return: Python code
            """
            import inspect

            if isinstance(self.obj, FunctionType):
                # Similar as ReturnnConfig.
                s = inspect.getsource(self.obj)
                if self.obj.__name__ == self.name:
                    return s
                return f"{s}\n{self.py_id_name()} = {self.obj.__name__}"
            return f"{self.py_id_name()} = {self.obj!r}"

    def __init__(self):
        self.obj_refs_by_name = {}  # type: Dict[str, _ProxyHandler.Proxy]
        self.obj_refs_by_id = {}  # type: Dict[int, _ProxyHandler.Proxy]

    def __repr__(self):
        return "\n".join(
            [
                f"<{self.__class__.__name__}:",
                *(f"  {value.py_id_name()} = {value!r}" for key, value in self.obj_refs_by_name.items()),
                ">",
            ]
        )

    def py_code_str(self):
        """
        :return: Python code
        """
        lines = []
        for _, value in self.obj_refs_by_name.items():
            lines.append(f"{value.py_code()}\n")
        return "".join(lines)

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
            if name_ not in self.obj_refs_by_name:
                return name_
            i = 0
            while True:
                name__ = f"{name_}_{i}"
                if name__ not in self.obj_refs_by_name:
                    return name__
                i += 1

        # Cannot use nest because nest does not support sets. Also nest requires them to be sorted.
        def _map(path, value):
            if isinstance(value, self._HandledTypes):
                if id(value) in self.obj_refs_by_id:
                    return self.obj_refs_by_id[id(value)]
                name = _unique_name(value)
                assert name not in self.obj_refs_by_name
                ref = _ProxyHandler.Proxy(name=name, path=path, obj=value)
                self.obj_refs_by_name[name] = ref
                self.obj_refs_by_id[id(value)] = ref
                return ref
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
