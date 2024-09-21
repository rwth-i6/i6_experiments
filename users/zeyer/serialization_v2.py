"""
New simplified config serialization, usually for RETURNN configs.
See :doc:`serialization.rst` for some overview.

This is conceptually similar to :class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`
and :func:`i6_experiments.common.setups.returnn.serialization.get_serializable_config`.

See :func:`serialize_config` for the main entry point.

Note: Sisyphus hashes are currently just defined by the config keys/values,
using the `sis_hash_helper` function, without any special handling.
That means, e.g. functions/classes get hashed by ``(obj.__module__, obj.__qualname__)``.

Note: Sisyphus Path objects are serialized directly using :func:`sisyphus.Path.get_path`.

We currently don't handle any generic object,
but only:
- primitive types (int, float, bool, str)
- Sisyphus Path objects
- RETURNN Dim objects
- dict, list, tuple
- functions, classes, modules

Note: We could handle any generic object, in the same way as pickle does it, or
:class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`,
by using ``obj = object.__new__(obj_type)`` and ``obj.__setstate__(...)``.
However, that generated code is somewhat ugly and complex.
For all our current use cases, we don't need this,
and we handle those cases explicitly.

Note: We do not handle circular references yet.
I think we would also need the more generic object creation for that
(``obj = object.__new__(obj_type)`` first),
which makes the generated code really ugly and complex.

TODO support post_config as additional argument to serialize_config.
TODO test on some real configs
"""

from __future__ import annotations

import sys
import os
import re
import builtins
from typing import Optional, Union, Any, Dict, List
from types import FunctionType, BuiltinFunctionType, ModuleType
from dataclasses import dataclass
import textwrap
import subprocess

from returnn.tensor import Dim, batch_dim, single_step_dim
from sisyphus import Path
from sisyphus.hash import sis_hash_helper
from i6_core.serialization.base import SerializerObject, Collection
from i6_experiments.common.utils.python import is_valid_python_identifier_name


def serialize_config(config: Dict[str, Any], *, inlining: bool = True) -> SerializedConfig:
    """serialize config. see module docstring for more info."""
    serializer = _Serializer(config)
    serializer.work_queue()
    if inlining:
        serializer.work_inlining()
    return SerializedConfig(code_list=list(serializer.assignments_dict_by_idx.values()))


@dataclass
class SerializedConfig:
    code_list: List[PyCode]

    def as_serialization_collection(self) -> Collection:
        """as serialization Collection"""
        return Collection(self.code_list)

    def as_serialized_code(self) -> str:
        """as serialized code"""
        return "".join(code.py_code for code in self.code_list)


class _Serializer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config.copy()
        self.required_names_by_value_ref: Dict[_Ref, str] = {}  # value ref -> var name
        for key, value in config.items():
            if _Ref(value) not in self.required_names_by_value_ref:
                self.required_names_by_value_ref[_Ref(value)] = key
        self.assignments_dict_by_value_ref: Dict[_Ref, PyCode] = {}  # value ref -> code
        self.assignments_dict_by_name: Dict[str, PyCode] = {}  # var name -> code
        self.assignments_dict_by_idx: Dict[int, PyCode] = {}  # idx -> code
        self.assignments_dict_by_value_by_type: Dict[type, Dict[Any, PyCode]] = {Dim: {}}  # type -> dict value -> code
        self.added_sys_paths = set()
        self._cur_added_refs: List[PyCode] = []
        self._next_alignment_idx = 0
        # We first serialize everything without inlining anything.
        # There we also count how often a value is used (ref_count).
        # Then we can inline those values which are not direct config entries
        # and which are only used once.
        self._inlining_stage = False

    def work_queue(self):
        self._inlining_stage = False
        queue: List[_AssignQueueItem] = [
            _AssignQueueItem(required_var_name=key, value=value) for key, value in self.config.items()
        ]
        queue.reverse()  # we will pop from the end
        while queue:
            try:
                queue_item = queue[-1]
                self._cur_added_refs.clear()
                self.handle_next_queue_item(queue_item)
                assert queue[-1] is queue_item
                queue.pop(-1)
            except _SerializationDependsOnNotYetSerializedOtherVarException as exc:
                queue.append(exc.queue_item)
                for code in self._cur_added_refs:
                    code.ref_count -= 1

    def work_inlining(self):
        self._inlining_stage = True
        self._next_alignment_idx = -1
        for assign in list(self.assignments_dict_by_idx.values()):
            assert assign.idx > self._next_alignment_idx
            self._next_alignment_idx = assign.idx
            if assign.py_name:
                new_assign = self._serialize_value_assignment(assign.value, name=assign.py_name)
                assign.py_value_repr = new_assign.py_value_repr
                assign.py_code = new_assign.py_code
        self._next_alignment_idx += 1

    def handle_next_queue_item(self, queue_item: _AssignQueueItem):
        value_ref = _Ref(queue_item.value)
        if value_ref in self.assignments_dict_by_value_ref:
            # Maybe it was already assigned before.
            return
        if queue_item.required_var_name is None:
            # Maybe the object got queued, and we wanted to assign it to a specific name.
            queue_item.required_var_name = self.required_names_by_value_ref.get(value_ref)
        name = queue_item.required_var_name
        if not name and value_ref in _InternalReservedNamesByValueRef:
            name = self._get_unique_suggested_name(
                _InternalReservedNamesByValueRef[value_ref], allow_internal_reserved_name=True
            )
        if not name and (
            isinstance(queue_item.value, (type, FunctionType, BuiltinFunctionType, ModuleType))
            or (getattr(queue_item.value, "__module__", None) and getattr(queue_item.value, "__qualname__", None))
            or (isinstance(queue_item.value, Dim) and queue_item.value.name)
        ):
            # For those types, prefer a name based on the value, even over any other suggested name.
            name = self._get_unique_suggested_name(self._suggest_name_from_value(queue_item.value))
        if not name and queue_item.suggested_var_name:
            name = self._get_unique_suggested_name(queue_item.suggested_var_name)
        if not name:
            name = self._get_unique_suggested_name(self._suggest_name_from_value(queue_item.value))
        serialized = self._serialize_value_assignment(value=queue_item.value, name=name)
        serialized.idx = self._next_alignment_idx
        self._next_alignment_idx += 1
        if queue_item.required_var_name:
            serialized.is_direct_config_entry = True
        assert serialized.py_name == name
        assert name not in self.assignments_dict_by_name  # double check
        self.assignments_dict_by_name[name] = serialized
        if value_ref in self.assignments_dict_by_value_ref:
            if serialized.is_direct_config_entry:
                # It should not happen that the previous entry was not a direct config entry,
                # it should have used some config key name
                # -- see the code above with required_names_by_value_ref.
                assert self.assignments_dict_by_value_ref[value_ref].is_direct_config_entry
        else:
            self.assignments_dict_by_value_ref[value_ref] = serialized
        value_dict = self.assignments_dict_by_value_by_type.get(type(queue_item.value))
        if value_dict is not None:
            if queue_item.value in value_dict:
                if serialized.is_direct_config_entry:
                    # Same reasoning as above for assignments_dict_by_value_ref.
                    assert value_dict[queue_item.value].is_direct_config_entry
            else:
                value_dict[queue_item.value] = serialized
        self.assignments_dict_by_idx[serialized.idx] = serialized

    @staticmethod
    def _suggest_name_from_value(value: Any) -> str:
        if isinstance(value, Dim):
            return _Serializer._suggested_name_for_dim(value)
        if getattr(value, "__module__", None) and getattr(value, "__qualname__", None):
            return f"{value.__module__}.{value.__qualname__}".replace(".", "_")
        if getattr(value, "__qualname__", None):
            return value.__qualname__.replace(".", "_")
        if getattr(value, "__name__", None):
            return value.__name__
        return type(value).__name__.lower()

    @staticmethod
    def _suggested_name_for_dim(dim: Dim) -> str:
        if not dim.name:
            return "dim"  # fallback
        name_ = dim.name
        name_ = re.sub(r"[^a-zA-Z0-9_]", "_", name_)
        if not name_:
            return "dim"  # fallback
        if name_[:1].isdigit():
            return "dim_" + name_
        if not name_.endswith("_dim"):
            name_ += "_dim"
        return name_

    def _get_unique_suggested_name(self, suggested_name: str, *, allow_internal_reserved_name: bool = False) -> str:
        # If we ever get here and the suggested name is not a valid Python identifier,
        # then we can sanitize it here.
        assert is_valid_python_identifier_name(suggested_name)  # not handled yet otherwise...
        if self._check_can_use_suggested_name(
            suggested_name, allow_internal_reserved_name=allow_internal_reserved_name
        ):
            return suggested_name
        i = 1
        while True:
            name = f"{suggested_name}_{i}"
            if self._check_can_use_suggested_name(name, allow_internal_reserved_name=allow_internal_reserved_name):
                return name
            i += 1

    def _check_can_use_suggested_name(self, name: str, *, allow_internal_reserved_name: bool = False) -> bool:
        if not allow_internal_reserved_name and name in _InternalReservedNames:
            return False
        if name in builtins.__dict__:  # e.g. `len`, `sum`, etc.
            return False
        if name in self.config:
            return False
        if name in self.assignments_dict_by_name:
            return False
        return True

    def _serialize_value_assignment(self, value: Any, name: str) -> PyCode:
        serialized = self._serialize_value(value=value, prefix=name, recursive=False)
        if isinstance(serialized, PyEvalCode):
            return PyCode(
                py_name=name,
                value=value,
                py_code=f"{name} = {serialized.py_value_repr}\n",
                py_value_repr=serialized,
            )
        elif isinstance(serialized, PyCode):
            return serialized
        else:
            raise TypeError(f"unexpected serialized type {type(serialized).__name__}")

    def _serialize_value(self, value: Any, prefix: str, *, recursive: bool = True) -> Union[PyEvalCode, PyCode]:
        value_ref = _Ref(value)
        if value is None:
            return PyEvalCode("None")
        if isinstance(value, (int, float, bool, str)):
            return PyEvalCode(repr(value))
        if isinstance(value, Path):
            # Note: If we would want to have Sisyphus file_caching support here,
            # we could also refer to that file_caching function,
            # and call it here in the generated code.
            return PyEvalCode(repr(value.get_path()))
        if getattr(value, "__module__", None) == "builtins":
            name: str = getattr(value, "__name__", None)
            if name and getattr(builtins, name, None) is value:
                assign = self.assignments_dict_by_name.get(name)
                if not assign or assign.idx >= self._next_alignment_idx:
                    return PyEvalCode(name)
                # name was overwritten. fallback to standard module access.
        if value_ref in self.assignments_dict_by_value_ref:
            assign = self.assignments_dict_by_value_ref[value_ref]
            if self._inlining_stage:
                if assign.idx >= self._next_alignment_idx:
                    pass  # self, or future ref, cannot use this, proceed serializing
                elif assign.is_direct_config_entry:
                    return PyEvalCode(assign.py_name)  # anyway need to keep this assignment, so just use it
                else:
                    assert assign.ref_count >= 1
                    if assign.ref_count > 1:
                        # there are multiple references, so we need to keep this assignment
                        return PyEvalCode(assign.py_name)
                    if not assign.py_value_repr:
                        return PyEvalCode(assign.py_name)  # we cannot inline this, so just use the assignment
                    # We can inline this.
                    # Thus remove the reference to this assignment.
                    assign.ref_count -= 1
                    assert assign.ref_count == 0
                    # Can delete this assignment.
                    del self.assignments_dict_by_value_ref[value_ref]
                    del self.assignments_dict_by_name[assign.py_name]
                    del self.assignments_dict_by_idx[assign.idx]
                    return assign.py_value_repr
            else:
                assign.ref_count += 1
                self._cur_added_refs.append(assign)
                return PyEvalCode(assign.py_name)
        if not self._inlining_stage:
            value_dict = self.assignments_dict_by_value_by_type.get(type(value))
            if value_dict is not None and value in value_dict:
                assign = value_dict.get(value)
                if assign is not None:
                    assign.ref_count += 1
                    self._cur_added_refs.append(assign)
                    return PyEvalCode(assign.py_name)
        if recursive:
            assert not self._inlining_stage  # should not get here when inlining
            raise _SerializationDependsOnNotYetSerializedOtherVarException(
                _AssignQueueItem(value=value, suggested_var_name=prefix)
            )
        if isinstance(value, dict):
            return self._serialize_dict(value, prefix)
        if isinstance(value, list):
            return self._serialize_list(value, prefix)
        if isinstance(value, tuple):
            return self._serialize_tuple(value, prefix)
        if isinstance(value, Dim):
            return self._serialize_dim(value, prefix)
        exc = None
        if isinstance(value, (type, FunctionType, BuiltinFunctionType, ModuleType)) or (
            getattr(value, "__module__", None) and getattr(value, "__qualname__", None)
        ):
            try:
                return self._serialize_global(value=value, name=prefix)
            except _SerializationCannotBeAsValue as exc_:
                exc = exc_
        assert not self._inlining_stage  # should really not happen in this stage
        raise NotImplementedError(
            f"cannot handle `({prefix}) = {value!r}` (value type {type(value).__name__})"
        ) from exc

    def _serialize_dict(self, values: dict, prefix: str) -> PyEvalCode:
        assert type(values) is dict  # nothing else expected/handled currently
        serialized_items = []
        for key, value in values.items():
            serialized_key = self._serialize_value(key, prefix=f"{prefix}_key", recursive=True)
            assert isinstance(serialized_key, PyEvalCode)
            if (isinstance(key, str) and is_valid_python_identifier_name(key)) or isinstance(key, (int, bool)):
                prefix_name = str(key)
            else:
                prefix_name = "value"
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{prefix_name}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(f"{serialized_key.py_inline()}: {serialized_value.py_inline()}")
        return PyEvalCode("{" + ", ".join(serialized_items) + "}")

    def _serialize_list(self, values: list, prefix: str) -> PyEvalCode:
        assert type(values) is list  # nothing else expected/handled currently
        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(serialized_value.py_inline())
        return PyEvalCode("[" + ", ".join(serialized_items) + "]")

    def _serialize_tuple(self, values: tuple, prefix: str) -> PyEvalCode:
        if not values:
            if type(values) is tuple:
                return PyEvalCode("()")
            # Assume namedtuple.
            type_s = self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True)
            assert isinstance(type_s, PyEvalCode)
            return PyEvalCode(f"{type_s.py_inline()}()")

        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            assert isinstance(serialized_value, PyEvalCode)
            serialized_items.append(serialized_value.py_inline())

        if type(values) is tuple:
            return PyEvalCode("(" + ", ".join(serialized_items) + ",)")
        # Assume namedtuple.
        # noinspection PyUnresolvedReferences,PyProtectedMember
        fields = values._fields
        assert len(fields) == len(serialized_items)
        value_type_str = self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True)
        assert isinstance(value_type_str, PyEvalCode)
        return PyEvalCode(
            f"{value_type_str.py_inline()}("
            + ", ".join(f"{key}={value}" for key, value in zip(fields, serialized_items))
            + ")"
        )

    def _serialize_dim(self, dim: Dim, prefix: str) -> Union[PyEvalCode, PyCode]:
        assert isinstance(dim, Dim)
        # See also returnn_common.nn.naming.ReturnnDimTagsProxy.dim_ref_repr
        # and returnn_common.nn.naming.ReturnnDimTagsProxy.DimRefProxy.dim_repr.
        if dim == batch_dim:
            return self._serialize_global(dim, prefix, mod_name="returnn.tensor", qualname="batch_dim")
        if dim == single_step_dim:
            return self._serialize_global(dim, prefix, mod_name="returnn.tensor", qualname="single_step_dim")

        if dim.match_priority:
            base_dim_str = self._serialize_value(dim.copy(match_priority=0), prefix=f"{prefix}_p0", recursive=True)
            assert isinstance(base_dim_str, PyEvalCode)
            return PyEvalCode(f"{base_dim_str.py_inline()}.copy(match_priority={dim.match_priority})")
        if not dim.derived_from_op and dim.get_same_base().derived_from_op:
            dim = dim.get_same_base()

        if dim.derived_from_op:
            if dim.derived_from_op.kind == "constant":
                v = dim.derived_from_op.attribs["value"]
                return PyEvalCode(str(v), need_brackets_when_inlined=v < 0)
            func_map = {"truediv_left": "div_left", "ceildiv_left": "ceildiv_left", "ceildiv_right": "ceildiv_right"}
            inputs_s: List[PyEvalCode] = [
                self._serialize_value(x, prefix=f"{prefix}_in{i}", recursive=True)
                for i, x in enumerate(dim.derived_from_op.inputs)
            ]
            assert all(isinstance(x, PyEvalCode) for x in inputs_s)
            if dim.derived_from_op.kind in func_map:
                assert len(dim.derived_from_op.inputs) == 2
                a, b = inputs_s
                a: PyEvalCode
                b: PyEvalCode
                return PyEvalCode(f"{a.py_inline()}.{func_map[dim.derived_from_op.kind]}({b.py_inline()})")
            op_str = {"add": "+", "mul": "*", "truediv_right": "//", "floordiv_right": "//"}[dim.derived_from_op.kind]
            s = f" {op_str} ".join(x.py_inline() for x in inputs_s)
            return PyEvalCode(s, need_brackets_when_inlined=True)

        # generic fallback
        dim_type_str = self._serialize_value(type(dim), prefix="Dim", recursive=True)
        assert isinstance(dim_type_str, PyEvalCode)
        kwargs = {"name": repr(dim.name)}
        if dim.kind is not None:
            kind_s = {Dim.Types.Batch: "Batch", Dim.Types.Spatial: "Spatial", Dim.Types.Feature: "Feature"}[dim.kind]
            kwargs["kind"] = f"{dim_type_str.py_inline()}.Types.{kind_s}"
        return PyEvalCode(
            f"{dim_type_str.py_inline()}"
            f"({dim.dimension}, {', '.join(f'{key}={value}' for key, value in kwargs.items())})"
        )

    def _serialize_global(
        self, value: Any, name: str, *, mod_name: Optional[str] = None, qualname: Optional[str] = None
    ) -> Union[PyEvalCode, PyCode]:
        mod_name = mod_name or getattr(value, "__module__", None)
        if not mod_name:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, no __module__"
            )
        mod = sys.modules.get(mod_name)
        if not mod:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, unknown __module__ {mod_name!r}"
            )
        qualname = qualname or getattr(value, "__qualname__", None)
        if not qualname:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, no __qualname__"
            )
        qualname_parts = qualname.split(".")
        obj = [mod]
        for i in range(len(qualname_parts)):
            if not hasattr(obj[-1], qualname_parts[i]):
                raise _SerializationCannotBeAsValue(
                    f"cannot handle {value!r} (type {type(value).__name__}) as global,"
                    f" qualname {qualname} not found,"
                    f" no {'.'.join(qualname_parts[:i + 1])} in module {mod_name}"
                )
            obj.append(getattr(obj[-1], qualname_parts[i]))
        if obj[-1] is not value:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global,"
                f" qualname {qualname} gives different object {obj[-1]!r}"
            )
        if len(qualname_parts) > 1:
            base_obj_repr = self._serialize_value(obj[-2], prefix=name + "_base")
            return PyEvalCode(f"{base_obj_repr}.{qualname_parts[-1]}")
        if "." in mod_name:
            # Maybe we can shorten the import.
            # Check if some of the parent modules already import the object.
            mod_name_parts = mod_name.split(".")
            for i in range(len(mod_name_parts)):
                parent_mod_name = ".".join(mod_name_parts[: i + 1])
                mod = sys.modules.get(parent_mod_name)
                if mod and getattr(mod, qualname, None) is value:
                    mod_name = parent_mod_name  # we can directly use this
                    break
        self._setup_module_import(mod_name)
        return PyCode(
            py_name=name,
            value=value,
            py_code=f"from {mod_name} import {qualname}\n"
            if qualname == name
            else f"from {mod_name} import {qualname} as {name}\n",
        )

    def _setup_module_import(self, mod_name: str):
        """make sure that the import works, by preparing ``sys.path`` if necessary"""
        if "." in mod_name:
            mod_name = mod_name.split(".", 1)[0]
        mod = sys.modules[mod_name]
        if not hasattr(mod, "__file__"):
            return  # assume builtin module or so
        mod_filename = mod.__file__
        if mod_filename.endswith("/__init__.py"):
            mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])
        else:
            mod_path = os.path.dirname(mod_filename)
        if mod_path in self.added_sys_paths:
            return  # already added
        base_sys_path = [path for path in _get_base_sys_path_list() if path]
        assert base_sys_path
        if mod_path in base_sys_path:
            return  # already in (base) sys.path
        assert mod_path in sys.path
        assert base_sys_path[0] in sys.path
        if sys.path.index(mod_path) < sys.path.index(base_sys_path[0]):
            code = PyCode(py_name=None, value=None, py_code=f"sys.path.insert(0, {mod_path!r})\n")
        else:
            code = PyCode(py_name=None, value=None, py_code=f"sys.path.append({mod_path!r})\n")
        code.idx = self._next_alignment_idx
        self._next_alignment_idx += 1
        self.assignments_dict_by_idx[code.idx] = code
        self.added_sys_paths.add(mod_path)


class _SerializationDependsOnNotYetSerializedOtherVarException(Exception):
    def __init__(self, queue_item: _AssignQueueItem):
        super().__init__(
            f"serialization depends on not yet serialized other var:"
            f" ({queue_item.suggested_var_name}) = {queue_item.value!r} (type {type(queue_item.value).__name__})"
        )
        self.queue_item = queue_item


class _SerializationCannotBeAsValue(Exception):
    """
    As value representation means that we can write::

        <var_name> = <value_repr>

    This is often the case, but not always.
    E.g. modules (e.g. ``sys``) cannot be serialized as value.
    All the primitive types can be serialized as value.
    """


@dataclass
class _AssignQueueItem:
    value: Any
    required_var_name: Optional[str] = None
    suggested_var_name: Optional[str] = None


@dataclass
class PyCode(SerializerObject):
    """
    The Python code will always assign some variable.

    E.g.::

        x = 42  # assign `x`

    Or::

        def f(): ...  # assign `f`

    Or::

        import sys  # assign `sys`
    """

    py_name: Optional[str]
    value: Any
    py_code: str
    py_value_repr: Optional[PyEvalCode] = None
    is_direct_config_entry: bool = False
    ref_count: int = 0  # by other statements
    idx: Optional[int] = None

    def __post_init__(self):
        self.use_for_hash = self.is_direct_config_entry

    def get(self) -> str:
        return self.py_code

    def _sis_hash(self) -> bytes:
        if not self.is_direct_config_entry:
            raise Exception(f"{self} should not be hashed. Maybe wrap this in a serialization Collection")
        return sis_hash_helper((self.py_name, self.value))


@dataclass
class PyEvalCode:
    """
    When some repr can represent the value directly.
    """

    py_value_repr: str
    need_brackets_when_inlined: bool = False  # e.g. for math expressions like `a + b`

    def py_inline(self) -> str:
        return f"({self.py_value_repr})" if self.need_brackets_when_inlined else self.py_value_repr


class _Ref:
    def __init__(self, value: Any):
        self.value = value

    def __repr__(self):
        return f"_Ref({self.value!r})"

    def __hash__(self):
        return id(self.value)

    def __eq__(self, other: _Ref):
        return self.value is other.value

    def __ne__(self, other: _Ref):
        return not (self == other)


# Avoid to use them, but if necessary (when inside the config), they can be used.
# The config keys always have precedence.
_InternalReservedNames = {
    "sys": sys,
    "batch_dim": batch_dim,
    "single_step_dim": single_step_dim,
    "Dim": Dim,
}
_InternalReservedNamesByValueRef = {_Ref(value): name for name, value in _InternalReservedNames.items()}


_base_sys_path_list: Optional[str] = None


def _get_base_sys_path_list() -> List[str]:
    global _base_sys_path_list
    if _base_sys_path_list is None:
        env_copy = os.environ.copy()
        env_copy.pop("PYTHONPATH", None)
        _base_sys_path_list = eval(
            subprocess.check_output([sys.executable, "-c", "import sys; print(sys.path)"], env=env_copy)
            .decode("utf8")
            .strip()
        )
        assert isinstance(_base_sys_path_list, list) and all(isinstance(p, str) for p in _base_sys_path_list)
    return _base_sys_path_list


def test_basic():
    assert serialize_config({"var1": 42, "var2": "foo"}).as_serialized_code() == "var1 = 42\nvar2 = 'foo'\n"


def test_recursive():
    d_base = {"key": 1}
    d_other = {"key": 2, "base": d_base}
    # It should serialize d_base first, even when we have d_other first here in the dict.
    assert serialize_config({"first": d_other, "second": d_base}).as_serialized_code() == textwrap.dedent(
        """\
        second = {'key': 1}
        first = {'key': 2, 'base': second}
        """
    )


def test_inlining():
    d = {"d": {"k1": 1, "k2": {"k3": 3, "k4": 4}}}
    assert serialize_config(d).as_serialized_code() == f"d = {d['d']!r}\n"
    assert serialize_config(d, inlining=False).as_serialized_code() == textwrap.dedent(
        """\
        d_k2 = {'k3': 3, 'k4': 4}
        d = {'k1': 1, 'k2': d_k2}
        """
    )


def test_builtin():
    d = {"func": sum}
    assert serialize_config(d).as_serialized_code() == f"func = sum\n"


def test_builtin_as_is():
    d = {"sum": sum}
    assert serialize_config(d).as_serialized_code() == f"sum = sum\n"  # might change in the future...


def test_builtin_overwrite():
    d = {"sum": 42, "func": sum}
    assert serialize_config(d).as_serialized_code() == f"sum = 42\nfrom builtins import sum as func\n"


def test_func():
    import i6_experiments
    from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model

    mod_filename = i6_experiments.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"get_model": _returnn_v2_get_model}
    assert serialize_config(config).as_serialized_code() == textwrap.dedent(
        f"""\
        sys.path.insert(0, {mod_path!r})
        from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model as get_model
        """
    )


def test_batch_dim():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"dim": batch_dim}
    assert serialize_config(config, inlining=False).as_serialized_code() == textwrap.dedent(
        f"""\
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim as dim
        """
    )


def test_dim():
    import returnn

    mod_filename = returnn.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    time_dim = Dim(None, name="time")
    feat_dim = Dim(42, name="feature")
    config = {"extern_data": {"data": {"dims": [batch_dim, time_dim, feat_dim]}}}
    assert serialize_config(config, inlining=False).as_serialized_code() == textwrap.dedent(
        f"""\
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim
        from returnn.tensor import Dim
        time_dim = Dim(None, name='time')
        feature_dim = Dim(42, name='feature')
        extern_data_data_dims = [batch_dim, time_dim, feature_dim]
        extern_data_data = {{'dims': extern_data_data_dims}}
        extern_data = {{'data': extern_data_data}}
        """
    )
    assert serialize_config(config).as_serialized_code() == textwrap.dedent(
        f"""\
        sys.path.insert(0, {mod_path!r})
        from returnn.tensor import batch_dim
        from returnn.tensor import Dim
        extern_data = {{'data': {{'dims': [batch_dim, Dim(None, name='time'), Dim(42, name='feature')]}}}}
        """
    )


def test_sis_path():
    from sisyphus import Path

    config = {"path": Path("/foo.txt")}
    assert serialize_config(config).as_serialized_code() == "path = '/foo.txt'\n"
