"""
New simplified config serialization, usually for RETURNN configs.
See :doc:`serialization.rst` for some overview.

This is conceptually similar to :class:`i6_experiments.common.utils.dump_py_code.PythonCodeDumper`
and :func:`i6_experiments.common.setups.returnn.serialization.get_serializable_config`.
"""

from __future__ import annotations

import sys
import os
import builtins
from typing import Optional, Union, Any, Dict, List
from types import FunctionType, BuiltinFunctionType, ModuleType
from dataclasses import dataclass
import textwrap
import subprocess
from returnn.tensor import Dim, batch_dim, single_step_dim
from i6_experiments.common.utils.python import is_valid_python_identifier_name


def serialize_config(config: Dict[str, Any], *, inlining: bool = True) -> List[PyCode]:
    """serialize config"""
    serializer = _Serializer(config)
    serializer.work_queue()
    if inlining:
        serializer.work_inlining()
    return list(serializer.assignments_dict_by_idx.values())


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
        self.added_sys_paths = set()
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
                self.handle_next_queue_item(queue_item)
                assert queue[-1] is queue_item
                queue.pop(-1)
            except _SerializationDependsOnNotYetSerializedOtherVarException as exc:
                queue.append(exc.queue_item)

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

    def handle_next_queue_item(self, assign: _AssignQueueItem):
        value_ref = _Ref(assign.value)
        if value_ref in self.assignments_dict_by_value_ref:
            # Maybe it was already assigned before.
            return
        if assign.required_var_name is None:
            # Maybe the object got queued, and we wanted to assign it to a specific name.
            assign.required_var_name = self.required_names_by_value_ref.get(value_ref)
        name = assign.required_var_name
        if not name and value_ref in _InternalReservedNamesByValueRef:
            name = self._get_unique_suggested_name(
                _InternalReservedNamesByValueRef[value_ref], allow_internal_reserved_name=True
            )
        if not name and (
            isinstance(assign.value, (type, FunctionType, BuiltinFunctionType, ModuleType, Dim))
            or (getattr(assign.value, "__module__", None) and getattr(assign.value, "__qualname__", None))
        ):
            # For those types, prefer a name based on the value, even over any other suggested name.
            name = self._get_unique_suggested_name(self._suggest_name_from_value(assign.value))
        if not name and assign.suggested_var_name:
            name = self._get_unique_suggested_name(assign.suggested_var_name)
        if not name:
            name = self._get_unique_suggested_name(self._suggest_name_from_value(assign.value))
        serialized = self._serialize_value_assignment(value=assign.value, name=name)
        serialized.idx = self._next_alignment_idx
        self._next_alignment_idx += 1
        if assign.required_var_name:
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
        self.assignments_dict_by_idx[serialized.idx] = serialized

    @staticmethod
    def _suggest_name_from_value(value: Any) -> str:
        if getattr(value, "__module__", None) and getattr(value, "__qualname__", None):
            return f"{value.__module__}.{value.__qualname__}".replace(".", "_")
        if getattr(value, "__qualname__", None):
            return value.__qualname__.replace(".", "_")
        if getattr(value, "__name__", None):
            return value.__name__
        return type(value).__name__.lower()

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
        if isinstance(serialized, str):
            return PyCode(py_name=name, value=value, py_code=f"{name} = {serialized}\n", py_value_repr=serialized)
        elif isinstance(serialized, PyCode):
            return serialized
        else:
            raise TypeError(f"unexpected serialized type {type(serialized).__name__}")

    def _serialize_value(self, value: Any, prefix: str, *, recursive: bool = True) -> Union[str, PyCode]:
        value_ref = _Ref(value)
        if value is None:
            return "None"
        if isinstance(value, (int, float, bool, str)):
            return repr(value)
        if getattr(value, "__module__", None) == "builtins":
            name = getattr(value, "__name__", None)
            if name and getattr(builtins, name, None) is value:
                assign = self.assignments_dict_by_name.get(name)
                if not assign or assign.idx >= self._next_alignment_idx:
                    return name
                # name was overwritten. fallback to standard module access.
        if value_ref in self.assignments_dict_by_value_ref:
            assign = self.assignments_dict_by_value_ref[value_ref]
            if self._inlining_stage:
                if assign.idx >= self._next_alignment_idx:
                    pass  # self, or future ref, cannot use this, proceed serializing
                elif assign.is_direct_config_entry:
                    return assign.py_name  # anyway need to keep this assignment, so just use it
                else:
                    assert assign.ref_count >= 1
                    if assign.ref_count > 1:
                        return assign.py_name  # there are multiple references, so we need to keep this assignment
                    if not assign.py_value_repr:
                        return assign.py_name  # we cannot inline this, so just use the assignment
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
                return assign.py_name
        if recursive:
            assert not self._inlining_stage  # should not get here when inlining
            raise _SerializationDependsOnNotYetSerializedOtherVarException(
                _AssignQueueItem(value=value, suggested_var_name=f"_{prefix}")
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

    def _serialize_dict(self, values: dict, prefix: str) -> str:
        assert type(values) is dict  # nothing else expected/handled currently
        serialized_items = []
        for key, value in values.items():
            serialized_key = self._serialize_value(key, prefix=f"{prefix}_key", recursive=True)
            if (isinstance(key, str) and is_valid_python_identifier_name(key)) or isinstance(key, (int, bool)):
                prefix_name = str(key)
            else:
                prefix_name = "value"
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{prefix_name}", recursive=True)
            serialized_items.append(f"{serialized_key}: {serialized_value}")
        return "{" + ", ".join(serialized_items) + "}"

    def _serialize_list(self, values: list, prefix: str) -> str:
        assert type(values) is list  # nothing else expected/handled currently
        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            serialized_items.append(serialized_value)
        return "[" + ", ".join(serialized_items) + "]"

    def _serialize_tuple(self, values: tuple, prefix: str) -> str:
        if not values:
            if type(values) is tuple:
                return "()"
            # Assume namedtuple.
            return self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True) + "()"

        serialized_items = []
        for idx, value in enumerate(values):
            serialized_value = self._serialize_value(value, prefix=f"{prefix}_{idx}", recursive=True)
            serialized_items.append(serialized_value)

        if type(values) is tuple:
            return "(" + ", ".join(serialized_items) + ",)"
        # Assume namedtuple.
        # noinspection PyUnresolvedReferences,PyProtectedMember
        fields = values._fields
        assert len(fields) == len(serialized_items)
        value_type_str = self._serialize_value(type(values), prefix=f"{prefix}_type", recursive=True)
        return f"{value_type_str}(" + ", ".join(f"{key}={value}" for key, value in zip(fields, serialized_items)) + ")"

    def _serialize_dim(self, value: Dim, prefix: str) -> str:
        assert isinstance(value, Dim)
        raise NotImplementedError  # TODO...

    def _serialize_global(self, value: Any, name: str) -> Union[str, PyCode]:
        mod_name = getattr(value, "__module__", None)
        if not mod_name:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, no __module__"
            )
        mod = sys.modules.get(mod_name)
        if not mod:
            raise _SerializationCannotBeAsValue(
                f"cannot handle {value!r} (type {type(value).__name__}) as global, unknown __module__ {mod_name!r}"
            )
        qualname = getattr(value, "__qualname__", None)
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
            return f"{base_obj_repr}.{qualname_parts[-1]}"
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
class PyCode:
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
    py_value_repr: Optional[str] = None
    is_direct_config_entry: bool = False
    ref_count: int = 0  # by other statements
    idx: Optional[int] = None


class _Ref:
    def __init__(self, value: Any):
        self.value = value

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


def _serialize_code_list(codes: List[PyCode]) -> str:
    return "".join(code.py_code for code in codes)


def test_basic():
    assert _serialize_code_list(serialize_config({"var1": 42, "var2": "foo"})) == "var1 = 42\nvar2 = 'foo'\n"


def test_recursive():
    d_base = {"key": 1}
    d_other = {"key": 2, "base": d_base}
    # It should serialize d_base first, even when we have d_other first here in the dict.
    assert _serialize_code_list(serialize_config({"first": d_other, "second": d_base})) == textwrap.dedent(
        """\
        second = {'key': 1}
        first = {'key': 2, 'base': second}
        """
    )


def test_inlining():
    d = {"d": {"k1": 1, "k2": {"k3": 3, "k4": 4}}}
    assert _serialize_code_list(serialize_config(d)) == f"d = {d['d']!r}\n"
    assert _serialize_code_list(serialize_config(d, inlining=False)) == textwrap.dedent(
        """\
        _d_k2 = {'k3': 3, 'k4': 4}
        d = {'k1': 1, 'k2': _d_k2}
        """
    )


def test_builtin():
    d = {"func": sum}
    assert _serialize_code_list(serialize_config(d)) == f"func = sum\n"


def test_builtin_as_is():
    d = {"sum": sum}
    assert _serialize_code_list(serialize_config(d)) == f"sum = sum\n"  # might change in the future...


def test_builtin_overwrite():
    d = {"sum": 42, "func": sum}
    assert _serialize_code_list(serialize_config(d)) == f"sum = 42\nfrom builtins import sum as func\n"


def test_func():
    import i6_experiments
    from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model

    mod_filename = i6_experiments.__file__
    assert mod_filename.endswith("/__init__.py")
    mod_path = os.path.dirname(mod_filename[: -len("/__init__.py")])

    config = {"get_model": _returnn_v2_get_model}
    assert _serialize_code_list(serialize_config(config)) == textwrap.dedent(
        f"""\
        sys.path.insert(0, {mod_path!r})
        from i6_experiments.users.zeyer.train_v3 import _returnn_v2_get_model as get_model
        """
    )
