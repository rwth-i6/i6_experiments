"""
Test the dump_py_code module. This is used by dependency_boundary.

Als see test_serialization_v2 for very similar tests.
"""

from typing import Any
from textwrap import dedent
from io import StringIO
import dataclasses

from i6_experiments.common.utils.dump_py_code import PythonCodeDumper


def serialize(obj: Any) -> str:
    out = StringIO()
    dumper = PythonCodeDumper(file=out)
    dumper.dump(obj, lhs="obj")
    code = out.getvalue()
    print("*** generated code: {")
    print(code)
    print("*** }")
    return code


def test_basic():
    assert serialize({"var1": 42, "var2": "foo"}) == dedent(
        """\
        obj = {
            'var1': 42,
            'var2': 'foo',
        }
        """
    )


def test_builtin():
    d = {"func": sum}
    code = serialize(d)
    assert code == dedent(
        """\
        import builtins
        obj = {
            'func': builtins.sum,
        }
        """
    )
    scope = {}
    exec(code, scope)
    assert scope["obj"]["func"] is sum


def _demo_func(*args):
    return sum(args, 42)


def test_func_in_dict():
    d = {"func": _demo_func}
    code = serialize(d)
    scope = {}
    exec(code, scope)
    assert scope["obj"]["func"] is _demo_func


def test_func():
    code = serialize(_demo_func)
    scope = {}
    exec(code, scope)
    assert scope["obj"] is _demo_func


@dataclasses.dataclass
class _DemoData:
    value: int


def test_dataclass():
    obj = _DemoData(42)
    code = serialize(obj)
    scope = {}
    exec(code, scope)
    obj_ = scope["obj"]
    assert obj_ is not obj
    assert isinstance(obj_, _DemoData)
    assert obj_.value == 42
    assert obj_ == obj


@dataclasses.dataclass(frozen=True)
class _FrozenDemoData:
    value: int


def test_dataclass_frozen():
    obj = _FrozenDemoData(42)
    code = serialize(obj)
    print(code)
    scope = {}
    exec(code, scope)
    obj_ = scope["obj"]
    assert obj_ is not obj
    assert isinstance(obj_, _FrozenDemoData)
    assert obj_.value == 42
    assert obj_ == obj
