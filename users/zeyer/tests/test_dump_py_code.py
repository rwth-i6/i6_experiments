"""
Test the dump_py_code module. This is used by dependency_boundary.

Als see test_serialization_v2 for very similar tests.
"""

from typing import Any, Callable
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


@dataclasses.dataclass
class _DataclassWithBoundMethod:
    name: str

    def default_collect_score_results(self, x: str) -> str:
        return self.name + " " + x

    collect_score_results_func: Callable[[str], str] = None

    def __post_init__(self):
        if self.collect_score_results_func is None:
            self.collect_score_results_func = self.default_collect_score_results  # bound method


def test_bound_method():
    obj = _DataclassWithBoundMethod("foo")
    assert obj.collect_score_results_func("bar") == "foo bar"
    assert obj.collect_score_results_func.__self__ is obj
    code = serialize(obj)
    print(code)
    scope = {}
    exec(code, scope)
    obj_ = scope["obj"]
    assert obj_ is not obj
    assert isinstance(obj_, _DataclassWithBoundMethod)
    assert obj_.collect_score_results_func is not obj.collect_score_results_func
    assert (
        obj_.default_collect_score_results.__func__
        is obj.default_collect_score_results.__func__
        is _DataclassWithBoundMethod.default_collect_score_results
    )
    assert obj_.collect_score_results_func.__self__ is obj_
