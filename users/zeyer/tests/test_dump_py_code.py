"""
Test the dump_py_code module. This is used by dependency_boundary.

Als see test_serialization_v2 for very similar tests.
"""

from typing import Any, Callable
from textwrap import dedent
from io import StringIO
import dataclasses
import functools

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


def _func(a, *, b):
    return a + b


def test_functools_partial():
    f_orig = functools.partial(_func, b=1)
    code = serialize(f_orig)
    # Not really checking the exact serialized code here,
    # but instead just testing to execute it.
    scope = {}
    exec(code, scope)
    f = scope["obj"]
    assert f is not f_orig
    assert isinstance(f, functools.partial)
    assert f.func is _func
    assert not f.args
    assert f.keywords == {"b": 1}
    assert f(2) == 3


def test_batch_dim():
    from returnn.tensor import batch_dim

    code = serialize(batch_dim)
    scope = {}
    exec(code, scope)
    f = scope["obj"]
    assert f is batch_dim


def test_dim():
    from returnn.tensor import Dim, batch_dim

    time_dim = Dim(None, name="time")
    feat_dim = Dim(42, name="feature")
    dims = [batch_dim, time_dim, feat_dim]
    code = serialize(dims)
    scope = {}
    exec(code, scope)
    dims_ = scope["obj"]
    assert dims_ is not dims
    assert len(dims_) == 3
    assert dims_[0] is batch_dim
    _, time_dim2, feat_dim2 = dims_
    assert time_dim2 is not time_dim
    assert time_dim2.size == time_dim.size
    assert time_dim2.name == time_dim.name
    assert feat_dim2 is not feat_dim
    assert feat_dim2.size == feat_dim.size
    assert feat_dim2.name == feat_dim.name
