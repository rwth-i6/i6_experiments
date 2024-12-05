"""
Test the dump_py_code module. This is used by dependency_boundary.

Als see test_serialization_v2 for very similar tests.
"""

from typing import Any
from textwrap import dedent
from io import StringIO
from i6_experiments.common.utils.dump_py_code import PythonCodeDumper


def serialize(obj: Any) -> str:
    out = StringIO()
    dumper = PythonCodeDumper(file=out)
    dumper.dump(obj, lhs="obj")
    return out.getvalue()


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
    print(code)
    scope = {}
    exec(code, scope)
    assert scope["obj"]["func"] is sum


def _demo_func(*args):
    return sum(args, 42)


def test_func_in_dict():
    d = {"func": _demo_func}
    code = serialize(d)
    print(code)
    scope = {}
    exec(code, scope)
    assert scope["obj"]["func"] is _demo_func


def test_func():
    code = serialize(_demo_func)
    print(code)
    scope = {}
    exec(code, scope)
    assert scope["obj"] is _demo_func
