"""
Helpers for the Sisyphus setup
"""

from __future__ import annotations
from typing import Tuple, Any
import os
import importlib
import contextlib
from sisyphus import tk


_my_dir = os.path.dirname(os.path.abspath(__file__))


def get_base_module(obj: Any) -> Tuple[str, str]:
    """
    :param obj: some object from a module, determines automatically the module
    :return: tuple (base module name, setup name),
        e.g. ("i6_experiments.users.zeyer.experiments.exp2024_04_23", "zeyer-2024-04-23-baselines")
        Uses :func:`get_base_module_from_module`.
    """
    return get_base_module_from_module(obj.__module__)


def get_base_module_from_module(module_name: str) -> Tuple[str, str]:
    """
    :param module_name: e.g. "i6_experiments.users.zeyer.experiments.exp2024_04_23"
    :return: tuple (base module name, setup name),
        e.g. ("i6_experiments.users.zeyer.experiments.exp2024_04_23", "zeyer-2024-04-23-baselines")
    """
    for pos in range(len(module_name), 0, -1):
        if pos < len(module_name) and module_name[pos] != ".":
            continue
        mod = importlib.import_module(module_name[:pos])
        setup_base_name = getattr(mod, "__setup_base_name__", None)
        if setup_base_name:
            return module_name[:pos], setup_base_name
    raise ValueError(
        f"Could not find base module name for {module_name}. Set __setup_base_name__ in the module or any parents."
    )


def get_setup_prefix_for_module(module_name: str) -> str:
    """
    :param module_name: e.g. "i6_experiments.users.zeyer.experiments.exp2024_04_23.foo.bar"
    :return: some setup prefix name, e.g. "foo/bar".  the base module is determined via :func:`get_base_module`.
        Or alternatively, any ``__setup_root_prefix__`` attribute in the module hierarchy is used
        when found earlier than __setup_base_name__.
    """
    for pos in range(len(module_name), 0, -1):
        if pos < len(module_name) and module_name[pos] != ".":
            continue
        mod = importlib.import_module(module_name[:pos])
        setup_root_prefix = getattr(mod, "__setup_root_prefix__", None)
        if setup_root_prefix:
            return setup_root_prefix
        if getattr(mod, "__setup_base_name__", None):
            return module_name[pos + 1 :].replace(".", "/")
    raise ValueError(f"Could not find setup prefix for {module_name}")


_register_output_enabled = True
_orig_register_output = tk.register_output


@contextlib.contextmanager
def disable_register_output():
    global _register_output_enabled
    old = _register_output_enabled
    old_register_output = tk.register_output
    try:
        _register_output_enabled = False
        tk.register_output = _no_op_register_output  # somewhat hacky...
        yield
    finally:
        _register_output_enabled = old
        tk.register_output = old_register_output


def _no_op_register_output(*_args, **_kwargs):
    pass


def is_register_output_enabled():
    return _register_output_enabled
