"""
Helpers for the Sisyphus setup
"""

from __future__ import annotations
from typing import Tuple, Any
import os
import importlib


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
        if module_name[pos] != ".":
            continue
        mod = importlib.import_module(module_name[:pos])
        setup_base_name = getattr(mod, "__setup_base_name__", None)
        if setup_base_name:
            return module_name[:pos], setup_base_name
    raise ValueError(f"Could not find base module name for {module_name}")


def get_setup_prefix_for_module(module_name: str) -> str:
    """
    :param module_name: e.g. "i6_experiments.users.zeyer.experiments.exp2024_04_23.foo.bar"
    :return: some setup prefix name, e.g. "foo/bar".  the base module is determined via :func:`get_base_module`
    """
    base_module_name, setup_name = get_base_module_from_module(module_name)
    return module_name[len(base_module_name) + 1 :].replace(".", "/")
