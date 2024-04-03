import os
from typing import Optional


_cwd = os.getcwd()
_my_dir = os.path.dirname(os.path.abspath(__file__))
_setup_base_dir: Optional[str] = None
_work_dir: Optional[str] = None


def get_setup_base_dir() -> str:
    global _setup_base_dir
    if _setup_base_dir:
        return _setup_base_dir
    # Check cwd for some pattern.
    if os.path.exists(f"{_cwd}/work") and os.path.exists(f"{_cwd}/recipe"):
        _setup_base_dir = _cwd
        return _setup_base_dir
    # Algo: Go up until we find the `work` dir which includes some training jobs.
    d = _my_dir
    while True:
        if os.path.exists(f"{d}/work") and os.path.exists(f"{d}/recipe"):
            break
        if d and d != "/":
            d = os.path.dirname(d)
            continue
        raise Exception(f"Could not find setup base dir, starting from {_my_dir}")
    _setup_base_dir = d
    return d


def get_work_dir() -> str:
    global _work_dir
    if _work_dir:
        return _work_dir
    setup_base_dir = get_setup_base_dir()
    work_dir = os.path.realpath(f"{setup_base_dir}/work")
    _work_dir = work_dir
    return work_dir


def get_work_dir_prefix() -> str:
    return get_setup_base_dir() + "/work/"


def get_work_dir_prefix2() -> str:
    return get_work_dir() + "/"
