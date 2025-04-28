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
    if is_valid_setup_base_dir(_cwd):
        _setup_base_dir = _cwd
        return _setup_base_dir
    # Algo: Go up until we find the `work` dir which includes some training jobs.
    d = _get_setup_base_dir_from_deep_dir(_my_dir)
    if not d:
        raise Exception(f"Could not find setup base dir, starting from {_my_dir}")
    _setup_base_dir = d
    return d


def _get_setup_base_dir_from_deep_dir(d_: str) -> Optional[str]:
    d = d_
    while True:
        if is_valid_setup_base_dir(d):
            return d
        if d and d != "/":
            d = os.path.dirname(d)
            continue
        return None


def set_setup_base_dir(d: str):
    global _setup_base_dir, _work_dir
    assert is_valid_setup_base_dir(d)
    _setup_base_dir = d
    _work_dir = None


def is_valid_setup_base_dir(d: str) -> bool:
    return os.path.exists(f"{d}/work") and os.path.exists(f"{d}/recipe")


def get_setup_base_dir_from_job(job: str) -> Optional[str]:
    if job.startswith(_cwd + "/") or job.startswith("work/") or job.startswith("output/"):
        if is_valid_setup_base_dir(_cwd):
            return _cwd
    d = _get_setup_base_dir_from_deep_dir(job)
    if d:
        return d
    if os.path.islink(job):
        d = _get_setup_base_dir_from_deep_dir(os.readlink(job))
        if d:
            return d
    return None


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
