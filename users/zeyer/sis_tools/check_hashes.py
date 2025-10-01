#!/usr/bin/env python3

"""
Check hashes...
"""

from __future__ import annotations
from typing import Any, List
import argparse
import os
import sys
import logging
import time
import hashlib
import copy
from dataclasses import dataclass
from collections import deque
from functools import reduce


# It will take the dir of the checked out git repo.
# So you can also only use it there...
_my_dir = os.path.dirname(os.path.realpath(__file__))
_base_recipe_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_setup_base_dir = os.path.dirname(_base_recipe_dir)
_sis_dir = f"{_setup_base_dir}/tools/sisyphus"


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_recipe_dir not in sys.path:
            sys.path.append(_base_recipe_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)

        os.environ["SIS_GLOBAL_SETTINGS_FILE"] = f"{_setup_base_dir}/settings.py"

        try:
            import sisyphus  # noqa
            import i6_experiments  # noqa
        except ImportError:
            print("setup base dir:", _setup_base_dir)
            print("sys.path:")
            for path in sys.path:
                print(f"  {path}")
            raise


_setup()

from sisyphus.loader import config_manager
from sisyphus import gs, tk, Path, Job
import sisyphus.hash
import sisyphus.job_path
from sisyphus.hash import sis_hash_helper as _orig_sis_hash_helper


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_files", nargs="*")
    arg_parser.add_argument("--target")
    args = arg_parser.parse_args()

    # Do that early, such that all imports of sis_hash_helper get our patched version.
    sisyphus.hash.sis_hash_helper = _patched_sis_hash_helper
    sisyphus.job_path.sis_hash_helper = _patched_sis_hash_helper

    start = time.time()
    config_manager.load_configs(args.config_files)
    load_time = time.time() - start
    logging.info("Config loaded (time needed: %.2f)" % load_time)

    sis_graph = tk.sis_graph
    if not args.target:
        print("--target not specified, printing all targets:")
        for name, target in sis_graph.targets_dict.items():
            print(f"Target: {name} -> {target.required_full_list}")
        sys.exit(0)
    
    target = sis_graph.targets_dict[args.target]
    print(f"Target: {args.target} -> {target.required_full_list}")
    path, = target.required_full_list  # assume only one output path
    assert isinstance(path, Path)
    assert not path.hash_overwrite

    # if name == "2024-denoising-lm/error_correction_model/base-puttingItTogether(low)-nEp200/recog-ext/dlm_sum_score_results.txt":
    # path = target.required_full_list[0]
    # assert not path.hash_overwrite
    # print("Job id:", path.creator._sis_id())

    # job sis hash is the job sis_id, which is cached.
    # sis_id: via sis_hash = cls._sis_hash_static(parsed_args)

    # idea: use script to dump hash reconstruction
    # always call sis_hash_helper with settrace to detect recursive calls to sis_hash_helper
    # dump always path, object_type -> hash, starting from target, where path == "/"
    # then recursively for all dependencies, adding path as "/" + number + object_type or so when going down.

    _stack.append(_StackEntry(None, ""))
    _patched_sis_hash_helper(path)
    _stack.pop(-1)
    assert not _stack

    for report in _reports:
        print(" ".join(report))


@dataclass
class _StackEntry:
    obj: Any
    key: str
    child_count: int = 0

    def as_str(self):
        if self.key:
            return f"#{self.key} ({type(self.obj).__name__})"


_visited_objs = {}  # id -> (obj, hash)
_queue = deque()
_stack: List[_StackEntry] = []
_reports: List[List[str]] = []


def _patched_sis_hash_helper(obj: Any) -> bytes:
    if id(obj) in _visited_objs:
        obj_, hash_ = _visited_objs[id(obj)]
        assert obj is obj_
        return hash_
    if not _stack:
        return _orig_sis_hash_helper(obj)

    if isinstance(obj, Job):
        _hash_helper_func = _sis_job_hash_helper
    elif isinstance(obj, Path):
        _hash_helper_func = _sis_path_hash_helper
    else:
        _hash_helper_func = _orig_sis_hash_helper

    new_stack_entry = _StackEntry(obj=obj, key=f"(#{_stack[-1].child_count})")
    _stack[-1].child_count += 1
    _stack.append(new_stack_entry)
    path = "/ " + " / ".join(entry.key for entry in _stack[2:])
    info = [path.strip(), f"({type(obj).__name__})"]
    if isinstance(obj, Path):
        info += [repr(obj.rel_path())]
    elif isinstance(obj, Job):
        info += [obj._sis_id()]
    elif isinstance(obj, (int, float, bool)):
        info += [repr(obj)]
    elif isinstance(obj, str):
        if len(obj) > 60:
            info += [repr(obj[:60]) + "..."]
        else:
            info += [repr(obj)]
    _reports.append(info)

    # Recursive call.
    hash_ = _hash_helper_func(obj)

    _visited_objs[id(obj)] = (obj, hash_)
    new_stack_entry_ = _stack.pop(-1)
    assert new_stack_entry is new_stack_entry_
    info.extend(["->", _short_hash_from_binary(hash_)])

    return hash_


def _sis_job_hash_helper(job: Job) -> bytes:
    hash_ = job._sis_hash()

    # Manual hash computation:
    hash_manual = _sis_job_id(job).encode()
    assert hash_ == hash_manual, f"{job} sis_hash mismatch: {hash_} != {hash_manual}"
    return hash_


_visited_jobs = set()


def _sis_job_id(job: Job) -> str:
    assert isinstance(job, Job)
    if job in _visited_jobs:
        return job._sis_id()
    _visited_jobs.add(job)
    # See JobSingleton.__call__
    cls = type(job)
    sis_hash = cls._sis_hash_static(_dict_lazy_pop(job._sis_kwargs))
    module_name = cls.__module__
    recipe_prefix = gs.RECIPE_PREFIX + "."
    if module_name.startswith(recipe_prefix):
        sis_name = module_name[len(recipe_prefix) :]
    else:
        sis_name = module_name
    sis_name = os.path.join(sis_name.replace(".", os.path.sep), cls.__name__)
    sis_id = "%s.%s" % (sis_name, sis_hash)
    assert job._sis_id() == sis_id, f"{job} sis_id mismatch: {job._sis_id()} != {sis_id}"
    return sis_id


class _dict_lazy_pop(dict):
    def pop(self, k, d=None):
        if k in self:
            return super().pop(k)
        return d


def _sis_path_hash_helper(self: Path) -> bytes:
    hash_ = self._sis_hash()

    if self.hash_overwrite is None:
        creator = self.creator
        path = self.path
    else:
        creator, path = self.hash_overwrite
    if hasattr(creator, "_sis_id"):
        _patched_sis_hash_helper(creator)  # make sure we recursively visit the job
        creator = f"{creator._sis_id()}/{gs.JOB_OUTPUT}"
    hash_manual = b"(Path, " + _patched_sis_hash_helper((creator, path)) + b")"
    assert hash_ == hash_manual, f"{self} sis_hash mismatch: {hash_} != {hash_manual}"
    return hash_


def _short_hash_from_binary(
    binary: bytes, length=12, chars="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
) -> str:
    h = hashlib.sha256(binary).digest()
    h = int.from_bytes(h, byteorder="big", signed=False)
    ls = []
    for i in range(length):
        ls.append(chars[int(h % len(chars))])
        h = h // len(chars)
    return "".join(ls)


if __name__ == "__main__":
    main()
