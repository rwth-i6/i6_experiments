#!/usr/bin/env python3

"""
Dump hash traces, to check hashes...

E.g.: You have two pipelines, and expect to get the same hash (e.g. for some particular output),
but you don't, and you want to find out why.

Idea: use this script to dump hash reconstruction.
We hook sis_hash_helper (here we just patch it; could also use settrace)
to detect recursive calls to sis_hash_helper.
Dump always path, object_type -> hash, starting from target, where path == "/"
then recursively for all dependencies, adding path as "/" + number + object_type or so when going down.
Dump that to a file, then you can do a diff.

Some objects need some special handling, e.g. Job and Path,
as they use cached values (e.g. the job sis_id),
and we want to see the dependencies there as well.
"""

from __future__ import annotations
from typing import Any, Optional, Union, List, Dict, Tuple
import argparse
import os
import sys
import logging
import time
import hashlib
from inspect import isclass, isfunction
from dataclasses import dataclass
from collections import deque
from contextlib import contextmanager
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
from sisyphus.hash import _obj_type_qualname, _BasicDictTypes, _BasicSeqTypes


def main():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    arg_parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument("config_files", nargs="*")
    arg_parser.add_argument("--custom-sis-import-paths", nargs="*")
    arg_parser.add_argument("--target")
    arg_parser.add_argument(
        "--output", help="output file, default: stdout. The idea is that you can do a diff on the file."
    )
    args = arg_parser.parse_args()

    # Do that early, such that all imports of sis_hash_helper get our patched version.
    sisyphus.hash.sis_hash_helper = _patched_sis_hash_helper
    sisyphus.job_path.sis_hash_helper = _patched_sis_hash_helper

    if args.custom_sis_import_paths:
        gs.IMPORT_PATHS = args.custom_sis_import_paths

    if gs.USE_VERBOSE_TRACEBACK:
        sys.excepthook_org = sys.excepthook
        if gs.VERBOSE_TRACEBACK_TYPE == "ipython":
            from IPython.core import ultratb

            sys.excepthook = ultratb.VerboseTB()
        elif gs.VERBOSE_TRACEBACK_TYPE == "better_exchook":
            # noinspection PyPackageRequirements
            import better_exchook

            better_exchook.install()
            better_exchook.replace_traceback_format_tb()
        else:
            raise Exception("invalid VERBOSE_TRACEBACK_TYPE %r" % gs.VERBOSE_TRACEBACK_TYPE)

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

    if args.target not in sis_graph.targets_dict:
        print(f"Error: Invalid target {args.target}, valid targets are:")
        for name, target in sis_graph.targets_dict.items():
            print(f"Target: {name} -> {target.required_full_list}")
        print("Error, exiting.")
        sys.exit(1)

    target = sis_graph.targets_dict[args.target]
    print(f"Target: {args.target} -> {target.required_full_list}")
    (path,) = target.required_full_list  # assume only one output path
    assert isinstance(path, Path)
    assert not path.hash_overwrite

    _stack.append(_StackEntry(None, "", next_child_key=""))
    with _enable_patched_sis_hash_helper(True):
        _patched_sis_hash_helper(path)
    _stack.pop(-1)
    assert not _stack

    output = sys.stdout
    if args.output:
        output = open(args.output, "w")
    for report in _reports:
        print("".join(report), file=output)
    if args.output:
        output.close()
        print("Done. Wrote to", args.output)
    else:
        print("Done. If you want to dump this to a file (e.g. for a diff), use --output <file>.")


@dataclass
class _StackEntry:
    obj: Any
    key: str
    child_count: int = 0
    next_child_key: Optional[str] = None
    hash: Optional[bytes] = None


_visited_objs: Dict[int, _StackEntry] = {}  # id -> _StackEntry
_queue = deque()
_stack: List[_StackEntry] = []
_enabled: bool = False
_reports: List[List[str]] = []


@contextmanager
def _enable_patched_sis_hash_helper(enabled: bool = True):
    global _enabled
    prev = _enabled
    _enabled = enabled
    try:
        yield
    finally:
        _enabled = prev


def _patched_sis_hash_helper(obj: Any) -> bytes:
    if not _stack or not _enabled:
        return _orig_sis_hash_helper(obj)
    if id(obj) in _visited_objs:
        stack_entry = _visited_objs[id(obj)]
        assert obj is stack_entry.obj
        if stack_entry.hash is not None:
            return stack_entry.hash
        return _orig_sis_hash_helper(obj)

    if isinstance(obj, Job):
        _hash_helper_func = _sis_job_hash_helper
    elif isinstance(obj, Path):
        _hash_helper_func = _sis_path_hash_helper
    elif type(obj) in _BasicSeqTypes:
        _hash_helper_func = _sis_seq_hash_helper
    elif isinstance(obj, _BasicDictTypes):
        _hash_helper_func = _sis_dict_hash_helper
    else:
        _hash_helper_func = _orig_sis_hash_helper

    if _stack[-1].next_child_key is not None:
        key = _stack[-1].next_child_key
        _stack[-1].next_child_key = None
    else:
        key = f"(#{_stack[-1].child_count})"
    new_stack_entry = _StackEntry(obj=obj, key=key)
    _stack[-1].child_count += 1
    _stack.append(new_stack_entry)
    _visited_objs[id(obj)] = new_stack_entry
    path = "/".join(f"{entry.key}:({type(entry.obj).__name__})" for entry in _stack[1:])
    info = [path]
    if isinstance(obj, Path):
        info += ["\n = ", obj.rel_path()]
    elif isinstance(obj, Job):
        info += ["\n = ", obj._sis_id()]
    elif isinstance(obj, (int, float, bool)):
        info += ["\n = ", repr(obj)]
    elif isinstance(obj, str):
        info += ["\n = ", (repr(obj[:60]) + "...") if len(obj) > 60 else repr(obj)]
    elif isfunction(obj) or isclass(obj):
        info += ["\n = ", f"{obj.__module__}.{obj.__qualname__}"]
    _reports.append(info)

    # Recursive call.
    hash_ = _hash_helper_func(obj)

    new_stack_entry.hash = hash_
    new_stack_entry_ = _stack.pop(-1)
    assert new_stack_entry is new_stack_entry_
    info.extend(["\n -> ", _short_hash_from_binary(hash_)])

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
    sis_hash = cls._sis_hash_static(_DictLazyPop(job._sis_kwargs))
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


class _DictLazyPop(dict):
    """
    Ignores pop or __delitem__ exceptions if key not present.
    We use this because the _sis_kwargs that is stored in the job
    sometimes has already some of the keys popped
    (see JobSingleton.__call__).
    """

    def pop(self, k, d=None):
        if k in self:
            return super().pop(k)
        return d

    def __delitem__(self, key):
        if key in self:
            super().__delitem__(key)


def _sis_path_hash_helper(self: Path) -> bytes:
    with _enable_patched_sis_hash_helper(False):
        hash_ = self._sis_hash()

    if self.hash_overwrite is None:
        creator = self.creator
        path = self.path
    else:
        creator, path = self.hash_overwrite
    if hasattr(creator, "_sis_id"):
        _patched_sis_hash_helper(creator)  # make sure we recursively visit the job
        creator = f"{creator._sis_id()}/{gs.JOB_OUTPUT}"
    with _enable_patched_sis_hash_helper(False):
        hash_manual = b"(Path, " + _patched_sis_hash_helper((creator, path)) + b")"
    assert hash_ == hash_manual, f"{self} sis_hash mismatch: {hash_} != {hash_manual}"
    return hash_


def _sis_seq_hash_helper(obj: Union[list, tuple]) -> bytes:
    with _enable_patched_sis_hash_helper(False):
        hash_ = _orig_sis_hash_helper(obj)
    # See _orig_sis_hash_helper for the original implementation.
    byte_list = [_obj_type_qualname(obj)]
    assert type(obj) in _BasicSeqTypes
    for i, item in enumerate(obj):
        _stack[-1].next_child_key = f"[{i}]"
        byte_list.append(_patched_sis_hash_helper(item))
    _stack[-1].next_child_key = None
    return _sis_hash_helper_finalize(byte_list, verify_orig_hash=hash_)


def _sis_kv_tuple_hash_helper(kv: Tuple[Any, Any]) -> bytes:
    with _enable_patched_sis_hash_helper(False):
        hash_ = _orig_sis_hash_helper(kv)
    key, value = kv
    # See _orig_sis_hash_helper for the original implementation.
    byte_list = [tuple.__qualname__.encode()]
    # We assume that the key hash is not of interest, so don't include it in the report.
    with _enable_patched_sis_hash_helper(False):
        byte_list.append(_orig_sis_hash_helper(key))
    _stack[-1].next_child_key = f"[{key!r}]"
    byte_list.append(_patched_sis_hash_helper(value))
    _stack[-1].next_child_key = None
    return _sis_hash_helper_finalize(byte_list, verify_orig_hash=hash_)


def _sis_dict_hash_helper(obj: dict) -> bytes:
    with _enable_patched_sis_hash_helper(False):
        hash_ = _orig_sis_hash_helper(obj)
    # See _orig_sis_hash_helper for the original implementation.
    byte_list = [_obj_type_qualname(obj)]
    assert isinstance(obj, _BasicDictTypes)
    byte_list += sorted(map(_sis_kv_tuple_hash_helper, obj.items()))
    return _sis_hash_helper_finalize(byte_list, verify_orig_hash=hash_)


def _sis_hash_helper_finalize(byte_list: List[bytes], *, verify_orig_hash: bytes) -> bytes:
    # Taken from _orig_sis_hash_helper.
    byte_str = b"(" + b", ".join(byte_list) + b")"
    if len(byte_str) > 4096:
        # hash long outputs to avoid arbitrary long return values. 4096 is just
        # picked because it looked good and not optimized,
        # it's most likely not that important.
        byte_str = hashlib.sha256(byte_str).digest()
    assert byte_str == verify_orig_hash, f"hash mismatch: {byte_str} != {verify_orig_hash}"
    return byte_str


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
