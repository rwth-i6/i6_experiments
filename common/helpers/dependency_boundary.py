"""
Dependency boundary -- Sisyphus graph breakpoint

WARNING: Code is work-in-progress, not stable!

You have a very huge pipeline with thousands of jobs,
but for your experiments, all the first 90% of the jobs never change,
and the job outputs are also changed among users.
Example: preprocessed data, features, HMM alignments, CART, etc.

So far, many people have done such graph breakpoint by hand,
i.e. calculated those things, then stored the files somewhere,
and then have separate setups which directly use those files as hardcoded inputs.

Here we want to have the possibility to easily switch between fixed inputs vs the full graph,
while keeping the same hash.

For some discussion on the specific design decisions here, see:
https://github.com/rwth-i6/i6_experiments/issues/78
"""

from typing import Any, Optional, TypeVar, Callable
from sisyphus.hash import short_hash
from sisyphus.tools import extract_paths
from i6_experiments.common.utils.dump_py_code import PythonCodeDumper
from i6_experiments.common.utils.diff import collect_diffs
import os
import sys
import textwrap
import importlib.util


T = TypeVar("T")


# noinspection PyShadowingBuiltins
def dependency_boundary(func: Callable[[], T], *, hash: Optional[str]) -> T:
    """
    It basically returns func(), or some object which has the same hash.

    :param func: Function which would create the whole graph to get some outputs.
        The return value could be anything, but usually would be a dict, namedtuple, dataclass or sth similar.
        This function is not called when we actually enable the dependency boundary.
    :param hash: sisyphus.hash.short_hash(func()), or None if you do not know this value yet.
        This value is used to verify the hash of the object.
        For new code when the hash is not known yet, you would pass None here, and it will print the hash on stdout.
    :return: func(), or object with same hash
    """
    hash_via_user = hash
    obj_via_cache = None
    hash_via_cache = None
    cached_paths_available = False

    cache_fn = get_cache_filename_for_func(func)
    if os.path.exists(cache_fn):
        try:
            obj_via_cache = load_obj_from_cache_file(cache_fn)
            hash_via_cache = short_hash(obj_via_cache)
            cached_paths_available = _paths_available(func, obj_via_cache)
        except Exception as exc:
            print(
                f"Dependency boundary for {func.__qualname__}:"
                f" error, exception {type(exc).__name__} {str(exc)!r} while loading the cache,"
                " will ignore the cache"
            )
            obj_via_cache = None
            hash_via_cache = None
            cached_paths_available = False

    if hash_via_user and hash_via_cache and hash_via_user == hash_via_cache and cached_paths_available:
        print(f"Dependency boundary for {func.__qualname__}: using cached object with hash {hash_via_user}")
        return obj_via_cache

    # Either user hash invalid, or cached hash invalid, or not all paths are available, or user hash not defined.
    # In any case, need to check actual function.
    obj_via_func = func()
    assert obj_via_func is not None  # unexpected
    hash_via_func = short_hash(obj_via_func)
    print(f"Dependency boundary for {func.__qualname__}: hash of original object = {hash_via_func}")

    if not hash_via_user:
        print(f"Dependency boundary for {func.__qualname__}: you should add the hash to the dependency_boundary call")

    if hash_via_user and hash_via_user != hash_via_func:
        print(
            f"Dependency boundary for {func.__qualname__}: error, given hash ({hash_via_user}) is invalid,"
            " please fix the hash given to the dependency_boundary call"
        )

    if hash_via_cache and hash_via_cache != hash_via_func:
        print(
            f"Dependency boundary for {func.__qualname__}: error, cached hash {hash_via_cache} is invalid,"
            " will recreate the cache"
        )
        hash_via_cache = None

    if not hash_via_cache:
        print(f"Dependency boundary for {func.__qualname__}: create or update cache {cache_fn!r}")
        save_obj_to_cache_file(obj_via_func, cache_filename=cache_fn)
        # Do some check that the dumped object has the same hash.
        obj_via_cache = load_obj_from_cache_file(cache_fn)
        hash_via_cache = short_hash(obj_via_cache)
        if hash_via_func != hash_via_cache:
            print(
                f"Dependency boundary for {func.__qualname__}: error, dumping logic stores inconsistent object,"
                f" dumped object hash {hash_via_cache}"
            )
            print("Differences:")
            diffs = collect_diffs("obj", obj_via_func, obj_via_cache)
            if diffs:
                for diff in diffs:
                    print(diff)
            else:
                print("(No differences detected?)")
            if hash_via_cache == hash_via_user:
                print(
                    f"Dependency boundary for {func.__qualname__}:"
                    f" error, user provided hash is matching to wrong cache!"
                )
                os.remove(cache_fn)  # make sure it is not used

    return obj_via_func


def get_cache_filename_for_func(func: Callable[[], T]) -> str:
    """
    :return: filename of autogenerated Python file
    """
    mod = sys.modules[getattr(func, "__module__")]
    mod_dir = os.path.dirname(os.path.abspath(mod.__file__))
    return f"{mod_dir}/_dependency_boundary_autogenerated_cache.{mod.__name__.split('.')[-1]}.{func.__qualname__}.py"


def save_obj_to_cache_file(obj: Any, *, cache_filename: str) -> None:
    """
    Save object.
    """
    with open(cache_filename, "w") as cache_f:
        cache_f.write(
            textwrap.dedent(
                """\
                \"\"\"
                Auto-generated code via dependency_boundary.
                Do not modify by hand!
                \"\"\"
        
                """
            )
        )
        PythonCodeDumper(file=cache_f, use_fake_jobs=True).dump(obj, lhs="obj")


def load_obj_from_cache_file(cache_filename: str) -> Any:
    """
    :return: previously saved object
    """
    cache_fn_mod_name = cache_filename.lstrip("/").replace("/", ".")
    spec = importlib.util.spec_from_file_location(cache_fn_mod_name, cache_filename)
    cache_fn_mod = importlib.util.module_from_spec(spec)
    sys.modules[cache_fn_mod_name] = cache_fn_mod
    spec.loader.exec_module(cache_fn_mod)
    obj = cache_fn_mod.obj
    assert obj is not None
    return obj


def _paths_available(func, obj: Any) -> bool:
    """
    :return: True if all paths in obj are available
    """
    paths = extract_paths(obj)
    for path in paths:
        if not path.available():
            print(f"Dependency boundary for {func.__qualname__}: path {path} in cached object not available")
            # No need to print this for all paths, just the first one is enough.
            return False
    return True
