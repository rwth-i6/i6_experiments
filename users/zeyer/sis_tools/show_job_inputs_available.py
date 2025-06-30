"""
For some job, show what inputs are available
"""

from typing import Any, Dict, Tuple, List
import enum

from sisyphus import Job, Path
from sisyphus.block import Block
from sisyphus.hash import get_object_state


def show_job_inputs_available(job: Job, *, assert_all_available: bool = False):
    # noinspection PyProtectedMember
    job_kwargs = job._sis_kwargs
    sis_inputs_to_key = extract_paths_with_key(job_kwargs)
    # noinspection PyProtectedMember
    sis_inputs_set = job._sis_inputs
    assert sis_inputs_set == set(sis_inputs_to_key.keys()), (sis_inputs_set, set(sis_inputs_to_key.keys()))

    not_available = []
    for path, key in sis_inputs_to_key.items():
        available = path.available()
        print(f"Input {key}: {path}: available: {available}")
        if not available:
            not_available.append((key, path))
    print("All inputs available:", not not_available)
    if assert_all_available:
        assert not not_available, f"Some inputs not available: {not_available}"


def extract_paths_with_key(args: Any) -> Dict[Path, Tuple[Any, ...]]:
    """
    Extract all :class:`Path` objects from the given arguments.
    Copy of Sisyphus tools.extract_paths but also provides the key of the path.
    """
    out = {}
    visited_obj_ids = {}  # id -> obj  # keep ref to obj alive, to avoid having same id for different objs
    queue: List[Tuple[Tuple[Any, ...], Any]] = [((), args)]  # list of (key, obj)
    while queue:
        key, obj = queue.pop()
        if id(obj) in visited_obj_ids:
            continue
        visited_obj_ids[id(obj)] = obj
        if obj is None:
            continue
        if isinstance(obj, (bool, int, float, complex, str)):
            continue
        if isinstance(obj, Block) or isinstance(obj, enum.Enum):
            continue
        if hasattr(obj, "_sis_path") and obj._sis_path is True and not type(obj) is type:
            out.setdefault(obj, key)
        elif isinstance(obj, (list, tuple, set, frozenset)):
            if isinstance(obj, (set, frozenset)):
                obj = sorted(obj)
            for i, v in enumerate(obj):
                queue.append((key + (i,), v))
        elif isinstance(obj, dict):
            for k, v in obj.items():
                if not type(k) == str or not k.startswith("_sis_"):
                    queue.append((key + (k,), v))
        else:
            queue.append((key, get_object_state(obj)))
    return out
