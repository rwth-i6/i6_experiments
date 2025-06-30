"""
Situation:
- A couple of experiments were run.
- There was a bug in the pipeline which caused some of the hashes to be incorrect
  (including all further jobs depending on the incorrect hash).
- You want to fix this.

Solution:
- Make symlinks for all jobs from the correct hash to the incorrect hash.

We can do that automatically.
This is what we do here.
We use Python ``sys.settrace`` to intercept :class:`sisyphus.job.JobSingleton.__call__`.
We are given some Python code (function) which generates all the jobs.
We run this code and intercept all jobs.
First we run it with the broken hashing.
Then we run it again with the correct hashing.
We match the jobs.
We can have multiple sanity checks to make sure we match the right jobs:

- All the created jobs are exactly of the same type in the same order (only hash might differ)
  (note: due to caching, we might see a few less...).
- The kwargs are the same (potentially mapping the hashes) (this is more than just the inputs/deps).
- The Python stack trace is the same.
  (But careful: there might be same jobs with different stack traces...)
- Alias / output names are the same.

Usage example::

    python recipe/i6_experiments/users/zeyer/sis_tools/hash_fix.py \
        --fix-func i6_experiments.users.zeyer.sis_tools.instanciate_delayed.use_instanciate_delayed_copy_instead_of_inplace \
        recipe/i6_experiments/.../...exp_sis_config....py

"""

from __future__ import annotations
import sys
import os
import stat
import time
from typing import Optional, Union, Callable, TypeVar, List, Tuple, Dict, Set
from functools import reduce
from types import FunctionType, CodeType
import traceback


_my_dir = os.path.dirname(__file__)
_base_dir = reduce(lambda p, _: os.path.dirname(p), range(4), _my_dir)
_sis_dir = os.path.dirname(_base_dir) + "/tools/sisyphus"

T = TypeVar("T")


def _setup():
    # In case the user started this script directly.
    if not globals().get("__package__"):
        globals()["__package__"] = "i6_experiments.users.zeyer.sis_tools"
        if _base_dir not in sys.path:
            sys.path.append(_base_dir)
        if _sis_dir not in sys.path:
            sys.path.append(_sis_dir)


_setup()


from sisyphus.job_path import Path, AbstractPath
from sisyphus.job import JobSingleton, Job
from sisyphus.hash import get_object_state
import sisyphus.global_settings as gs
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.training import PtCheckpoint
from returnn.util.basic import obj_diff_list


def hash_fix(
    exp_func: Union[FunctionType, Callable[[], None]],
    *,
    fix_func: Union[FunctionType, Callable[[], None]],
    dry_run: bool,
):
    """
    See module docstring.

    :param exp_func: The experiment function which generates all the jobs.
    :param fix_func: The function which should enable the correct hashing.
    :param dry_run: If True, just print what would be done. If False, actually create the symlinks.
    """
    print(
        f"Hash fix,"
        f" exp_func={exp_func.__module__}.{exp_func.__qualname__},"
        f" fix_func={fix_func.__module__}.{fix_func.__qualname__},"
        f" dry_run{dry_run}"
    )

    # Useful for debugging.
    gs.JOB_ADD_STACKTRACE_WITH_DEPTH = 100

    # Run with the broken hashing first.

    # Note: Previously, we used sys.settrace to intercept all job creations.
    # (Also see returnn.util.debug.PyTracer for reference on that.)
    # However, that turned out to be too slow,
    # as all the hashing and recursive iteration through nested state dicts
    # consumed quite some Python computation.
    orig_job_type_call = JobSingleton.__call__
    _job_create_code = _get_func_code(JobSingleton.__call__)
    _created_jobs_broken: List[Job] = []
    _created_jobs_broken_visited: Set[Job] = set()

    def _wrapped_job_type_call(*args, **kwargs):
        job = orig_job_type_call(*args, **kwargs)
        assert isinstance(job, Job)
        if job in _created_jobs_broken_visited:
            return job
        _created_jobs_broken_visited.add(job)
        _created_jobs_broken.append(job)
        _maybe_update_job(job)
        return job

    # Trace all jobs.
    print("Collect jobs with broken hashing...")
    JobSingleton.__call__ = _wrapped_job_type_call
    start = time.time()
    exp_func()
    print(f"Elapsed time: {time.time() - start:.3f} sec")
    print(f"Collected {len(_created_jobs_broken)} jobs with broken hashing.")

    # Enable the correct hashing.
    print(f"Enable correct behavior via fix_func {fix_func.__module__}.{fix_func.__qualname__}.")
    fix_func()

    # Trace all jobs again.
    _created_jobs_correct: List[Job] = []
    _created_jobs_correct_visited: Set[Job] = set()

    def _wrapped_job_type_call(*args, **kwargs):
        job = orig_job_type_call(*args, **kwargs)
        assert isinstance(job, Job)
        if job in _created_jobs_correct_visited:
            return job
        _created_jobs_correct_visited.add(job)
        _created_jobs_correct.append(job)
        _maybe_update_job(job)
        return job

    print("Collect jobs with correct hashing...")
    JobSingleton.__call__ = _wrapped_job_type_call
    start = time.time()
    exp_func()
    JobSingleton.__call__ = orig_job_type_call
    print(f"Elapsed time: {time.time() - start:.3f} sec")
    print(f"Collected {len(_created_jobs_correct)} jobs with correct hashing.")

    print("Matching jobs...")
    # For every correct job, there can be one or multiple broken jobs,
    # but for every broken job, there is exactly one correct job (maybe itself if it is already correct),
    # or maybe none if it is covered by some caching mechanism.
    map_correct_to_broken: Dict[Job, Job] = {}  # first broken one we found, if no correct
    map_broken_to_correct: Dict[Job, Job] = {}
    job_broken_idx = 0
    for job_correct_idx, job_correct in enumerate(_created_jobs_correct):
        # There can be multiple broken jobs for one correct job,
        # thus the map_correct_to_broken mapping is not unique,
        # thus we use map_from_other=map_broken_to_correct.
        job_broken_idx, jobs_broken = _find_matching_jobs(
            job=job_correct,
            job_name="correct",
            jobs_other=_created_jobs_broken,
            job_other_name="broken",
            job_other_start_idx=job_broken_idx,
            map_from_other=map_broken_to_correct,
        )
        for job_broken in jobs_broken:
            if job_broken in _created_jobs_correct_visited:  # Same job also created with correct hashing.
                if job_broken == job_correct:
                    continue  # no need to add this to map_broken_to_correct
                # This is weird... should be unique.
                # noinspection PyProtectedMember
                raise Exception(
                    f"Job broken {job_broken} is a correct job by itself,"
                    f" but it also matches to the correct job {job_correct}.\n"
                    f"Correct job args: {job_correct._sis_kwargs}\n"
                    f"Broken job args: {job_broken._sis_kwargs}"
                )
            assert job_broken != job_correct
            assert job_broken not in map_broken_to_correct
            assert job_broken not in _created_jobs_correct_visited
            map_broken_to_correct[job_broken] = job_correct
        if job_correct in _created_jobs_broken_visited:
            # It should have matched above.
            # No need to handle this further, no need to add it to map_correct_to_broken.
            continue
        assert job_correct not in map_broken_to_correct
        job_broken = jobs_broken[0]  # take first one
        print(f"Matched correct job {job_correct_idx} {job_correct} to broken job {job_broken_idx} {job_broken}")
        job_broken_idx += 1
        map_correct_to_broken[job_correct] = job_broken

    # All matched, no error, so we are good. Now proposing new symlinks.
    for job_correct, job_broken in map_correct_to_broken.items():
        print(f"Job correct {job_correct} -> broken {job_broken}")
        # noinspection PyProtectedMember
        job_correct_path, job_broken_path = job_correct._sis_path(abspath=True), job_broken._sis_path(abspath=True)
        try:
            existing_broken_content = os.listdir(job_broken_path)
        except FileNotFoundError:
            existing_broken_content = None
        if not existing_broken_content:
            print(f"  Job broken path {job_broken_path} has no content {existing_broken_content}, ignoring")
            continue
        try:
            stat_ = os.stat(job_correct_path, follow_symlinks=False)
        except FileNotFoundError:
            stat_ = None
        job_correct_path_existing_symlink = (
            os.readlink(job_correct_path) if stat_ and stat.S_ISLNK(stat_.st_mode) else None
        )
        job_correct_path_target_symlink = job_broken_path
        if job_correct_path_existing_symlink:
            if job_correct_path_target_symlink == job_correct_path_existing_symlink:
                print(f"  Job correct path already has correct symlink")
                continue
            try:
                existing_content = os.listdir(job_correct_path_existing_symlink)
            except FileNotFoundError:
                existing_content = None
            if existing_content:
                print(
                    f"  Job correct path has different symlink with content"
                    f" -> {job_correct_path_existing_symlink}, ignoring"
                )
                continue
            print(
                f"  Job correct path has wrong symlink -> {job_correct_path_existing_symlink}"
                f" with content {existing_content}, removing"
            )
            if not dry_run:
                os.remove(job_correct_path)  # recreate it
        elif stat_:
            print(f"  Job correct path already exists: {stat.filemode(stat_.st_mode)}")
            continue
        print(f"  Create symlink {job_correct_path} -> {job_correct_path_target_symlink}")
        if not dry_run:
            assert not os.path.exists(job_correct_path) and os.path.exists(job_correct_path_target_symlink)
            os.symlink(job_correct_path_target_symlink, job_correct_path)


def _find_matching_jobs(
    job: Job,
    *,
    jobs_other: List[Job],
    job_other_start_idx: int,
    map_to_other: Optional[Dict[Job, Job]] = None,
    map_from_other: Optional[Dict[Job, Job]] = None,
    job_name: str,
    job_other_name: str,
) -> Tuple[int, List[Job]]:
    assert job_other_start_idx < len(jobs_other)
    job_other_idx = job_other_start_idx
    wrapped_around = False
    match_kwargs = dict(
        job=job,
        job_name=job_name,
        job_other_name=job_other_name,
        map_to_other=map_to_other,
        map_from_other=map_from_other,
    )
    matching_jobs = []
    matching_job_indices = []
    while True:
        if job_other_idx >= len(jobs_other):
            assert not wrapped_around
            job_other_idx = 0
            wrapped_around = True
        if wrapped_around and job_other_idx == job_other_start_idx:
            break
        job_other = jobs_other[job_other_idx]
        is_matching, _ = _is_matching_job(job_other=job_other, **match_kwargs)
        if is_matching:
            matching_jobs.append(job_other)
            matching_job_indices.append(job_other_idx)
        job_other_idx += 1

    if matching_jobs:
        return matching_job_indices[0], matching_jobs

    # We assume that all correct jobs have a matching broken job, thus getting here is an error.
    # Collect some information for debugging why it is not matching
    # (maybe some bug in the matching logic).
    stacktrace_strs = []
    # noinspection PyProtectedMember
    stacktrace = job._sis_stacktrace[0]
    stacktrace_strs.append(f"{job_name.capitalize()} job create traceback:\n")
    stacktrace_strs.extend(traceback.format_list(stacktrace))
    job_other_idx = job_other_start_idx
    job_other = jobs_other[job_other_idx]
    if type(job_other) is type(job):
        # noinspection PyProtectedMember
        stacktrace = job_other._sis_stacktrace[0]
        stacktrace_strs.append(f"Potential matching {job_other_name} job {job_other_idx} create traceback:\n")
        stacktrace_strs.extend(traceback.format_list(stacktrace))

    jobs_broken_ = []
    for i in range(max(0, job_other_start_idx - 3), min(job_other_start_idx + 4, len(jobs_other))):
        s = " -> " if job_other_start_idx == i else "    "
        _, non_match_reason = _is_matching_job(job_other=jobs_other[i], **match_kwargs)
        jobs_broken_.append(f"{s}{i} {jobs_other[i]} ({non_match_reason})")

    matching_jobs_other_ignoring_in = []
    for job_other in jobs_other:
        is_matching, _ = _is_matching_job(job_other=job_other, **match_kwargs, check_inputs=False)
        if is_matching:
            _, non_match_reason = _is_matching_job(job_other=job_other, **match_kwargs)
            matching_jobs_other_ignoring_in.append(f"  {job_other} ({non_match_reason})")

    raise Exception(
        f"Could not find matching {job_other_name} job for {job_name} job: {job}\n"
        f"{job_other_name.capitalize()} job candidates:\n{'\n'.join(jobs_broken_)}\n"
        f"Matching {job_other_name} jobs ignoring inputs:\n{'\n'.join(matching_jobs_other_ignoring_in) or '  <none>'}\n"
        f"{''.join(stacktrace_strs)}"
    )


def _is_matching_job(
    *,
    job: Job,
    job_other: Job,
    map_to_other: Optional[Dict[Job, Job]] = None,
    map_from_other: Optional[Dict[Job, Job]] = None,
    check_inputs: bool = True,
    job_name: str,
    job_other_name: str,
) -> Tuple[bool, str]:
    if type(job_other) is not type(job):
        return False, "Different type"
    if job_other.job_id() == job.job_id():  # fast path
        return True, "<Matching>"

    job_other_aliases: Set[str] = job_other.get_aliases() or set()
    job_aliases: Set[str] = job.get_aliases() or set()
    if job_other_aliases:
        # Allow that the broken job has some fewer aliases.
        # Due to broken hashing, there might be multiple jobs with different hashes which should actually be the same.
        if not job_other_aliases.issubset(job_aliases):
            return False, (
                f"Different aliases: {job_other_name} job {job_other_aliases} vs {job_name} job {job_aliases}"
            )
    elif job_aliases:
        return False, f"Different aliases: {job_other_name} job {job_other_aliases} vs {job_name} job {job_aliases}"

    if check_inputs:
        assert not (map_to_other and map_from_other)
        # noinspection PyProtectedMember
        job_other_kwargs = _map_kwargs_to_other(job_other._sis_kwargs, map_to_other=map_from_other)
        # noinspection PyProtectedMember
        job_kwargs = _map_kwargs_to_other(job._sis_kwargs, map_to_other=map_to_other)
        if job_kwargs != job_other_kwargs:
            diff_ls = obj_diff_list(job_other_kwargs, job_kwargs)
            if diff_ls[:1] == ["dict diff:"]:
                diff_ls = diff_ls[1:]  # remove this prefix
            diff_s = " ".join(diff_ls[:1] + ([f"({len(diff_ls) - 1} more)"] if len(diff_ls) > 1 else []))
            return False, f"Different kwargs. {job_other_name.capitalize()} vs {job_name}: {diff_s or '<empty?>'}"

    # noinspection PyProtectedMember
    job_stacktraces, job_broken_stacktraces = job._sis_stacktrace, job_other._sis_stacktrace
    equal, not_equal_reason = _are_stacktraces_equal(job_broken_stacktraces, job_stacktraces)
    if not equal:
        return False, f"Different stacktrace: {not_equal_reason}"
    return True, "<Matching>"


def _map_job_path_to_other(path: AbstractPath, map_to_other: Optional[Dict[Job, Job]]) -> AbstractPath:
    if path.creator is None:
        return path
    if map_to_other and path.creator in map_to_other:
        job_other = map_to_other[path.creator]
        return Path(path.path, creator=job_other)
    return path


def _map_kwargs_to_other(obj: T, *, map_to_other: Optional[Dict[Job, Job]]) -> T:
    if obj is None or isinstance(obj, (str, int, float, bool, str, complex)):
        return obj
    if isinstance(obj, AbstractPath):
        obj = _map_job_path_to_other(obj, map_to_other=map_to_other)
        return obj.get_path()
    if isinstance(obj, PtCheckpoint):
        return _map_kwargs_to_other(obj.path, map_to_other=map_to_other)
    if isinstance(obj, Job):
        raise TypeError(f"Unexpected Job object in kwargs: {obj}")
    if isinstance(obj, dict):
        return {k: _map_kwargs_to_other(v, map_to_other=map_to_other) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return type(obj)(_map_kwargs_to_other(v, map_to_other=map_to_other) for v in obj)
    if isinstance(obj, ReturnnConfig):
        return _map_kwargs_to_other(obj.config, map_to_other=map_to_other)
    return _map_kwargs_to_other(get_object_state(obj), map_to_other=map_to_other)


def _get_func_code(func: Union[FunctionType, Callable]) -> CodeType:
    while getattr(func, "__wrapped__", None) is not None:
        func = func.__wrapped__
    return func.__code__


def _are_stacktraces_equal(
    stacktraces1: List[List[traceback.FrameSummary]], stacktraces2: List[List[traceback.FrameSummary]]
) -> Tuple[bool, str]:
    assert stacktraces1 and stacktraces2
    stacktraces1_ = set(_stacktrace_as_key(stacktrace) for stacktrace in stacktraces1)
    stacktraces2_ = set(_stacktrace_as_key(stacktrace) for stacktrace in stacktraces2)
    if stacktraces1_.intersection(stacktraces2_):
        return True, "<Matching>"

    common: List[Union[traceback.FrameSummary, str]] = []
    for frame1, frame2 in zip(stacktraces1[0], stacktraces2[0]):
        if frame1.filename != frame2.filename:
            return False, f"{common}, frame differs: {frame1} vs {frame2}"
        if frame1.filename == __file__:
            common.append(f"({os.path.basename(__file__)})")
            continue
        if frame1.lineno != frame2.lineno:
            return False, f"{common}, frame differs: {frame1} vs {frame2}"
        common.append(frame1)
    if len(stacktraces1[0]) != len(stacktraces2[0]):
        return False, f"{common}, different length"
    return False, f"{common}, unknown diff?"


def _stacktrace_as_key(stacktrace: List[traceback.FrameSummary]):
    res: List[Tuple[str, Optional[int]]] = []
    for frame in stacktrace:
        res.append((frame.filename, frame.lineno if frame.filename != __file__ else None))
    return tuple(res)


def _maybe_update_job(job: Job):
    # Could also run job._sis_runnable(), but this here is doing a bit less work, and thus faster.
    # noinspection PyProtectedMember
    if job._sis_update_possible():
        # noinspection PyProtectedMember
        job._sis_update_inputs()


def _main():
    """When called directly"""
    import argparse
    import importlib
    from returnn.util import better_exchook

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_file", help="Sisyphus config file (entry point with `py` func)")
    arg_parser.add_argument("--fix-func", help="Func to enable fixed behavior")
    arg_parser.add_argument("--no-dry-run", action="store_true", help="Actually create the symlinks")
    args = arg_parser.parse_args()
    better_exchook.install()

    # like sisyphus.loader.ConfigManager.load_config_file
    filename = args.config_file
    # maybe remove import path prefix such as "recipe/"
    for load_path in gs.IMPORT_PATHS:
        if load_path.endswith("/") and filename.startswith(load_path):
            filename = filename[len(load_path) :]
            break
    filename = filename.replace(os.path.sep, ".")  # allows to use tab completion for file selection
    assert all(part.isidentifier() for part in filename.split(".")), "Config name is invalid: %s" % filename
    module_name, function_name = filename.rsplit(".", 1)
    config = importlib.import_module(module_name)
    exp_func = getattr(config, function_name)  # the `py` func in the Sisyphus config file
    assert callable(exp_func)

    module_name, function_name = args.fix_func.rsplit(".", 1)
    mod = importlib.import_module(module_name)
    fix_func = getattr(mod, function_name)
    assert callable(fix_func)

    hash_fix(exp_func, fix_func=fix_func, dry_run=not args.no_dry_run)


if __name__ == "__main__":
    _main()
