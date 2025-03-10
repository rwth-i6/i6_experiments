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
- The dependencies are the same (potentially mapping the hashes).
- The Python stack trace is the same.
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
from typing import Union, Callable, TypeVar, List, Tuple, Dict, Set
from functools import reduce
from types import FunctionType, CodeType, FrameType
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


from sisyphus import Path
from sisyphus.job import JobSingleton, Job
import sisyphus.global_settings as gs
from sisyphus import tools
from i6_experiments.users.zeyer.recog import GetBestRecogTrainExp


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

    def _wrapped_job_type_call(*args, **kwargs):
        job = orig_job_type_call(*args, **kwargs)
        assert isinstance(job, Job)
        _created_jobs_broken.append(job)
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

    def _wrapped_job_type_call(*args, **kwargs):
        job = orig_job_type_call(*args, **kwargs)
        assert isinstance(job, Job)
        _created_jobs_correct.append(job)
        return job

    print("Collect jobs with correct hashing...")
    JobSingleton.__call__ = _wrapped_job_type_call
    start = time.time()
    exp_func()
    print(f"Elapsed time: {time.time() - start:.3f} sec")
    print(f"Collected {len(_created_jobs_correct)} jobs with correct hashing.")

    print("Matching jobs...")
    map_correct_to_broken: Dict[Job, Job] = {}
    job_broken_idx = 0
    for job_correct in _created_jobs_correct:
        # We assume that all correct jobs have a matching broken job.
        job_broken_idx, job_broken = _find_matching_job(
            job_correct=job_correct,
            jobs_broken=_created_jobs_broken,
            job_broken_start_idx=job_broken_idx,
            map_correct_to_broken=map_correct_to_broken,
        )
        if job_correct.job_id() == job_broken.job_id():
            continue  # not interesting
        print(f"Matched correct job {job_correct} to broken job {job_broken}")
        job_broken_idx += 1
        map_correct_to_broken[job_correct] = job_broken

    # All matched, no error, so we are good. Now proposing new symlinks.
    for job_correct, job_broken in map_correct_to_broken.items():
        print(f"Job correct {job_correct} -> broken {job_broken}")
        # noinspection PyProtectedMember
        job_correct_path, job_broken_path = job_correct._sis_path(abspath=True), job_broken._sis_path(abspath=True)
        if not os.path.exists(job_broken_path):
            print(f"  Job broken path does not exist: {job_broken_path}")
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
            print(f"  Job correct path has wrong symlink -> {job_correct_path_existing_symlink}, removing")
            if not dry_run:
                os.remove(job_correct_path)  # recreate it
        elif stat_:
            print(f"  Job correct path already exists: {stat.filemode(stat_.st_mode)}")
            continue
        print(f"  Create symlink {job_correct_path} -> {job_correct_path_target_symlink}")
        if not dry_run:
            assert not os.path.exists(job_correct_path) and os.path.exists(job_correct_path_target_symlink)
            os.symlink(job_correct_path_target_symlink, job_correct_path)


def _find_matching_job(
    *, job_correct: Job, jobs_broken: List[Job], job_broken_start_idx: int, map_correct_to_broken: Dict[Job, Job]
) -> Tuple[int, Job]:
    assert job_broken_start_idx < len(jobs_broken)
    job_broken_idx = job_broken_start_idx
    while job_broken_idx < len(jobs_broken):
        job_broken = jobs_broken[job_broken_idx]
        is_matching, is_non_matching_reason = _is_matching_job(
            job_broken=job_broken, job_correct=job_correct, map_correct_to_broken=map_correct_to_broken
        )
        if is_matching:
            return job_broken_idx, job_broken
        job_broken_idx += 1

    # We assume that all correct jobs have a matching broken job, thus getting here is an error.
    # noinspection PyProtectedMember
    stacktraces = job_correct._sis_stacktrace
    stacktrace_str = "".join(traceback.format_list(stacktraces[0])) if stacktraces else "<no stacktrace>"
    jobs_broken_ = []
    for i in range(max(0, job_broken_start_idx - 3), min(job_broken_start_idx + 4, len(jobs_broken))):
        s = " -> " if job_broken_start_idx == i else "    "
        _, non_match_reason = _is_matching_job(
            job_broken=jobs_broken[i], job_correct=job_correct, map_correct_to_broken=map_correct_to_broken
        )
        jobs_broken_.append(f"{s}{jobs_broken[i]} ({non_match_reason})")
    raise Exception(
        f"Could not find matching broken job for correct job: {job_correct}\n"
        f"Broken job candidates:\n{'\n'.join(jobs_broken_)}\n"
        f"Correct job create traceback:\n{stacktrace_str}"
    )


def _is_matching_job(*, job_broken: Job, job_correct: Job, map_correct_to_broken: Dict[Job, Job]) -> Tuple[bool, str]:
    if type(job_broken) is not type(job_correct):
        return False, "Different type"
    if job_broken.get_aliases() != job_correct.get_aliases():
        return False, (
            f"Different aliases: Broken job {job_broken.get_aliases()} vs correct job {job_correct.get_aliases()}"
        )
    if isinstance(job_broken, GetBestRecogTrainExp):
        # Special handling for GetBestRecogTrainExp as we dynamically add more inputs,
        # and the new job with correct hash might not know about all inputs yet.
        # (Actually, we might use this logic here just for all jobs?)
        # noinspection PyProtectedMember
        job_broken_kwargs, job_correct_kwargs = job_broken._sis_kwargs, job_correct._sis_kwargs
        job_broken_inputs = tools.extract_paths(job_broken_kwargs)
        job_correct_inputs = tools.extract_paths(job_correct_kwargs)
    else:
        # noinspection PyProtectedMember
        job_broken_inputs, job_correct_inputs = job_broken._sis_inputs, job_correct._sis_inputs
    job_broken_inputs: Set[Path]
    job_correct_inputs: Set[Path]
    job_correct_inputs = set(_map_correct_job_path_to_broken(p, map_correct_to_broken) for p in job_correct_inputs)
    job_broken_inputs_ = set(p.get_path() for p in job_broken_inputs)
    job_correct_inputs_ = set(p.get_path() for p in job_correct_inputs)
    # Note: We allow that the broken job has some fewer inputs, e.g. when some Path was converted to str.
    if not job_broken_inputs_.issubset(job_correct_inputs_):
        return False, (
            f"Different inputs. Broken deps that are not in correct deps: "
            f"{sorted(p for p in job_broken_inputs_ if p not in job_correct_inputs_)}"
        )
    return True, "<Matching>"


def _map_correct_job_path_to_broken(path: Path, map_correct_to_broken: Dict[Job, Job]) -> Path:
    if path.creator is None:
        return path
    if path.creator in map_correct_to_broken:
        broken_job = map_correct_to_broken[path.creator]
        return Path(path.path, creator=broken_job)
    return path


def _get_func_code(func: Union[FunctionType, Callable]) -> CodeType:
    while getattr(func, "__wrapped__", None) is not None:
        func = func.__wrapped__
    return func.__code__


def _main():
    """When called directly"""
    import argparse
    import importlib

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config_file", help="Sisyphus config file (entry point with `py` func)")
    arg_parser.add_argument("--fix-func", help="Func to enable fixed behavior")
    arg_parser.add_argument("--no-dry-run", action="store_true", help="Actually create the symlinks")
    args = arg_parser.parse_args()

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
