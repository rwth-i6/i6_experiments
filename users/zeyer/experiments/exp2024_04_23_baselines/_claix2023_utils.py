"""
Common helpers for the RWTH ITC (CLAIX 2023) cluster.

Also see feature request on Sisyphus hook for a maybe better way to do this:
https://github.com/rwth-i6/sisyphus/issues/269
"""

import os
import re
from sisyphus import tk

_user_name = os.environ.get("USER", None)
if not _user_name:
    if os.environ.get("HOME", None):
        _user_name = os.path.basename(os.environ["HOME"])
assert _user_name, f"cannot obtain user name from environment: {os.environ}"

target_dir = f"/hpcwork/p0023999/{_user_name}/setups/exp2024_04_23_baselines"
alias_pattern = re.compile("(.*/|^)(aed|ctc|lm)/.*")


def _is_matching_job(job: tk.Job, *, depth: int = 5) -> bool:
    if getattr(job, "_claix_is_matching_job", None) is not None:
        return getattr(job, "_claix_is_matching_job")
    if any(alias_pattern.match(alias) for alias in (job.get_aliases() or [])):
        job._claix_is_matching_job = True
        return True
    if depth > 0:
        for path in job._sis_inputs:
            if path.creator:
                if _is_matching_job(path.creator, depth=depth - 1):
                    return True
    job._claix_is_matching_job = False
    return False


def should_symlink_jobs() -> bool:
    """currently only for my specific user"""
    if "az668407" in os.environ.get("USER", ""):
        return True
    if "az668407" in os.environ.get("WORK", ""):
        return True  # work folder, this should be set even when not in interactive session
    return False


def setup_job_symlinks():
    """
    Create symlinks for all jobs with aliases matching the alias_pattern.
    Run this at the very end of your Sisyphus config.
    """

    if not should_symlink_jobs():
        print("Not symlinking jobs.")
        return

    # Collect all jobs.
    jobs = tk.sis_graph.jobs()

    for job in jobs:
        if _is_matching_job(job):
            job_dir = job._sis_path()
            # if "ReturnnTrainingJob" in job.__class__.__name__:
            #     print(f"{job}")
            if os.path.islink(job_dir):
                # print(f"({job}: Symlink to {os.readlink(job_dir)})")
                continue
            if os.path.exists(job_dir):
                content = os.listdir(job_dir)
                if content:
                    # print(f"({job}: Warning: Already exists, has content: {content})")
                    continue
                print(f"({job}: Already exists, but empty, thus removing)")
                os.rmdir(job_dir)

            assert job_dir[:1] != "/"  # relative path
            assert target_dir[:1] == "/"  # absolute path
            job_target_dir = target_dir + "/" + job_dir
            os.makedirs(job_target_dir, exist_ok=True)
            # make parent dirs but not dir itself
            parent_dir = os.path.dirname(job_dir)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            print(f"{job}: Create symlink to: {job_target_dir}")
            os.symlink(job_target_dir, job_dir, target_is_directory=True)
