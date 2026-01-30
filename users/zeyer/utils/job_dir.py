from typing import Union
import os
from sisyphus import Job
from . import sis_path
from ..sis_tools.sis_common import get_job_from_work_output


def get_job_base_dir(job: Union[str, Job]) -> str:
    """
    :param job:
    :return: the base directory of the job, including "work/" prefix if applicable
    """
    if isinstance(job, Job):
        # noinspection PyProtectedMember
        job = job._sis_path()
        assert _is_job_dir(job), f"job not valid: {job}"
        return job
    elif job.startswith("work/") or job.startswith("/"):
        assert _is_job_dir(job), f"job dir not valid: {job}"
        return job
    elif os.path.isdir(job) and _is_job_dir(job):
        return job
    elif os.path.isfile(job):
        # might be an output of the job
        d = "work/" + get_job_from_work_output(os.path.realpath(job))
        assert _is_job_dir(d), f"job dir not valid: job {job}, work dir {d}"
        return d
    else:
        work_dir_prefix = sis_path.get_work_dir_prefix()
        d = work_dir_prefix + job
        assert _is_job_dir(d), f"job dir not valid: {d}, job {job}"
        return d


def _is_job_dir(d: str) -> bool:
    if not os.path.isdir(d):
        return False
    if not os.path.isfile(d + "/info"):
        return False
    if not os.path.isdir(d + "/output"):
        return False
    return True
