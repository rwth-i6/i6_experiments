from typing import Union
import os
from sisyphus import Job
from . import sis_path


def get_job_base_dir(job: Union[str, Job]) -> str:
    """
    :param job:
    :return: the base directory of the job
    """
    if isinstance(job, Job):
        # noinspection PyProtectedMember
        job = job._sis_path()
        assert os.path.isdir(job), f"job not valid: {job}"
        return job
    elif job.startswith("work/") or job.startswith("/"):
        assert os.path.isdir(job), f"job dir not valid: {job}"
        return job
    else:
        work_dir_prefix = sis_path.get_work_dir_prefix()
        assert os.path.isdir(work_dir_prefix + job), f"job dir not valid: job {job}, work dir prefix {work_dir_prefix}"
        return work_dir_prefix + job
