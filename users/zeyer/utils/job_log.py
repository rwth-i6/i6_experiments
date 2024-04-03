from typing import Optional, Union, Tuple, TextIO
import os
import tarfile
from io import TextIOWrapper
from sisyphus import Job
from contextlib import contextmanager
from . import sis_path


@contextmanager
def open_job_log(job: Union[str, Job], task: str = "run", index: int = 1) -> Tuple[Optional[TextIO], Optional[str]]:
    """
    :param job: dir or Job object
    :param task:
    :param index:
    """
    if isinstance(job, Job):
        # noinspection PyProtectedMember
        job = job._sis_path()
        assert os.path.isdir(job), f"job not valid: {job}"
    elif job.startswith("work/") or job.startswith("/"):
        assert os.path.isdir(job), f"job dir not valid: {job}"
    else:
        work_dir_prefix = sis_path.get_work_dir_prefix()
        assert os.path.isdir(work_dir_prefix + job), f"job dir not valid: job {job}, work dir prefix {work_dir_prefix}"
        job = work_dir_prefix + job
    log_base_fn = f"log.{task}.{index}"
    log_fn = f"{job}/{log_base_fn}"
    if os.path.exists(log_fn):
        yield open(log_fn), log_fn
        return
    tar_fn = f"{job}/finished.tar.gz"
    if os.path.exists(tar_fn):
        with tarfile.open(tar_fn) as tarf:
            while True:
                f = tarf.next()
                if not f:
                    break
                if f.name == log_base_fn:
                    f = tarf.extractfile(f)
                    yield TextIOWrapper(f), f"{tar_fn}:{log_base_fn}"
                    return
    yield None, None
