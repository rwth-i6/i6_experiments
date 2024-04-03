from typing import Optional, Union, Tuple, TextIO
import os
import tarfile
from io import TextIOWrapper
from sisyphus import Job
from contextlib import contextmanager
from .job_dir import get_job_base_dir


@contextmanager
def open_job_log(job: Union[str, Job], task: str = "run", index: int = 1) -> Tuple[Optional[TextIO], Optional[str]]:
    """
    :param job: dir or Job object
    :param task:
    :param index:
    """
    job_dir = get_job_base_dir(job)
    log_base_fn = f"log.{task}.{index}"
    log_fn = f"{job_dir}/{log_base_fn}"
    if os.path.exists(log_fn):
        yield open(log_fn), log_fn
        return
    tar_fn = f"{job_dir}/finished.tar.gz"
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
